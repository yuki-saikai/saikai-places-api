# main.py --- v0.1 (simple)
import os
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Query
import googlemaps

app = FastAPI(title="saikai-places-api", version="0.1.0")

# ---- 環境変数（最低限） ----
GMP_API_KEY = os.environ.get("GMP_API_KEY")  # ★必須
if not GMP_API_KEY:
    # 起動は続ける（/healthzはOK）→ リクエスト時に明示エラーにする
    pass

# googlemaps クライアントは遅延生成（起動失敗を避ける）
_gmaps: Optional[googlemaps.Client] = None


def gmaps() -> googlemaps.Client:
    global _gmaps
    if _gmaps is None:
        key = os.environ.get("GMP_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="GMP_API_KEY is not set")
        _gmaps = googlemaps.Client(key=key)
    return _gmaps


def _normalize_station_names(values: List[str]) -> List[str]:
    names: List[str] = []
    for v in values:
        for part in v.split(","):
            name = part.strip()
            if name and name not in names:
                names.append(name)
    return names


def _to_lat_lng(geo_item: dict) -> Tuple[float, float]:
    loc = geo_item["geometry"]["location"]
    return loc["lat"], loc["lng"]


def _dedupe_candidates(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for it in items:
        pid = it.get("place_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(it)
    return out


@app.get("/plan")
def plan(
    station_name: List[str] = Query(...),
    category: str = "restaurant",
    radius_m: int = 800,
    language: str = "ja",
    region: str = "JP",
    max_results: int = 5,
):
    """
    使い方（例）
    GET /plan?station_name=渋谷駅&category=restaurant&radius_m=1200&max_results=5
    GET /plan?station_name=渋谷駅&station_name=新宿駅&radius_m=1200&max_results=5
    GET /plan?station_name=渋谷駅,新宿駅&radius_m=1200&max_results=5
    """
    try:
        gm = gmaps()

        stations = _normalize_station_names(station_name)
        if not stations:
            raise HTTPException(status_code=400, detail="station_name is required")

        # 単一駅は従来通り
        if len(stations) == 1:
            station = stations[0]

            # 1) 駅名→座標（Geocoding）
            geo = gm.geocode(station, language=language, region=region)
            if not geo:
                raise HTTPException(status_code=404, detail="station not found")
            lat, lng = _to_lat_lng(geo[0])

            # 2) 周辺検索（Nearby）
            nearby = gm.places_nearby(
                location=(lat, lng), radius=radius_m, type=category, language=language
            )
            results = (nearby.get("results") or [])[:max_results]

            # 3) 最小レスポンス（余計な情報は返さない）
            spots = []
            for r in results:
                spots.append(
                    {
                        "name": r.get("name"),
                        "place_id": r.get("place_id"),
                        "location": r.get("geometry", {}).get("location"),
                        "rating": r.get("rating"),
                        "user_ratings_total": r.get("user_ratings_total"),
                        "price_level": r.get("price_level"),
                        "open_now": (
                            r.get("opening_hours", {}).get("open_now")
                            if r.get("opening_hours")
                            else None
                        ),
                    }
                )

            return {
                "station": {"name": station, "lat": lat, "lng": lng},
                "category": category,
                "radius_m": radius_m,
                "spots": spots,
            }

        # ---- 複数駅: 中間駅（移動しやすい駅）を推定 ----
        origin_coords: List[Tuple[float, float]] = []
        origin_payload = []
        for s in stations:
            geo = gm.geocode(s, language=language, region=region)
            if not geo:
                raise HTTPException(status_code=404, detail=f"station not found: {s}")
            lat, lng = _to_lat_lng(geo[0])
            origin_coords.append((lat, lng))
            origin_payload.append({"name": s, "lat": lat, "lng": lng})

        # 1) 地理的中心
        centroid_lat = sum(c[0] for c in origin_coords) / len(origin_coords)
        centroid_lng = sum(c[1] for c in origin_coords) / len(origin_coords)

        # 2) 駅候補を広めに収集（各駅周辺 + 中央付近）
        candidate_results: List[dict] = []
        candidate_radius = max(radius_m, 8000)
        for lat, lng in origin_coords:
            res = gm.places_nearby(
                location=(lat, lng),
                radius=candidate_radius,
                type="train_station",
                language=language,
            )
            candidate_results.extend(res.get("results") or [])

        centroid_res = gm.places_nearby(
            location=(centroid_lat, centroid_lng),
            radius=candidate_radius,
            type="train_station",
            language=language,
        )
        candidate_results.extend(centroid_res.get("results") or [])

        candidates = _dedupe_candidates(candidate_results)[:40]
        if not candidates:
            raise HTTPException(
                status_code=404, detail="central station candidates not found"
            )

        dest_coords: List[Tuple[float, float]] = []
        for c in candidates:
            loc = c.get("geometry", {}).get("location") or {}
            dest_coords.append((loc.get("lat"), loc.get("lng")))

        # 3) 交通利便性（transitの所要時間）で最良候補を選ぶ
        matrix = gm.distance_matrix(
            origins=origin_coords,
            destinations=dest_coords,
            mode="transit",
            language=language,
            region=region,
        )

        best_idx = None
        best_max = None
        best_sum = None
        rows = matrix.get("rows") or []
        for j in range(len(dest_coords)):
            durations = []
            for i in range(len(origin_coords)):
                try:
                    elem = rows[i]["elements"][j]
                except Exception:
                    elem = None
                if not elem or elem.get("status") != "OK":
                    durations = []
                    break
                durations.append(elem["duration"]["value"])
            if not durations:
                continue

            max_d = max(durations)
            sum_d = sum(durations)
            if (
                best_idx is None
                or max_d < best_max
                or (max_d == best_max and sum_d < best_sum)
            ):
                best_idx = j
                best_max = max_d
                best_sum = sum_d

        if best_idx is None:
            # 交通所要時間が取れない場合は、地理的中心に最も近い駅を採用
            def _sqdist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
                return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

            centroid = (centroid_lat, centroid_lng)
            best_idx = min(
                range(len(dest_coords)), key=lambda i: _sqdist(dest_coords[i], centroid)
            )

        best = candidates[best_idx]
        best_loc = best.get("geometry", {}).get("location") or {}
        best_lat = best_loc.get("lat")
        best_lng = best_loc.get("lng")

        # 4) 中間駅の周辺駅を返す
        nearby = gm.places_nearby(
            location=(best_lat, best_lng),
            radius=radius_m,
            type="train_station",
            language=language,
        )
        results = (nearby.get("results") or [])[:max_results]

        stations_out = []
        for r in results:
            stations_out.append(
                {
                    "name": r.get("name"),
                    "place_id": r.get("place_id"),
                    "location": r.get("geometry", {}).get("location"),
                    "rating": r.get("rating"),
                    "user_ratings_total": r.get("user_ratings_total"),
                }
            )

        return {
            "input_stations": origin_payload,
            "central_station": {
                "name": best.get("name"),
                "place_id": best.get("place_id"),
                "lat": best_lat,
                "lng": best_lng,
            },
            "radius_m": radius_m,
            "nearby_stations": stations_out,
        }

    except HTTPException:
        raise
    except Exception as e:
        # 生の例外は出さず、型だけ返す（ログ汚染を防ぐ）
        raise HTTPException(
            status_code=500, detail=f"internal error: {type(e).__name__}"
        )
