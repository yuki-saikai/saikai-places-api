# main.py --- v0.1 (simple)
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
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


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    # 依存が満たされているかの簡易チェック
    ok = os.environ.get("GMP_API_KEY") is not None
    return {"ready": ok, "missing": [] if ok else ["GMP_API_KEY"]}


@app.get("/plan")
def plan(
    station_name: str,
    category: str = "restaurant",
    radius_m: int = 800,
    language: str = "ja",
    region: str = "JP",
    max_results: int = 5,
):
    """
    使い方（例）
    GET /plan?station_name=渋谷駅&category=restaurant&radius_m=1200&max_results=5
    """
    try:
        gm = gmaps()

        # 1) 駅名→座標（Geocoding）
        geo = gm.geocode(station_name, language=language, region=region)
        if not geo:
            raise HTTPException(status_code=404, detail="station not found")
        loc = geo[0]["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]

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
            "station": {"name": station_name, "lat": lat, "lng": lng},
            "category": category,
            "radius_m": radius_m,
            "spots": spots,
        }

    except HTTPException:
        raise
    except Exception as e:
        # 生の例外は出さず、型だけ返す（ログ汚染を防ぐ）
        raise HTTPException(
            status_code=500, detail=f"internal error: {type(e).__name__}"
        )
