# main.py --- v0.2 (middle station finder)
import os
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import googlemaps
from googlemaps import exceptions as gmaps_exceptions

app = FastAPI(title="saikai-places-api", version="0.2.0")

# ---- ターミナル駅リスト ----
TERMINAL_STATIONS = [
    "新宿駅",
    "渋谷駅",
    "池袋駅",
    "有楽町駅",
    "銀座駅",
    "表参道駅",
    "六本木駅",
    "上野駅",
    "東京駅",
    "品川駅",
    "横浜駅",
    "大宮駅",
    "立川駅",
    "秋葉原駅",
    "北千住駅",
    "赤羽駅",
    "川崎駅",
]


# ---- リクエストボディモデル ----
class PlanRequest(BaseModel):
    station_names: List[str]
    language: str = "ja"
    region: str = "JP"


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
    """駅名のリストを正規化（カンマ区切り対応、重複除去）"""
    names: List[str] = []
    for v in values:
        for part in v.split(","):
            name = part.strip()
            if name and name not in names:
                names.append(name)
    return names


def _get_transit_duration_seconds(
    gm: googlemaps.Client,
    origin: str,
    destination: str,
    language: str,
    region: str,
) -> Optional[int]:
    """
    公共交通機関での所要時間（秒）を取得。
    経路が見つからない場合はNoneを返す。
    """
    try:
        # 公共交通機関では出発時刻の指定が重要（その日の朝9時を基準にする）
        today = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

        routes = gm.directions(
            origin=origin,
            destination=destination,
            mode="transit",
            language=language,
            region=region,
            departure_time=today,
            alternatives=False,
        )
        if not routes:
            return None

        duration = routes[0].get("legs", [{}])[0].get("duration", {}).get("value")
        return duration
    except gmaps_exceptions.ApiError as e:
        # APIエラーの詳細をログ出力（本番では適切なロギングに置き換える）
        print(f"API Error for {origin} -> {destination}: {e.status}")
        return None
    except Exception as e:
        print(f"Exception for {origin} -> {destination}: {type(e).__name__} - {str(e)}")
        return None


def _calculate_variance(durations: List[int]) -> float:
    """所要時間リストの分散を計算（値が小さいほど各駅から均等）"""
    if len(durations) < 2:
        return 0.0
    try:
        return statistics.variance(durations)
    except Exception:
        return float("inf")


def _find_middle_stations(
    gm: googlemaps.Client,
    input_stations: List[str],
    terminal_stations: List[str],
    language: str,
    region: str,
    top_n: int = 3,
) -> List[Dict]:
    """
    入力駅から各ターミナル駅への所要時間を計算し、
    分散が小さい（=各駅から均等にアクセスしやすい）上位N駅を返す。
    """
    terminal_data = []
    min_required_routes = max(
        2, len(input_stations) // 2
    )  # 最低でも半数以上の駅から経路が必要

    for terminal in terminal_stations:
        durations = []
        duration_details = {}

        for station in input_stations:
            duration = _get_transit_duration_seconds(
                gm, station, terminal, language, region
            )
            if duration is not None:
                durations.append(duration)
                duration_details[station] = duration

        # 最低限必要な経路数が見つからない場合はスキップ
        if len(durations) < min_required_routes:
            continue

        # 分散を計算（一部の駅からしかルートがない場合はペナルティ）
        if len(durations) < len(input_stations):
            # 欠損駅の数に応じてペナルティを加算
            base_variance = _calculate_variance(durations) if len(durations) > 1 else 0
            missing_penalty = (len(input_stations) - len(durations)) * 10000
            variance = base_variance + missing_penalty
        else:
            variance = _calculate_variance(durations)

        terminal_data.append(
            {
                "station": terminal,
                "variance": variance,
                "avg_duration_minutes": (
                    sum(durations) / len(durations) / 60 if durations else None
                ),
                "durations": duration_details,
                "route_count": len(durations),
                "total_stations": len(input_stations),
            }
        )

    # 分散が小さい順にソート
    terminal_data.sort(key=lambda x: x["variance"])

    return terminal_data[:top_n]


@app.post("/plan")
def plan(request: PlanRequest = Body(...)):
    """
    複数の駅から公共交通機関でアクセスしやすい中間地点となるターミナル駅を3駅抽出。

    使い方（例）
    POST /plan
    {
        "station_names": ["吉祥寺駅", "川崎駅"],
        "language": "ja",
        "region": "JP"
    }
    """
    try:
        gm = gmaps()

        stations = _normalize_station_names(request.station_names)
        if not stations:
            raise HTTPException(status_code=400, detail="station_names is required")

        if len(stations) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 stations are required to find middle stations",
            )

        # 中間駅を抽出
        middle_stations = _find_middle_stations(
            gm=gm,
            input_stations=stations,
            terminal_stations=TERMINAL_STATIONS,
            language=request.language,
            region=request.region,
            top_n=3,
        )

        if not middle_stations:
            raise HTTPException(
                status_code=404, detail="No suitable middle stations found"
            )

        return {
            "input_stations": stations,
            "middle_stations": [
                {
                    "station": ms["station"],
                    "avg_duration_minutes": (
                        round(ms["avg_duration_minutes"], 1)
                        if ms["avg_duration_minutes"]
                        else None
                    ),
                    "route_count": ms.get("route_count"),
                    "durations_seconds": ms.get("durations"),
                }
                for ms in middle_stations
            ],
        }

    except HTTPException:
        raise
    except gmaps_exceptions.ApiError as e:
        raise HTTPException(status_code=502, detail=f"gmaps api error: {e.status}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"internal error: {type(e).__name__}"
        )
