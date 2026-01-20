# main.py --- v0.3 (simplified)
import os
import statistics
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import googlemaps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="saikai-places-api", version="0.3.0")

# ---- Configuration ----
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


# ---- Models ----
class PlanRequest(BaseModel):
    station_names: List[str]
    language: str = "ja"
    region: str = "JP"


# ---- Google Maps Client ----
_gmaps: Optional[googlemaps.Client] = None


def get_gmaps() -> googlemaps.Client:
    global _gmaps
    if _gmaps is None:
        key = os.environ.get("GMP_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="GMP_API_KEY is not set")
        _gmaps = googlemaps.Client(key=key)
    return _gmaps


# ---- Helpers ----
def _normalize_station_names(values: List[str]) -> List[str]:
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
    Get duration in seconds. Tries TRANSIT first, falls back to DRIVING if transit fails (e.g., date issues).
    """
    try:
        # Try Transit
        routes = gm.directions(
            origin=origin,
            destination=destination,
            mode="transit",
            language=language,
            region=region,
            departure_time=datetime.now(),
            alternatives=False,
        )

        # Fallback to Driving if no transit routes found (common with future dates/no schedule data)
        if not routes:
            print(f"  Fallback: Driving mode for {origin} -> {destination}")
            routes = gm.directions(
                origin=origin,
                destination=destination,
                mode="driving",
                language=language,
                region=region,
            )

        if not routes:
            return None

        # Return duration of the first leg
        return routes[0].get("legs", [{}])[0].get("duration", {}).get("value")

    except Exception as e:
        print(f"Error fetching route {origin}->{destination}: {e}")
        return None


def _calculate_variance(durations: List[int]) -> float:
    if len(durations) < 2:
        return 0.0
    try:
        return statistics.variance(durations)
    except:
        return float("inf")


def _find_middle_stations(
    gm: googlemaps.Client,
    input_stations: List[str],
    terminal_stations: List[str],
    language: str,
    region: str,
    top_n: int = 3,
) -> List[Dict]:
    terminal_data = []
    min_routes = max(2, len(input_stations) // 2)

    for terminal in terminal_stations:
        durations = []
        duration_map = {}

        for station in input_stations:
            seconds = _get_transit_duration_seconds(
                gm, station, terminal, language, region
            )
            if seconds is not None:
                durations.append(seconds)
                duration_map[station] = seconds

        if len(durations) < min_routes:
            continue

        # Calculate score (variance + penalty for missing stations)
        variance = _calculate_variance(durations) if len(durations) > 1 else 0
        if len(durations) < len(input_stations):
            variance += (len(input_stations) - len(durations)) * 10000

        terminal_data.append(
            {
                "station": terminal,
                "variance": variance,
                "avg_duration_minutes": (
                    (sum(durations) / len(durations) / 60) if durations else 0
                ),
                "durations": duration_map,
                "route_count": len(durations),
            }
        )

    terminal_data.sort(key=lambda x: x["variance"])
    return terminal_data[:top_n]


# ---- Endpoints ----
@app.get("/")
def root():
    return {"status": "running", "service": "saikai-places-api"}


@app.post("/plan")
def plan(request: PlanRequest = Body(...)):
    stations = _normalize_station_names(request.station_names)
    if len(stations) < 2:
        raise HTTPException(status_code=400, detail="At least 2 stations required")

    gm = get_gmaps()
    results = _find_middle_stations(
        gm, stations, TERMINAL_STATIONS, request.language, request.region
    )

    if not results:
        raise HTTPException(status_code=404, detail="No suitable middle stations found")

    return {"input_stations": stations, "middle_stations": results}
