# main.py --- v0.6 (Gemini 3.0 support)
# Gemini 3.0 Key Features:
# - thinking_level: Controls reasoning depth (low for speed, high for complex tasks)
# - temperature: Keep at 1.0 (default) for Gemini 3.0 models
# - thought_signatures: Automatically handled by SDK
import os
import statistics
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import googlemaps
from google import genai
from google.genai import types
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
    restaurant_type: str = (
        "restaurant"  # Google Places API type (restaurant, cafe, etc.)
    )


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


# ---- Gemini Client ----
_gemini_client: Optional[genai.Client] = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


# ---- Helpers ----
def _normalize_station_names(values: List[str]) -> List[str]:
    names: List[str] = []
    for v in values:
        for part in v.split(","):
            name = part.strip()
            if name and name not in names:
                names.append(name)
    return names


def _match_restaurant_by_name(
    gemini_name: str, google_restaurants: List[Dict]
) -> Optional[Dict]:
    """
    Match a restaurant name from Gemini response with Google Maps data.
    Uses exact match first, then falls back to partial match.

    Args:
        gemini_name: Restaurant name from Gemini response
        google_restaurants: List of restaurants from Google Maps API

    Returns:
        Matched restaurant data or None if no match found
    """
    # Step 1: Exact match (case-sensitive)
    for restaurant in google_restaurants:
        if restaurant["name"] == gemini_name:
            return restaurant

    # Step 2: Exact match (case-insensitive)
    gemini_name_lower = gemini_name.lower()
    for restaurant in google_restaurants:
        if restaurant["name"].lower() == gemini_name_lower:
            return restaurant

    # Step 3: Partial match (contains)
    for restaurant in google_restaurants:
        # Check if Gemini name is contained in Google name
        if gemini_name in restaurant["name"] or restaurant["name"] in gemini_name:
            return restaurant
        # Case-insensitive partial match
        if (
            gemini_name_lower in restaurant["name"].lower()
            or restaurant["name"].lower() in gemini_name_lower
        ):
            return restaurant

    # No match found
    return None


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


def _get_station_location(
    gm: googlemaps.Client, station_name: str, region: str
) -> Optional[Dict]:
    """
    Get latitude and longitude for a station.
    """
    try:
        geocode_result = gm.geocode(station_name, region=region)
        if geocode_result:
            location = geocode_result[0]["geometry"]["location"]
            return {"lat": location["lat"], "lng": location["lng"]}
    except Exception as e:
        print(f"Error geocoding {station_name}: {e}")
    return None


def _search_nearby_restaurants(
    gm: googlemaps.Client,
    location: Dict,
    place_type: str = "restaurant",
    radius: int = 500,
    language: str = "ja",
) -> List[Dict]:
    """
    Search for restaurants within radius using Places API.
    Returns up to 20 results (prominence order).

    Supported place_type values (Google Places API - Table 1):
    - restaurant (default)
    - cafe
    - bar
    - bakery
    - meal_delivery
    - meal_takeaway
    - night_club
    - food (general category)

    Other common types:
    - tourist_attraction
    - park
    - museum
    - shopping_mall
    - amusement_park
    - aquarium
    - art_gallery
    - bowling_alley
    - movie_theater
    - spa
    - gym
    - library

    For full list, see: https://developers.google.com/maps/documentation/places/web-service/supported_types
    """
    try:
        print(
            f"🔍 Searching restaurants near location: lat={location['lat']}, lng={location['lng']}, radius={radius}m"
        )
        places_result = gm.places_nearby(
            location=(location["lat"], location["lng"]),
            radius=radius,
            type=place_type,
            language=language,
        )

        restaurants = []
        for place in places_result.get("results", []):
            # Skip permanently closed businesses
            if place.get("business_status") == "CLOSED_PERMANENTLY":
                continue

            # Build photo URL if available
            photo_url = None
            if place.get("photos"):
                photo_ref = place["photos"][0].get("photo_reference")
                if photo_ref:
                    photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference={photo_ref}&key={os.environ.get('GMP_API_KEY')}"

            # Build Google Maps URL
            place_id = place.get("place_id")
            maps_url = (
                f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                if place_id
                else None
            )

            restaurants.append(
                {
                    "name": place.get("name"),
                    "rating": place.get("rating"),
                    "photo_url": photo_url,
                    "maps_url": maps_url,
                }
            )

        print(f"✅ Found {len(restaurants)} restaurants")
        return restaurants

    except Exception as e:
        print(f"❌ Error searching restaurants: {e}")
        return []


def _load_prompt_template() -> str:
    """
    Load prompt template from file.
    """
    try:
        prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "recommend_restaurants.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading prompt template: {e}")
        raise HTTPException(status_code=500, detail="Failed to load prompt template")


def _parse_json_from_response(response_text: str) -> Optional[Dict]:
    """
    Extract and parse JSON from Gemini response.
    Handles cases where JSON might be wrapped in markdown code blocks.
    """
    try:
        # Try direct parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_pattern = r"```(?:json)?\s*({.*?})\s*```"
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object directly in text
        json_pattern = r"{.*}"
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return None


def _get_gemini_recommendations(
    station_name: str, restaurants: List[Dict]
) -> List[Dict]:
    """
    Get restaurant recommendations from Gemini API and merge with Google Maps data.

    Args:
        station_name: Station name
        restaurants: List of restaurants from Google Maps API with full data

    Returns:
        List of integrated recommendations with Gemini insights + Google Maps data
    """
    if not restaurants:
        print(f"⚠️  No restaurants available for {station_name}")
        return []

    try:
        print(
            f"\n🤖 Calling Gemini API for {station_name} with {len(restaurants)} restaurants..."
        )
        client = get_gemini_client()
        prompt_template = _load_prompt_template()

        # Build restaurant list string
        restaurant_list = "\n".join(
            [f"- {r['name']} (評価: {r.get('rating', 'N/A')})" for r in restaurants]
        )

        # Fill in template
        prompt = prompt_template.replace("{station_name}", station_name).replace(
            "{restaurant_list}", restaurant_list
        )

        # Call Gemini API with new client structure and Gemini 3.0 model
        response = client.models.generate_content(
            model="gemini-3-flash-preview",  # Gemini 3.0 Flash model
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"  # Low thinking level for faster, lower-cost responses
                ),
                temperature=1.0,  # Gemini 3 recommends keeping temperature at default 1.0
            ),
        )

        if not response or not response.text:
            print(f"❌ Gemini API returned empty response for {station_name}")
            return []

        print(f"✅ Gemini API response received ({len(response.text)} chars)")

        # Parse JSON from response
        parsed_data = _parse_json_from_response(response.text)

        if not parsed_data or "recommendations" not in parsed_data:
            print(f"❌ Failed to parse Gemini response for {station_name}")
            return []

        print(
            f"📝 Parsed {len(parsed_data.get('recommendations', []))} recommendations from Gemini"
        )

        # Integrate Gemini recommendations with Google Maps data
        integrated_recommendations = []

        print(f"🔗 Starting data integration...")
        for gemini_rec in parsed_data["recommendations"]:
            gemini_name = gemini_rec.get("name", "")

            # Match with Google Maps data
            google_data = _match_restaurant_by_name(gemini_name, restaurants)

            if google_data:
                match_type = (
                    "exact_match"
                    if google_data["name"] == gemini_name
                    else "partial_match"
                )
                print(f"  ✓ '{gemini_name}' → '{google_data['name']}' ({match_type})")
                # Successfully matched - merge data
                integrated_recommendations.append(
                    {
                        "name": google_data["name"],  # Use Google Maps name (canonical)
                        "reason": gemini_rec.get("reason", ""),
                        "recommended_menu": gemini_rec.get("recommended_menu", ""),
                        "rating": google_data.get("rating"),
                        "photo_url": google_data.get("photo_url"),
                        "maps_url": google_data.get("maps_url"),
                        "match_status": match_type,
                    }
                )
            else:
                # No match found - include Gemini data only with warning
                print(f"  ✗ '{gemini_name}' → No match found (using Gemini data only)")
                integrated_recommendations.append(
                    {
                        "name": gemini_name,
                        "reason": gemini_rec.get("reason", ""),
                        "recommended_menu": gemini_rec.get("recommended_menu", ""),
                        "rating": None,
                        "photo_url": None,
                        "maps_url": None,
                        "match_status": "no_match",
                    }
                )

        print(
            f"✅ Integration complete: {len(integrated_recommendations)} recommendations"
        )
        return integrated_recommendations

    except Exception as e:
        print(f"Error calling Gemini API for {station_name}: {e}")
        return []


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

        # Find the station with max duration
        max_duration_station = None
        if duration_map:
            max_duration_station = max(duration_map, key=duration_map.get)

        terminal_data.append(
            {
                "station": terminal,
                "variance": variance,
                "avg_duration_minutes": round(
                    (sum(durations) / len(durations) / 60) if durations else 0
                ),
                "max_duration_minutes": round(max(durations) / 60 if durations else 0),
                "max_duration_from_station": max_duration_station,
                "durations": duration_map,
                "route_count": len(durations),
            }
        )

    terminal_data.sort(key=lambda x: x["avg_duration_minutes"])
    print(f"\n✅ Found {len(terminal_data)} candidate stations, returning top {top_n}")
    for i, station in enumerate(terminal_data[:top_n], 1):
        print(
            f"  {i}. {station['station']}: avg={station['avg_duration_minutes']}min, max={station['max_duration_minutes']}min (from {station['max_duration_from_station']})"
        )
    return terminal_data[:top_n]


# ---- Endpoints ----
@app.get("/")
def root():
    return {"status": "running", "service": "saikai-places-api"}


@app.get("/models")
def list_models():
    """
    List all available Gemini models with their version information.
    Filters to show Gemini models and highlights Gemini 3.0 models.
    """
    try:
        client = get_gemini_client()
        models = []
        gemini_3_models = []

        for model in client.models.list():
            model_info = {
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
                "supported_actions": model.supported_actions,
            }

            # Separate Gemini 3.0 models
            if "gemini-3" in model.name:
                gemini_3_models.append(model_info)
            else:
                models.append(model_info)

        return {
            "gemini_3_models": gemini_3_models,
            "other_models": models,
            "recommended": "gemini-3-flash-preview",
            "note": "Gemini 3.0 models support thinking_level parameter for controlling reasoning depth",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")


@app.post("/plan")
def plan(request: PlanRequest = Body(...)):
    """
    Find optimal middle stations and get restaurant recommendations.
    Returns integrated recommendations with Gemini insights + Google Maps data.
    """
    print(f"\n{'='*70}")
    print(f"🚀 Starting /plan request")
    print(f"{'='*70}")

    stations = _normalize_station_names(request.station_names)
    print(f"📍 Input stations: {stations}")
    if len(stations) < 2:
        raise HTTPException(status_code=400, detail="At least 2 stations required")

    gm = get_gmaps()
    results = _find_middle_stations(
        gm, stations, TERMINAL_STATIONS, request.language, request.region
    )

    if not results:
        print(f"❌ No suitable middle stations found")
        raise HTTPException(status_code=404, detail="No suitable middle stations found")

    # Build response with integrated recommendations
    print(f"\n📦 Building response for {len(results)} candidate stations...\n")
    response_data = {"input_stations": stations, "candidate_stations": []}

    for idx, station_data in enumerate(results, 1):
        station_name = station_data["station"]
        print(f"\n{'-'*70}")
        print(f"🎯 Processing candidate {idx}/{len(results)}: {station_name}")
        print(f"{'-'*70}")

        location = _get_station_location(gm, station_name, request.region)

        candidate_info = {
            "station": station_name,
            "avg_duration_minutes": station_data["avg_duration_minutes"],
            "max_duration_minutes": station_data["max_duration_minutes"],
            "max_duration_from_station": station_data["max_duration_from_station"],
            "recommendations": [],
        }

        if location:
            print(f"📍 Location: lat={location['lat']:.6f}, lng={location['lng']:.6f}")
            restaurants = _search_nearby_restaurants(
                gm, location, request.restaurant_type, language=request.language
            )

            # Get integrated recommendations (Gemini + Google Maps data)
            integrated_recommendations = _get_gemini_recommendations(
                station_name, restaurants
            )
            candidate_info["recommendations"] = integrated_recommendations
            candidate_info["total_restaurants_found"] = len(restaurants)
            print(
                f"✅ Completed {station_name}: {len(integrated_recommendations)} recommendations"
            )
        else:
            print(f"❌ Failed to get location for {station_name}")
            candidate_info["error"] = "Failed to get station location"

        response_data["candidate_stations"].append(candidate_info)

    print(f"\n{'='*70}")
    print(
        f"🎉 Request complete! Returning {len(response_data['candidate_stations'])} stations"
    )
    print(f"{'='*70}\n")
    return response_data
