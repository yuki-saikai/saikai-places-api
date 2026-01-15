import os
import json
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from google.cloud import secretmanager
from google.cloud import aiplatform
import googlemaps

# ---------------------------
# モデル: 入出力バリデーション
# ---------------------------

TravelMode = Literal["walking", "transit", "driving"]


class SearchOptions(BaseModel):
    category: str = Field(default="restaurant", description="検索カテゴリ")
    radius_m: int = Field(default=1200, ge=100, le=5000)
    max_results: int = Field(default=5, ge=1, le=10)
    travel_mode: TravelMode = Field(default="walking")


class Filters(BaseModel):
    open_required: bool = True
    min_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    price_levels: Optional[List[int]] = Field(default=None)  # 0..4


class Locale(BaseModel):
    language: str = "ja"
    region: str = "JP"


class PlanRequest(BaseModel):
    station_name: str
    meeting_time: str  # ISO8601
    search: SearchOptions = SearchOptions()
    filters: Filters = Filters()
    locale: Locale = Locale()

    @field_validator("meeting_time")
    @classmethod
    def check_isoformat(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
            return v
        except Exception:
            raise ValueError("meeting_time must be ISO8601 format")


# ---------------------------
# 定数・ユーティリティ
# ---------------------------

CATEGORY_MAP = {
    # Places Nearby の type として解釈
    "restaurant": "restaurant",
    "cafe": "cafe",
    "bar": "bar",
    "store": "store",  # 小売一般
    "shopping_mall": "shopping_mall",
    "tourist_attraction": "tourist_attraction",
    "lodging": "lodging",
    # イベントは「textsearch」を使う想定（拡張）
    "event": None,
}


def get_secret(secret_name: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.environ["PROJECT_ID"]
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")


# ---------------------------
# 初期化（GMP / Gemini）
# ---------------------------

app = FastAPI(title="places-orchestrator", version="1.0.0")

PROJECT_ID = os.environ["PROJECT_ID"]
LOCATION = os.environ.get("GCP_REGION", "asia-northeast1")

GMP_API_KEY = get_secret("gmp-api-key")
gmaps = googlemaps.Client(key=GMP_API_KEY)

aiplatform.init(project=PROJECT_ID, location=LOCATION)
# Pro: 品質重視／ Flash: コスト・レイテンシ重視
GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-1.5-pro")
gemini = aiplatform.GenerativeModel(GEMINI_MODEL_ID)

# ---------------------------
# エンドポイント
# ---------------------------


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "places-orchestrator"}


@app.post("/plan")
def plan(req: PlanRequest):
    try:
        # 1) Geocoding
        geo = gmaps.geocode(
            req.station_name, language=req.locale.language, region=req.locale.region
        )
        if not geo:
            raise HTTPException(status_code=404, detail="Station not found")
        origin = geo[0]["geometry"]["location"]
        origin_lat, origin_lng = origin["lat"], origin["lng"]

        # 2) 周辺検索（カテゴリによってNearby/Textを切替）
        resolved_type = CATEGORY_MAP.get(req.search.category, None)
        if req.search.category == "event":
            # 仮：イベントは Text Search（キーワード検索）に倒す例
            text_query = f"{req.station_name} 周辺 イベント {req.search.radius_m}m"
            nearby = gmaps.places(
                query=text_query,
                language=req.locale.language,
                location=(origin_lat, origin_lng),
            )
            candidates = nearby.get("results", [])
        else:
            nearby = gmaps.places_nearby(
                location=(origin_lat, origin_lng),
                radius=req.search.radius_m,
                type=resolved_type,
                language=req.locale.language,
            )
            candidates = nearby.get("results", [])

        # 3) 事前フィルタリング（rating/price/opening）
        filtered = []
        for p in candidates:
            rating = p.get("rating", 0)
            user_ratings_total = p.get("user_ratings_total", 0)
            price_level = p.get("price_level", None)
            # open_now は Nearby のレスポンスに入る場合がある（常にではない）
            if req.filters.min_rating and rating < req.filters.min_rating:
                continue
            if req.filters.price_levels and (
                price_level not in req.filters.price_levels
            ):
                continue
            filtered.append(p)

        top = filtered[: req.search.max_results]

        # 4) 詳細取得（Details）
        details = []
        fields = [
            "place_id",
            "name",
            "geometry",
            "opening_hours",
            "rating",
            "user_ratings_total",
            "price_level",
            "photos",
        ]
        for p in top:
            pid = p["place_id"]
            d = gmaps.place(place_id=pid, language=req.locale.language, fields=fields)
            details.append(d)

        if not details:
            return {
                "spots": [],
                "rationale": "条件に合致する候補が見つかりませんでした。",
                "route": {
                    "mode": req.search.travel_mode.upper(),
                    "distance_meters": 0,
                    "duration_minutes": 0,
                },
                "meta": {
                    "station_geocoded": {"lat": origin_lat, "lng": origin_lng},
                    "meeting_time": req.meeting_time,
                    "category_resolved": req.search.category,
                    "data_sources": ["GMP:Geocoding", "GMP:Places"],
                },
            }

        # 5) Distance Matrix（到着可否の根拠）
        arrival_epoch = int(datetime.fromisoformat(req.meeting_time).timestamp())
        destinations = [
            f"{d['result']['geometry']['location']['lat']},{d['result']['geometry']['location']['lng']}"
            for d in details
        ]
        mode_map = {"walking": "walking", "transit": "transit", "driving": "driving"}
        dm = gmaps.distance_matrix(
            origins=[f"{origin_lat},{origin_lng}"],
            destinations=destinations,
            mode=mode_map.get(req.search.travel_mode, "walking"),
            arrival_time=arrival_epoch,
            language=req.locale.language,
        )

        # 6) Gemini で最終整形（JSONスキーマ強制）
        prompt = f"""
最寄り駅: {req.station_name} (lat={origin_lat}, lng={origin_lng})
集合日時: {req.meeting_time}
検索カテゴリ（解釈）: {req.search.category}
Filters: rating>={req.filters.min_rating}, price in {req.filters.price_levels}, open_required={req.filters.open_required}

候補スポット（GMP Place Details; 上限{req.search.max_results}件）:
{json.dumps(details, ensure_ascii=False)}

距離・所要時間（GMP Distance Matrix）:
{json.dumps(dm, ensure_ascii=False)}

要件:
- 集合時刻に間に合うか（arrive_in_time）を Distance Matrix に基づいて判断
- open_required=true の場合は、opening_hours から営業中/外を判断（不明なら推定理由をreasonsへ）
- 出力は JSON のみ。以下スキーマを厳密に満たすこと
スキーマ:
{{
  "spots":[
    {{
      "name": "string",
      "place_id": "string",
      "location": {{"lat": 0, "lng": 0}},
      "arrive_in_time": true,
      "eta_minutes": 0,
      "open_status": "OPEN|CLOSED|UNKNOWN",
      "rating": 0.0,
      "user_ratings_total": 0,
      "price_level": 0,
      "photos": [{{"photo_reference": "string", "width": 0, "height": 0}}],
      "reasons": ["string"]
    }}
  ],
  "rationale": "string",
  "route": {{
    "mode": "{req.search.travel_mode.upper()}",
    "polyline": "",
    "distance_meters": 0,
    "duration_minutes": 0
  }}
}}
        """

        result = gemini.generate_content(prompt, response_mime_type="application/json")

        # Gemini 応答は text/候補のどちらか。ここでは text を想定
        text = getattr(result, "text", None)
        if not text:
            # 互換：candidates[0].content.parts[0].text のような場合に対応
            try:
                text = result.candidates[0].content.parts[0].text  # type: ignore
            except Exception:
                raise HTTPException(
                    status_code=500, detail="Gemini response parse error"
                )

        payload = json.loads(text)

        # 7) 返却（メタ情報を付加）
        payload.setdefault("meta", {})
        payload["meta"].update(
            {
                "station_geocoded": {"lat": origin_lat, "lng": origin_lng},
                "meeting_time": req.meeting_time,
                "category_resolved": req.search.category,
                "data_sources": [
                    "GMP:Geocoding",
                    "GMP:Places",
                    "GMP:Details",
                    "GMP:DistanceMatrix",
                    "Gemini",
                ],
            }
        )
        return payload

    except HTTPException:
        raise
    except Exception as e:
        # 例外はメッセージ最小化（座標などの生値をログに残さない設計）
        raise HTTPException(
            status_code=500, detail=f"internal error: {type(e).__name__}"
        )
