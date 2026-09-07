from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
from datetime import datetime, timedelta, timezone
import uuid
import math
import asyncio

app = FastAPI(title="Beavies Surf Forecast API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (replace with DB in production)
SPOTS: Dict[str, Dict[str, Any]] = {}
SESSIONS: Dict[str, Dict[str, Any]] = {}
RATINGS: List[Dict[str, Any]] = []
USER_PREFS: Dict[str, Dict[str, Any]] = {}

# Models
class Spot(BaseModel):
    id: Optional[str] = None
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    region: Optional[str] = None
    country: Optional[str] = None
    orientation: Optional[str] = Field(None, description="Cardinal orientation, e.g., W, SW")
    notes: Optional[str] = None

class CreateSpot(BaseModel):
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    region: Optional[str] = None
    country: Optional[str] = None
    orientation: Optional[str] = None
    notes: Optional[str] = None

class UpdateSpot(BaseModel):
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: Optional[str] = None
    country: Optional[str] = None
    orientation: Optional[str] = None
    notes: Optional[str] = None

class SessionCreate(BaseModel):
    user_id: str
    spot_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

class RatingCreate(BaseModel):
    user_id: str
    spot_id: str
    session_id: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    notes: Optional[str] = None

class ForecastRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    hourly: Optional[List[str]] = None

# Utility functions
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
DEFAULT_HOURLY = [
    "wave_height", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "temperature_2m", "precipitation", "swell_wave_height", "swell_wave_direction",
    "swell_wave_period", "wind_wave_height", "wind_wave_direction", "wind_wave_period"
]

MARINE_BASE = "https://marine-api.open-meteo.com/v1/marine"
MARINE_FIELDS = {"wave_height", "wave_direction", "wave_period", "swell_wave_height",
                 "swell_wave_direction", "swell_wave_period", "wind_wave_height",
                 "wind_wave_direction", "wind_wave_period"}
WEATHER_FIELDS = {"wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                  "temperature_2m", "precipitation"}

async def fetch_open_meteo(lat: float, lon: float, hourly: Optional[List[str]] = None):
    """Fetch weather and marine forecasts separately, then align by UTC timestamp."""
    if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(422, detail="Coordinates out of range")
    fields = list(dict.fromkeys(hourly or DEFAULT_HOURLY))
    if not fields or set(fields) - (MARINE_FIELDS | WEATHER_FIELDS):
        raise HTTPException(422, detail="Unsupported hourly variable")
    async with httpx.AsyncClient(timeout=20) as client:
        async def fetch_group(base, selected):
            params = {"latitude": lat, "longitude": lon, "hourly": ",".join(selected), "timezone": "GMT"}
            if base == OPEN_METEO_BASE:
                params["wind_speed_unit"] = "ms"
            try:
                response = await client.get(base, params=params)
                response.raise_for_status()
                data = response.json()
                rows = data["hourly"]
                times = rows["time"]
                if not isinstance(times, list) or not times or len(times) != len(set(times)):
                    raise ValueError("Invalid timestamps")
                for value in times:
                    datetime.fromisoformat(value)
                for field in selected:
                    if not isinstance(rows.get(field), list) or len(rows[field]) != len(times):
                        raise ValueError("Misaligned forecast")
                    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)) for v in rows[field]):
                        raise ValueError("Invalid observation")
                return data
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise HTTPException(502, detail="Forecast provider unavailable or returned invalid data")
        groups = [(MARINE_BASE, [f for f in fields if f in MARINE_FIELDS]),
                  (OPEN_METEO_BASE, [f for f in fields if f in WEATHER_FIELDS])]
        results = await asyncio.gather(*(fetch_group(base, selected) for base, selected in groups if selected))
    # Union keeps missing provider hours explicitly null rather than shifting observations.
    times = sorted(set(t for data in results for t in data["hourly"]["time"]))
    merged = {"time": times}
    units = {"time": "iso8601"}
    for data in results:
        rows = data["hourly"]
        units.update(data.get("hourly_units", {}))
        for field, values in rows.items():
            if field == "time":
                continue
            lookup = dict(zip(rows["time"], values))
            merged[field] = [lookup.get(t) for t in times]
    return {"latitude": lat, "longitude": lon, "timezone": "GMT", "utc_offset_seconds": 0,
            "hourly": merged, "hourly_units": units,
            "source": "Open-Meteo weather and marine forecast models",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data_kind": "forecast", "attribution_url": "https://open-meteo.com/"}

# Simple personalization/ML heuristic
# Computes a surf score 0-100 based on conditions and user preferences
# Prefs: preferred_wave_min/max (m), preferred_period_min/max (s), max_wind (m/s), preferred_wind_dir (deg or list)

def circular_diff(a: float, b: float) -> float:
    d = abs((a - b + 180) % 360 - 180)
    return d

def score_conditions(hour: Dict[str, Any], prefs: Dict[str, Any]) -> float:
    wave = hour.get("swell_wave_height")
    if wave is None:
        wave = hour.get("wave_height")
    period = hour.get("swell_wave_period")
    wind = hour.get("wind_speed_10m")
    wind_dir = hour.get("wind_direction_10m")

    score = 0.0
    weights = {"wave": 0.45, "period": 0.25, "wind": 0.2, "wind_dir": 0.1}

    # Wave height
    if wave is not None:
        wmin = prefs.get("preferred_wave_min", 0.6)
        wmax = prefs.get("preferred_wave_max", 2.5)
        if wave < wmin:
            score_wave = max(0, 1 - (wmin - wave) / max(0.01, wmin)) * 50
        elif wave > wmax:
            score_wave = max(0, 1 - (wave - wmax) / max(0.01, wmax)) * 65
        else:
            score_wave = 100
        score += score_wave * weights["wave"] / 100

    # Period
    if period is not None:
        pmin = prefs.get("preferred_period_min", 8)
        pmax = prefs.get("preferred_period_max", 18)
        if pmin <= period <= pmax:
            score_period = 100
        else:
            delta = min(abs(period - pmin), abs(period - pmax))
            score_period = max(0, 100 - delta * 8)
        score += score_period * weights["period"] / 100

    # Wind speed (lower is better up to a point)
    if wind is not None:
        max_wind = prefs.get("max_wind", 8)
        if wind <= max_wind:
            score_wind = 100 - (wind / max(0.01, max_wind)) * 40
        else:
            score_wind = max(0, 80 - (wind - max_wind) * 8)
        score += score_wind * weights["wind"] / 100

    # Wind direction (prefer offshore) if user provided a preferred direction
    pref_dir = prefs.get("preferred_wind_dir")
    if pref_dir is not None and wind_dir is not None:
        if isinstance(pref_dir, list):
            diff = min(circular_diff(wind_dir, d) for d in pref_dir)
        else:
            diff = circular_diff(wind_dir, pref_dir)
        score_dir = max(0, 100 - (diff / 180) * 100)
        score += score_dir * weights["wind_dir"] / 100

    return round(score * 100, 2)  # rescale to 0-100


def build_hour_objects(open_meteo: Dict[str, Any]) -> List[Dict[str, Any]]:
    hourly = open_meteo.get("hourly", {})
    times = hourly.get("time", [])
    result = []
    for i, t in enumerate(times):
        obj = {"time": t}
        for k, v in hourly.items():
            if k == "time":
                continue
            if isinstance(v, list) and i < len(v):
                obj[k] = v[i]
        result.append(obj)
    return result

# Dependencies
async def get_user_prefs(user_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return USER_PREFS.get(user_id or "default", {})

# Health
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# Spots CRUD
@app.get("/spots", response_model=List[Spot])
async def list_spots():
    return [Spot(**s) for s in SPOTS.values()]

@app.post("/spots", response_model=Spot)
async def create_spot(body: CreateSpot):
    sid = str(uuid.uuid4())
    spot = Spot(id=sid, **body.dict())
    SPOTS[sid] = spot.dict()
    return spot

@app.get("/spots/{spot_id}", response_model=Spot)
async def get_spot(spot_id: str):
    s = SPOTS.get(spot_id)
    if not s:
        raise HTTPException(404, detail="Spot not found")
    return Spot(**s)

@app.patch("/spots/{spot_id}", response_model=Spot)
async def update_spot(spot_id: str, body: UpdateSpot):
    s = SPOTS.get(spot_id)
    if not s:
        raise HTTPException(404, detail="Spot not found")
    data = {k: v for k, v in body.dict().items() if v is not None}
    s.update(data)
    SPOTS[spot_id] = s
    return Spot(**s)

@app.delete("/spots/{spot_id}")
async def delete_spot(spot_id: str):
    if spot_id in SPOTS:
        del SPOTS[spot_id]
        return {"deleted": True}
    raise HTTPException(404, detail="Spot not found")

# Sessions and Ratings
@app.post("/sessions")
async def create_session(body: SessionCreate):
    if body.spot_id not in SPOTS:
        raise HTTPException(404, detail="Spot not found")
    sid = str(uuid.uuid4())
    sess = body.dict()
    sess.update({"id": sid})
    SESSIONS[sid] = sess
    return sess

@app.post("/ratings")
async def create_rating(body: RatingCreate):
    if body.spot_id not in SPOTS:
        raise HTTPException(404, detail="Spot not found")
    if body.session_id and body.session_id not in SESSIONS:
        raise HTTPException(404, detail="Session not found")
    rec = body.dict()
    rec.update({"id": str(uuid.uuid4()), "created_at": datetime.utcnow().isoformat()})
    RATINGS.append(rec)
    # update user prefs simple learning: push wave/period/wind toward conditions of good ratings
    if body.rating >= 4:
        SP = SPOTS[body.spot_id]
        # No direct conditions here; learning is done on forecast personalization endpoint when data present
        USER_PREFS.setdefault(body.user_id, {})
        USER_PREFS[body.user_id]["last_high_rating_spot"] = body.spot_id
    return rec

@app.get("/ratings")
async def list_ratings(user_id: Optional[str] = None, spot_id: Optional[str] = None):
    out = RATINGS
    if user_id:
        out = [r for r in out if r["user_id"] == user_id]
    if spot_id:
        out = [r for r in out if r["spot_id"] == spot_id]
    return out

# Forecast endpoints
@app.get("/forecast/live")
async def forecast_live(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    hourly: Optional[str] = Query(None, description="Comma-separated hourly variables"),
):
    try:
        hrs = hourly.split(",") if hourly else None
        data = await fetch_open_meteo(latitude, longitude, hrs)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to fetch forecast: {e}")

@app.get("/forecast/spot/{spot_id}")
async def forecast_for_spot(spot_id: str, hourly: Optional[str] = None):
    s = SPOTS.get(spot_id)
    if not s:
        raise HTTPException(404, detail="Spot not found")
    hrs = hourly.split(",") if hourly else None
    try:
        return await fetch_open_meteo(s["latitude"], s["longitude"], hrs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to fetch forecast: {e}")

@app.get("/forecast/personalized/{spot_id}")
async def personalized_forecast(
    spot_id: str,
    user_id: Optional[str] = Query(None),
    hours: int = Query(48, ge=1, le=168),
):
    s = SPOTS.get(spot_id)
    if not s:
        raise HTTPException(404, detail="Spot not found")
    prefs = USER_PREFS.get(user_id or "default", {})
    data = await fetch_open_meteo(s["latitude"], s["longitude"], None)
    hours_data = build_hour_objects(data)[:hours]

    # If we have high ratings for this user at this spot, nudge prefs toward those hours' conditions
    user_ratings = [r for r in RATINGS if r["user_id"] == user_id and r["spot_id"] == spot_id and r["rating"] >= 4]
    if user_ratings:
        # naive: use most recent rating time window to set prefs bounds if available from session
        # Fallback: keep defaults
        pass

    scored = []
    for h in hours_data:
        scored.append({**h, "score": score_conditions(h, prefs)})

    return {"spot": s, "hours": scored}

# Preferences endpoints
class PrefsUpdate(BaseModel):
    preferred_wave_min: Optional[float] = None
    preferred_wave_max: Optional[float] = None
    preferred_period_min: Optional[float] = None
    preferred_period_max: Optional[float] = None
    max_wind: Optional[float] = None
    preferred_wind_dir: Optional[Any] = None  # deg or list

@app.get("/users/{user_id}/prefs")
async def get_prefs(user_id: str):
    return USER_PREFS.get(user_id, {})

@app.patch("/users/{user_id}/prefs")
async def patch_prefs(user_id: str, body: PrefsUpdate):
    prefs = USER_PREFS.setdefault(user_id, {})
    for k, v in body.dict(exclude_none=True).items():
        prefs[k] = v
    return prefs

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Root
@app.get("/")
async def root():
    return {"name": app.title, "version": app.version}
