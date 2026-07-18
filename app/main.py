import os
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.trip import router as trip_router
from app.core.supabase_keepalive import check_and_ping

app = FastAPI(
    title="PackPals AI Backend Orchestrator", 
    description="Streamlined architectural layout exposing LangGraph pipelines for mobile suitcase packing assistance.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    check_and_ping()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    check_and_ping()
    started_at = time.time()

    print(
        f"[REQUEST START] {request.method} {request.url.path}",
        flush=True,
    )

    try:
        response = await call_next(request)

        duration_ms = round((time.time() - started_at) * 1000)

        print(
            f"[REQUEST END] {request.method} {request.url.path} "
            f"status={response.status_code} duration_ms={duration_ms}",
            flush=True,
        )

        return response

    except Exception as exc:
        duration_ms = round((time.time() - started_at) * 1000)

        print(
            f"[REQUEST ERROR] {request.method} {request.url.path} "
            f"duration_ms={duration_ms} error={exc}",
            flush=True,
        )

        raise

app.include_router(chat_router)
app.include_router(trip_router)

@app.get("/")
async def root():
    return {
        "ok": True,
        "message": "PackPals backend is running",
    }

@app.get("/api/wakeup")
async def wakeup():
    return {
        "ok": True,
        "status": "awake",
        "service": "packpals-backend",
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)