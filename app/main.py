import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.trip import router as trip_router

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