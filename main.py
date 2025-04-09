from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender import MovieRecommender
import uvicorn
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CineWhiz Movie Recommendation API",
    description="Backend for CineWhiz movie recommendation platform using FastAPI and scikit-learn",
    version="1.0.0"
)

# Allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cinewhiz.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = None

@app.on_event("startup")
def load_recommender():
    global recommender
    try:
        logger.info("🔄 Initializing Movie Recommender...")
        recommender = MovieRecommender()
        logger.info("✅ Recommender system loaded successfully.")
    except Exception as e:
        logger.exception("❌ Failed to initialize recommender system.")
        recommender = None

@app.get("/")
def root():
    logger.info("📡 Root endpoint accessed.")
    return {"message": "🎬 CineWhiz API is live and running!"}

@app.get("/recommend")
def recommend(title: str):
    logger.info(f"🎯 Request for recommendations based on: {title}")
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender system unavailable.")
    try:
        result = recommender.recommend(title)
        if result == ["Movie not found in dataset."]:
            raise HTTPException(status_code=404, detail=f"Movie '{title}' not found.")
        return {"recommended": result}
    except Exception as e:
        logger.exception(f"🚨 Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Render injects PORT into env var — so we bind to it
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Launching CineWhiz API on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
    logger.info("✅ CineWhiz API is live!")