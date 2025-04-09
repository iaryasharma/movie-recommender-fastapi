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

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cinewhiz.vercel.app"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender system
try:
    logger.info("Initializing Movie Recommender...")
    recommender = MovieRecommender()
    logger.info("Recommender system loaded successfully.")
except Exception as e:
    logger.error(f"Failed to initialize recommender system: {e}")
    recommender = None

@app.get("/")
def home():
    logger.info("Root endpoint hit.")
    return {"message": "🎬 CineWhiz API is live and running!"}

@app.get("/recommend")
def recommend(title: str):
    logger.info(f"Received recommendation request for: {title}")
    if not recommender:
        logger.error("Recommender system is not initialized.")
        raise HTTPException(status_code=500, detail="Recommender system is unavailable.")

    result = recommender.recommend(title)
    if result == ["Movie not found in dataset."]:
        logger.warning(f"Movie '{title}' not found in dataset.")
        raise HTTPException(status_code=404, detail="Movie not found")
    logger.info(f"Recommendations for '{title}': {result}")
    return {"recommended": result}

# For local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting app on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
