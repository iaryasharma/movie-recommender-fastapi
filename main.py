from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender import MovieRecommender
import uvicorn
import os

app = FastAPI(
    title="CineWhiz Movie Recommendation API",
    description="Backend for CineWhiz movie recommendation platform using FastAPI and scikit-learn",
    version="1.0.0"
)

# CORS settings to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cinewhiz.vercel.app"],  # Update with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your recommender system
recommender = MovieRecommender()

# Root endpoint to confirm API is running
@app.get("/")
def home():
    return {"message": "🎬 CineWhiz API is live and running!"}

# Recommendation endpoint
@app.get("/recommend")
def recommend(title: str):
    result = recommender.recommend(title)
    if result == ["Movie not found in dataset."]:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"recommended": result}

# Run locally with: python main.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
