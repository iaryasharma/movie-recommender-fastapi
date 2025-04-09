from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender import MovieRecommender

app = FastAPI(
    title="CineWhiz Movie Recommendation API",
    description="Backend for CineWhiz movie recommendation platform using FastAPI and scikit-learn",
    version="1.0.0"
)

# Allow CORS only from your Vercel frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cinewhiz.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender
recommender = MovieRecommender()

@app.get("/")
def home():
    return {"message": "Movie Recommendation API is working!"}

@app.get("/recommend")
def recommend(title: str):
    result = recommender.recommend(title)
    if result == ["Movie not found in dataset."]:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"recommended": result}
