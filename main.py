from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from recommender import MovieRecommender

app = FastAPI()

# Allow cross-origin requests from all domains (use specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
