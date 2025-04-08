# 🎮 Movie Recommendation System (FastAPI + NLP)

This project is a **Movie Recommendation System** built using **FastAPI** for the backend and **Natural Language Processing (NLP)** techniques to generate content-based recommendations.

Frontend (Next.js) is integrated via API routes to display real-time recommendations to users.

---

## 📚 Overview
This system allows users to input a movie name and receive similar movie recommendations based on content like genre, keywords, cast, and crew.

It uses the **Bag-of-Words (BoW)** model and **CountVectorizer** to convert movie metadata into a format that allows for similarity comparisons using **Cosine Similarity**.

---

## 🧑‍💻 Tech Stack
- **FastAPI** (Python backend API)
- **Pandas** (Data manipulation)
- **scikit-learn** (CountVectorizer, Cosine Similarity)
- **TMDB 5000 Movie Dataset**
- **Next.js** (Frontend)

---

## 📊 How It Works

### Step 1: Data Loading and Preprocessing
- Load `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
- Merge on `id` to bring in both movie metadata and cast/crew information
- Extract and clean important columns: `overview`, `genres`, `keywords`, `cast`, `crew`

### Step 2: Feature Engineering
- Parse JSON-like fields using `ast.literal_eval`
- Extract top 3 cast members and director
- Create a `tags` field by combining overview, genres, keywords, cast, and director

### Step 3: Vectorization using CountVectorizer

#### What is Bag-of-Words (BoW)?
The **Bag-of-Words model** is a way of representing text data:
- It converts a collection of words (text) into fixed-length numeric feature vectors.
- Each unique word becomes a feature (column), and the value is the count of that word in a document.

We use `CountVectorizer` from scikit-learn:
```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movie_tags).toarray()
```

This creates a matrix where:
- Rows represent movies
- Columns represent words (features)
- Values represent frequency of the word in that movie's tags

### Step 4: Similarity Matrix
Using **cosine similarity**, we measure how close one movie's vector is to another's. This gives us the top N similar movies.

---

## 📂 Project Structure
```
.
├── Main.py                # FastAPI routes and app
├── recommender.py         # MovieRecommender class and NLP logic
├── requirements.txt       # Python dependencies
├── script.py              # Script to convert Csv's into JSON
├── tmdb_5000_movies.csv   # Movie dataset
├── tmdb_5000_credits.csv  # Cast & Crew dataset
└── README.md
```

---

## 🚀 API Endpoints
### `GET /`
Returns a basic health check message.

### `GET /recommend?title=Inception`
Returns top 5 recommended movies based on the title provided.

Response:
```json
{
  "recommended": [
    "The Helix... Loaded",
    "The Count of Monte Cristo",
    "Flatliners",
    "Cypher",
    "Transformers: Revenge of the Fallen"
  ]
}
```

---

## 🚧 Frontend Integration
Frontend is developed using **Next.js**:
- Connects to FastAPI endpoint `/recommend?title=...`
- Displays a search input and recommended results on screen

You can customize the UI easily with components like:
```js
fetch("http://localhost:8000/recommend?title=Inception")
  .then(res => res.json())
  .then(data => setRecommendations(data.recommended))
```

---

## 🏃‍♀️ Getting Started

### ✅ Prerequisites
Make sure you have the following installed:
- Python 3.8+
- `pip` (Python package manager)
- `virtualenv` (recommended)
- `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project directory

### ⚖️ Setup Instructions
#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/movie-recommender-fastapi.git
cd movie-recommender-fastapi
```

#### 2. Create and Activate Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install fastapi uvicorn pandas scikit-learn
```

#### 4. Run the FastAPI Server
```bash
uvicorn Main:app --reload
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🎓 Key Learning Outcomes
- Used NLP concepts (Bag-of-Words, Vectorization, Cosine Similarity)
- Implemented a scalable REST API using FastAPI
- Integrated backend with a modern frontend stack

---

## ❌ Limitations
- Only works with exact movie titles present in dataset
- Doesn't handle spelling mistakes or partial matches (future work: add fuzzy matching)

---

## ✨ Future Improvements
- Use TF-IDF or Word Embeddings for better accuracy

---

## 📖 References
- TMDB 5000 Movie Dataset: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- FastAPI Docs: https://fastapi.tiangolo.com/
- scikit-learn Docs: https://scikit-learn.org/stable/modules/feature_extraction.html

---

Made with ❤️ by [Arya Sharma](https://fragnite.vercel.app)