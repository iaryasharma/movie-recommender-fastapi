import pandas as pd
import ast
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        try:
            # Define absolute paths for dataset files (robust for deployment environments)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            movies_path = os.path.join(base_dir, "tmdb_5000_movies.csv")
            credits_path = os.path.join(base_dir, "tmdb_5000_credits.csv")

            # Load both CSV files
            logging.info("📁 Loading CSV files...")
            self.movies = pd.read_csv(movies_path)
            self.credits = pd.read_csv(credits_path)
            logging.info("✅ CSV files loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"❌ File not found: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Error reading CSV files: {e}")
            raise

        try:
            # Merge the two datasets on movie ID
            self.movies = self.movies.merge(self.credits, left_on='id', right_on='movie_id')

            # Select and rename relevant columns for processing
            self.movies = self.movies[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            self.movies.rename(columns={'title_x': 'title'}, inplace=True)

            # Drop rows with any missing values to avoid processing errors
            self.movies.dropna(inplace=True)

            # Convert JSON-like strings to lists of names (e.g., genres, keywords)
            self.movies['genres'] = self.movies['genres'].apply(self._convert)
            self.movies['keywords'] = self.movies['keywords'].apply(self._convert)
            self.movies['cast'] = self.movies['cast'].apply(self._convert_cast)
            self.movies['crew'] = self.movies['crew'].apply(self._get_director)

            # Tokenize the overview text
            self.movies['overview'] = self.movies['overview'].apply(lambda x: x.split())

            # Combine all relevant text fields into a single list of tags
            self.movies['tags'] = (
                self.movies['overview'] +
                self.movies['genres'] +
                self.movies['keywords'] +
                self.movies['cast'] +
                self.movies['crew']
            )

            # Join the tags list into a lowercase string for vectorization
            self.movies['tags'] = self.movies['tags'].apply(lambda x: " ".join(x).lower())

            # Save lowercase versions of titles for quick matching in recommend()
            self.movies['title_lower'] = self.movies['title'].str.lower()

            # Convert text data into vectors using Bag-of-Words
            self.cv = CountVectorizer(max_features=5000, stop_words='english')
            self.vectors = self.cv.fit_transform(self.movies['tags']).toarray()

            # Compute cosine similarity matrix between all movies
            self.similarity = cosine_similarity(self.vectors)

            logging.info("🚀 Recommender system initialized successfully.")
        except Exception as e:
            logging.error(f"❌ Error initializing recommender system: {e}")
            raise

    # Convert JSON stringified list (genres/keywords) to list of 'name' values
    def _convert(self, text):
        try:
            return [i['name'] for i in ast.literal_eval(text)]
        except (ValueError, SyntaxError, TypeError):
            return []

    # Extract top 3 cast members from the cast field
    def _convert_cast(self, text):
        try:
            return [i['name'] for i in ast.literal_eval(text)[:3]]
        except (ValueError, SyntaxError, TypeError):
            return []

    # Extract director's name from the crew list
    def _get_director(self, text):
        try:
            crew = ast.literal_eval(text)
            return [person['name'] for person in crew if person.get('job') == 'Director']
        except (ValueError, SyntaxError, TypeError):
            return []

    # Main recommendation method
    def recommend(self, movie_title):
        # Convert input to lowercase and search for exact title match
        movie_title = movie_title.lower()
        movie_index = self.movies[self.movies['title_lower'] == movie_title].index

        # If movie is not in dataset, return a message
        if len(movie_index) == 0:
            return ["Movie not found in dataset."]

        # Get index and compute similarity with all other movies
        idx = movie_index[0]
        distances = self.similarity[idx]

        # Sort movies based on similarity scores, skipping the first (itself)
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        # Return the top 5 most similar movie titles
        return [self.movies.iloc[i[0]].title for i in movie_list]
