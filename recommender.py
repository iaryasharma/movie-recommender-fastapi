import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class MovieRecommender:
    def __init__(self):
        try:
            # Load and read datasets
            logging.info("Loading CSV files...")
            self.movies = pd.read_csv("tmdb_5000_movies.csv")
            self.credits = pd.read_csv("tmdb_5000_credits.csv")
            logging.info("CSV files loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error reading CSV files: {e}")
            raise

        try:
            # Merge movies and credits dataset using 'id' and 'movie_id'
            self.movies = self.movies.merge(self.credits, left_on='id', right_on='movie_id')

            # Select and rename relevant columns
            self.movies = self.movies[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            self.movies.rename(columns={'title_x': 'title'}, inplace=True)

            # Drop any rows with null values
            self.movies.dropna(inplace=True)

            # Convert stringified JSON columns to list of names
            self.movies['genres'] = self.movies['genres'].apply(self._convert)
            self.movies['keywords'] = self.movies['keywords'].apply(self._convert)
            self.movies['cast'] = self.movies['cast'].apply(self._convert_cast)
            self.movies['crew'] = self.movies['crew'].apply(self._get_director)

            # Split overview text into list of words
            self.movies['overview'] = self.movies['overview'].apply(lambda x: x.split())

            # Combine all text features into a single 'tags' column
            self.movies['tags'] = self.movies['overview'] + self.movies['genres'] + self.movies['keywords'] + self.movies['cast'] + self.movies['crew']
            self.movies['tags'] = self.movies['tags'].apply(lambda x: " ".join(x).lower())

            # Create Bag of Words vectors from 'tags' column
            self.cv = CountVectorizer(max_features=5000, stop_words='english')
            self.vectors = self.cv.fit_transform(self.movies['tags']).toarray()

            # Calculate cosine similarity matrix for recommendations
            self.similarity = cosine_similarity(self.vectors)

            logging.info("Recommender system initialized.")
        except Exception as e:
            logging.error(f"Error initializing recommender system: {e}")
            raise

    # Convert genres/keywords from stringified list of dicts to list of names
    def _convert(self, text):
        try:
            return [i['name'] for i in ast.literal_eval(text)]
        except:
            return []

    # Extract top 3 cast members
    def _convert_cast(self, text):
        try:
            return [i['name'] for i in ast.literal_eval(text)[:3]]
        except:
            return []

    # Extract director's name from crew list
    def _get_director(self, text):
        try:
            crew = ast.literal_eval(text)
            for person in crew:
                if person['job'] == 'Director':
                    return [person['name']]
        except:
            return []
        return []

    # Recommend top 5 similar movies based on the title
    def recommend(self, movie_title):
        movie_title = movie_title.lower()
        movie_index = self.movies[self.movies['title'].str.lower() == movie_title].index
        if len(movie_index) == 0:
            return ["Movie not found in dataset."]

        movie_index = movie_index[0]
        distances = self.similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        return [self.movies.iloc[i[0]].title for i in movie_list]
