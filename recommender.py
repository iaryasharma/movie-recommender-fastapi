import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        # Load and merge datasets
        self.movies = pd.read_csv("tmdb_5000_movies.csv")
        self.credits = pd.read_csv("tmdb_5000_credits.csv")
        self.movies = self.movies.merge(self.credits, left_on='id', right_on='movie_id')

        # Select and rename relevant columns
        self.movies = self.movies[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        self.movies.rename(columns={'title_x': 'title'}, inplace=True)
        self.movies.dropna(inplace=True)

        # Extract features
        self.movies['genres'] = self.movies['genres'].apply(self._convert)
        self.movies['keywords'] = self.movies['keywords'].apply(self._convert)
        self.movies['cast'] = self.movies['cast'].apply(self._convert_cast)
        self.movies['crew'] = self.movies['crew'].apply(self._get_director)
        self.movies['overview'] = self.movies['overview'].apply(lambda x: x.split())

        # Combine all features into a single 'tags' column
        self.movies['tags'] = self.movies['overview'] + self.movies['genres'] + self.movies['keywords'] + self.movies['cast'] + self.movies['crew']
        self.movies['tags'] = self.movies['tags'].apply(lambda x: " ".join(x).lower())

        # Convert text to vectors using CountVectorizer (BoW model)
        self.cv = CountVectorizer(max_features=5000, stop_words='english')
        self.vectors = self.cv.fit_transform(self.movies['tags']).toarray()

        # Compute similarity matrix
        self.similarity = cosine_similarity(self.vectors)

    def _convert(self, text):
        return [i['name'] for i in ast.literal_eval(text)]

    def _convert_cast(self, text):
        return [i['name'] for i in ast.literal_eval(text)[:3]]  # Top 3 cast

    def _get_director(self, text):
        crew = ast.literal_eval(text)
        for person in crew:
            if person['job'] == 'Director':
                return [person['name']]
        return []

    def recommend(self, movie_title):
        movie_title = movie_title.lower()
        movie_index = self.movies[self.movies['title'].str.lower() == movie_title].index
        if len(movie_index) == 0:
            return ["Movie not found in dataset."]

        movie_index = movie_index[0]
        distances = self.similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        return [self.movies.iloc[i[0]].title for i in movie_list]
