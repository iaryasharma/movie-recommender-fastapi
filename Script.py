import pandas as pd
import json
import numpy as np
import ast

# Helper to safely parse JSON-like strings in CSV
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

# Load both CSV files
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

# Keep only the necessary columns
merged_df = merged_df[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew', 'tagline', 'release_date', 'vote_average']]

# Rename title column
merged_df.rename(columns={'title_x': 'title'}, inplace=True)

# Replace NaN with None for valid JSON
merged_df = merged_df.replace({np.nan: None})

# Parse JSON-like columns
for column in ['genres', 'keywords', 'cast', 'crew']:
    merged_df[column] = merged_df[column].apply(safe_literal_eval)

# Optional: Convert genre/keyword names for simpler frontend rendering
def extract_names(items, limit=None):
    names = [item['name'] for item in items if 'name' in item]
    return names[:limit] if limit else names

merged_df['genres'] = merged_df['genres'].apply(extract_names)
merged_df['keywords'] = merged_df['keywords'].apply(extract_names)
merged_df['cast'] = merged_df['cast'].apply(lambda x: extract_names(x, limit=3))  # top 3 cast
merged_df['crew'] = merged_df['crew'].apply(lambda crew: [member['name'] for member in crew if member.get('job') == 'Director'])

# Convert to JSON records
movies_json = merged_df.to_dict(orient='records')

# Save to a cleaned JSON file
with open('movies.json', 'w', encoding='utf-8') as f:
    json.dump(movies_json, f, ensure_ascii=False, indent=2)

print("✅ Cleaned JSON file saved as 'movies.json'")
