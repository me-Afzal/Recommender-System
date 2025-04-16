from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Load and preprocess the dataset
movie_df = pd.read_csv('movies_raw.csv')
movie_df['genres'] = movie_df['genres'].apply(lambda x: [d['name'] for d in json.loads(x)] if isinstance(x, str) else [])
movie_df['keywords'] = movie_df['keywords'].apply(lambda x: [d['name'] for d in json.loads(x)] if isinstance(x, str) else [])
movie_df = movie_df[['original_title', 'genres', 'keywords', 'overview']]
movie_df['genres'] = movie_df['genres'].apply(lambda x: " ".join(x))
movie_df['keywords'] = movie_df['keywords'].apply(lambda x: " ".join(x))
movie_df['tags'] = movie_df['genres'] + " " + movie_df['keywords'] + " " + movie_df['overview']
movie_df.dropna(subset=['tags'], inplace=True)
movie_df = movie_df[['original_title', 'tags']].rename(columns={'original_title': 'Title'})
movie_df.reset_index(drop=True, inplace=True)

titles = movie_df['Title'].apply(lambda x: x.lower())
indices = pd.Series(movie_df.index, index=titles)

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(movie_df['tags']).toarray()
sim_matrix = cosine_similarity(vectors)

# Recommender function
def recommend(title):
    title = title.lower()
    if title not in indices:
        return None
    idx = indices[title]
    sim_score = list(enumerate(sim_matrix[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:11]
    sim_indices = [x[0] for x in sim_score]
    return list(movie_df['Title'].iloc[sim_indices])

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error = None
    if request.method == 'POST':
        movie_title = request.form['movie']
        recommendations = recommend(movie_title)
        if recommendations is None:
            error = "Movie not found. Please try another title."
    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    app.run(debug=True)
