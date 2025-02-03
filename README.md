# Task3

**Create a simple recommendation system that suggests items to
users based on their preferences. You can use techniques like
collaborative filtering or content-based filtering to recommend
movies, books, or products to users.**


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_movie_recommendation_system():
    # Sample dataset
    data = {
        'title': ['Inception', 'The Matrix', 'Interstellar', 'The Dark Knight', 'Fight Club'],
        'description': [
            'A thief who enters the dreams of others to steal secrets.',
            'A hacker discovers reality is a simulation controlled by machines.',
            'A team of explorers travel through a wormhole in space.',
            'A vigilante fights crime in Gotham City.',
            'An insomniac and a soap salesman create an underground fight club.'
        ]
    }
    df = pd.DataFrame(data)

    # Convert text to numerical representation
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def recommend_movies(movie_title, num_recommendations=2):
        if movie_title not in df['title'].values:
            return "Movie not found in database."
        
        movie_index = df[df['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommended_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]
        return df.iloc[recommended_indices]['title'].tolist()
    
    return recommend_movies

# Create recommendation system
recommend = create_movie_recommendation_system()

# Example usage
print(recommend('Inception'))
