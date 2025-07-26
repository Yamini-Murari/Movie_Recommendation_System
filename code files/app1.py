import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
#download datasets in the same folder 
def load_movie_data():
    try:
        movies_df = pd.read_csv(
            'movies.dat',
            sep='::',
            engine='python',
            names=['movieId','title','genres'],
            encoding='ISO-8859-1'
        )
        ratings_df = pd.read_csv(
            'ratings.dat',
            sep='::',
            engine='python',
            names=['userId','movieId','rating','timestamp'],
            encoding='ISO-8859-1'
        )
        return movies_df, ratings_df
    except FileNotFoundError:
        st.error("could not find dataset")
        return None,None
#similarity matrix based on user ratings
def user_similarity_matrix(ratings_df, movies_df):
    merged_df = ratings_df.merge(movies_df,on='movieId')

    #creating a pivot table with users and their ratings
    ratings_pivot = merged_df.pivot(index='userId',columns='title',values='rating').fillna(0)

    #get cosine similarity between movies based on user ratings
    similarity = cosine_similarity(ratings_pivot.T)
    sim_df = pd.DataFrame(similarity,index=ratings_pivot.columns,columns=ratings_pivot.columns)
    return sim_df

#type a movie,return top N similar ones
def get_similar_movies(movie_title,sim_df,top_n=6):
    if movie_title not in sim_df.columns:
        return []
    sorted_scores=sim_df[movie_title].sort_values(ascending=False)

    #skipping the first one because it's the same movie
    similar_movies = sorted_scores.iloc[1:top_n+1].index.tolist()
    return similar_movies

#get a genre-based similarity matrix using TF-IDF
def compute_genre_similarity_matrix(movies_df):
    genre_df = movies_df.copy()
    genre_df['genres'] = genre_df['genres'].str.replace('|',' ',regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix=tfidf.fit_transform(genre_df['genres'])

    #cosine similarity between movies based on genres
    genre_sim = cosine_similarity(genre_matrix)
    return pd.DataFrame(genre_sim,index=genre_df['title'],columns=genre_df['title'])

#combine content and collaborative filtering for better recs
def hybrid_recommendation(movie_title,user_sim_df,genre_sim_df,top_n=5):
    collab_candidates=get_similar_movies(movie_title,user_sim_df,top_n * 2)
    genre_candidates=get_similar_movies(movie_title,genre_sim_df,top_n * 2)

    combined = list(pd.unique(collab_candidates + genre_candidates))

    #return top N
    return combined[:top_n]

#search to get possible movie matches from input
def search_movies(query,movie_df,limit=20):
    if not query:
        return []
    
    results_df = movie_df[movie_df['title'].str.lower().str.contains(query.lower())]
    matched_titles = results_df['title'].head(limit).tolist()
    return matched_titles

#streamlit app starts here
def main():
    st.title("🎬🎥 Movie Recommender 📽")
    movies,ratings=load_movie_data()
    if movies is None or ratings is None:
        return

    st.write("Just type a movie you liked, and I'll suggest a few others you might enjoy.")

    user_input = st.text_input("Movie name:")
    if user_input:
        matches = search_movies(user_input,movies)

        if not matches:
            st.warning("Hmm... I couldn't find anything that matches. Maybe check your spelling?")
            return

        selected_title = st.selectbox("Pick the correct movie:", matches)

        if st.button("Get Recommendations"):
            user_sim_matrix =user_similarity_matrix(ratings,movies)
            genre_sim_matrix=compute_genre_similarity_matrix(movies)

            recs = hybrid_recommendation(selected_title,user_sim_matrix,genre_sim_matrix)

            if recs:
                st.subheader("You might enjoy these:")
                for movie in recs:
                    st.write("- " + movie)
            else:
                st.info("Couldn't find much. Maybe try a different movie?")

if __name__ == "__main__":
    main()
