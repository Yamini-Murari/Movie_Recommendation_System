# Movie_Recommendation_System

A hybrid movie recommender system built using **Collaborative Filtering** and **Content-Based Filtering**, powered by **Python**, **Pandas**, **Scikit-learn**, and a user-friendly interface with **Streamlit**.

## 📌 Features

- 🔍 **Search** for a movie you like
- 🤝 **Collaborative Filtering** based on user ratings (MovieLens dataset)
- 🧠 **Content-Based Filtering** using movie genres and TF-IDF
- 🔗 **Hybrid Recommendation** combining both methods
- 💡 Simple and intuitive **Streamlit UI**

## Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Streamlit**
- **MovieLens 100k Dataset**


## 📂 Dataset

This project uses the [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/).  
Place the following files in the project directory:
movies.dat
ratings.dat

## How to Run Streamlit App in VS Code

 1. Open VS Code

 2. Open the folder where your project files are located

 3. Ensure These Files Are Present

    app.py

    movies.dat

    ratings.dat

    requirements.txt

4. open Terminal

5. Install Required Packages

    pandas

    numpy

    scikit-learn
  
    streamlit

6. In the terminal inside VS Code, run:

     -  streamlit run app.py

   Streamlit will launch the app and give a URL like:
   
     - Local URL: http://localhost:8501

7. See the App in Browser

You’ll see your Movie Recommendation App UI where you can:

  - Search for a movie

  - Select from matches

  - Click “Get Recommendations”

  - View the top recommended movies 
