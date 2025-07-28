# Netflix-TV-Shows-Content-Explorer
Netflix &amp; TV Shows Content Explorer is a self project which is an interactive Streamlit web application that helps users explore and discover movies and TV shows using a comprehensive dataset. This tool enables recommendations, genre-based filtering, and language-specific top content—all in one place.

🔍 Features
🔎 Recommendation Engine
Enter the name of any movie or TV show and get the top 5 similar recommendations based on genre, rating, and metadata.
🎭 Top by Genre
Select any genre to view the top 20 highest-rated movies or TV shows globally in that category.
🌐 Top by Language
Choose a language and discover the top 20 most popular and highest-rated shows/movies available in that language.
ℹ️ Dataset Notes
Certain fields may show None due to missing or incomplete data in the original dataset. The dataset is taken from Kaggle.

📁 Files
File Name	              Description
streamlit netflix.py	  Main Streamlit app file
content_df.pkl	        Preprocessed content DataFrame (movies + series)
README.md	              Project overview and instructions
It also includes codes in the form of python (.py) file and jupyter notebook (.ipynb)


📦 Requirements
Python 3.7+
Streamlit
Pandas, Numpy, Matlplotlib
Joblib
(Optional) Scikit-learn (for generating cosine similarity matrix, if preprocessing is redone)

📌 Notes
The UI is optimized for full-width viewing. Best viewed in desktop browsers at 100% to 90% zoom.
Language codes like en, ko, hi are mapped to user-friendly names for display.
Missing data (e.g., None for director) is due to gaps in the source dataset.
Any missing or inaccurate information in the recommendations or listings may be due to limitations or gaps in the original dataset
The total project size is approximately 7.6 GB, which exceeds GitHub’s file size limits. Therefore, e.g., cosine_sim.pkl is NOT included in this repository.

📄 License
This project is for educational purposes. Attribution encouraged if re-used.

