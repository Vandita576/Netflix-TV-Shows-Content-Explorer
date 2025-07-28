#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[4]:


movies_df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Projects\Data Science\1. Netflix Content Analysis\dataset\Movies till 2025.csv")
series_df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Projects\Data Science\1. Netflix Content Analysis\dataset\TV Shows till 2025.csv")


# In[5]:


content_df = pd.concat([movies_df, series_df], ignore_index=True)


# In[6]:


movies_only = content_df[content_df['type'] == 'Movie']
series_only = content_df[content_df['type'] == 'TV Show']
# Extracting only movies and series from combined dataset


# In[7]:


print("Movies shape:", movies_df.shape)
print("TV Shows shape:", series_df.shape)


# In[8]:


print(movies_df.head(3))
print(series_df.head(3))


# In[9]:


# Summary statistics for numeric columns
print(movies_df.describe())
print(series_df.describe())


# In[10]:


# Data types and non-null counts
print(movies_df.info())
print(series_df.info())


# In[11]:


print("Duplicate Movies:", movies_df.duplicated().sum())
print("Duplicate Series:", series_df.duplicated().sum())


# In[12]:


print("Movies missing values:\n", movies_df.isnull().sum())
print("Series missing values:\n", series_df.isnull().sum())


# In[13]:


# Convert dates to datetime
movies_df['date_added'] = pd.to_datetime(movies_df['date_added'], errors='coerce')
series_df['date_added'] = pd.to_datetime(series_df['date_added'], errors='coerce')


# In[14]:


# Clean genres (lowercase, strip whitespace)
movies_df['genres'] = movies_df['genres'].str.lower().str.strip()
series_df['genres'] = series_df['genres'].str.lower().str.strip()


# In[15]:


# Duration extracted in numeric form

def convert_duration(value):
    if pd.isna(value): return None
    value = str(value).lower()
    if 'h' in value or 'min' in value:
        parts = value.replace('h', 'h ').split()
        total = 0
        for part in parts:
            if 'h' in part:
                total += int(part.replace('h', '')) * 60
            elif 'min' in part:
                total += int(part.replace('min', ''))
        return total
    elif 'season' in value:
        return int(value.split()[0])
    return None

movies_df['duration_mins'] = movies_df['duration'].apply(convert_duration)
series_df['duration_seasons'] = series_df['duration'].apply(convert_duration)


# In[16]:


# View Missing Data Count and Percentage
def missing_report(df, name):
    print(f"\nMissing values in {name}:\n")
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': percent})
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

missing_report(movies_df, 'movies_df')
missing_report(series_df, 'series_df')


# In[17]:


# Cleaning to movies_df and series_df

# Fill common columns
for df in [movies_df, series_df]:
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['duration'] = df['duration'].fillna('Not Available')
    df['genres'] = df['genres'].fillna('Uncategorized')
    df['description'] = df['description'].fillna('Not Available')

# Handle date_added separately (can be left as NaT)
movies_df['date_added'] = pd.to_datetime(movies_df['date_added'], errors='coerce')
series_df['date_added'] = pd.to_datetime(series_df['date_added'], errors='coerce')

# Fill numeric NaNs for movies_df
movies_df['budget'] = movies_df['budget'].fillna(0)
movies_df['revenue'] = movies_df['revenue'].fillna(0)

# vote_average / popularity — if small missing %: drop
movies_df.dropna(subset=['vote_average'], inplace=True)
series_df.dropna(subset=['vote_average'], inplace=True)


# In[18]:


# Re-check missing data
missing_report(movies_df, 'movies_df (after cleaning)')
missing_report(series_df, 'series_df (after cleaning)')


#  -----------EXPLORATORY DATA ANALYSIS--------------

# In[19]:


print("Movies:", movies_df.shape)
print("Series:", series_df.shape)


# In[20]:


print("Movies by type:\n", movies_df['type'].value_counts())
print("Series by type:\n", series_df['type'].value_counts())


# In[21]:


print("Unique movie genres:", movies_df['genres'].nunique())
print("Unique series genres:", series_df['genres'].nunique())
print("Movie ratings:", movies_df['rating'].unique())
print("Series ratings:", series_df['rating'].unique())


# In[22]:


# Year-wise content release trend
movies_by_year = movies_df['release_year'].value_counts().sort_index()
series_by_year = series_df['release_year'].value_counts().sort_index()


# In[23]:


# Year Netflix added the content (date_added)
movies_df['added_year'] = movies_df['date_added'].dt.year
series_df['added_year'] = series_df['date_added'].dt.year

added_movies = movies_df['added_year'].value_counts().sort_index()
added_series = series_df['added_year'].value_counts().sort_index()


# In[24]:


# Most Common Genres
def top_genres(df, name):
    genres = df['genres'].dropna().str.lower().str.split(', ')
    flat_list = [genre for sublist in genres for genre in sublist if genre]
    top = Counter(flat_list).most_common(10)
    print(f"\nTop genres in {name}:\n")
    for g, c in top:
        print(f"{g}: {c}")

top_genres(movies_df, 'Movies')
top_genres(series_df, 'TV Shows')


# In[25]:


# Mean rating and popularity
print("Movie avg vote:", movies_df['vote_average'].mean())
print("Series avg vote:", series_df['vote_average'].mean())

print("Top 5 most popular movies:\n", movies_df[['title', 'popularity']].sort_values(by='popularity', ascending=False).head())
print("Top 5 most popular series:\n", series_df[['title', 'popularity']].sort_values(by='popularity', ascending=False).head())


# In[26]:


# Top countries
print("Top 5 movie countries:\n", movies_df['country'].value_counts().head())
print("Top 5 series countries:\n", series_df['country'].value_counts().head())

# Top languages
print("Top 5 movie languages:\n", movies_df['language'].value_counts().head())
print("Top 5 series languages:\n", series_df['language'].value_counts().head())


# In[27]:


movies_df['title_length'] = movies_df['title'].str.len()
series_df['title_length'] = series_df['title'].str.len()

print("Longest movie title:\n", movies_df.loc[movies_df['title_length'].idxmax()])
print("Longest series title:\n", series_df.loc[series_df['title_length'].idxmax()])


# -----------Visualization — Netflix Content Analysis----------

# In[28]:


# Style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# In[29]:


# Year-wise Content Release Trend
# Movies
movies_by_year = movies_df['release_year'].value_counts().sort_index()
sns.lineplot(x=movies_by_year.index, y=movies_by_year.values)
plt.title("Movies Released per Year")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()

# TV Shows
series_by_year = series_df['release_year'].value_counts().sort_index()
sns.lineplot(x=series_by_year.index, y=series_by_year.values, color="orange")
plt.title("TV Shows Released per Year")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()


# In[30]:


# Year Netflix Added Content
# Movies added to Netflix
added_movies = movies_df['added_year'].value_counts().sort_index()
sns.barplot(x=added_movies.index, y=added_movies.values, palette='Blues_d')
plt.title("Movies Added to Netflix by Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.show()

# TV Shows added to Netflix
added_series = series_df['added_year'].value_counts().sort_index()
sns.barplot(x=added_series.index, y=added_series.values, palette='Oranges_d')
plt.title("TV Shows Added to Netflix by Year")
plt.xlabel("Year")
plt.ylabel("Number of TV Shows")
plt.xticks(rotation=45)
plt.show()


# In[31]:


# Top 10 Genres
def plot_top_genres(df, label, color):
    genres_series = df['genres'].dropna().str.lower().str.split(', ')
    all_genres = [genre for sublist in genres_series for genre in sublist]
    genre_counts = Counter(all_genres)
    top_genres = dict(genre_counts.most_common(10))
    
    sns.barplot(x=list(top_genres.values()), y=list(top_genres.keys()), palette=color)
    plt.title(f"Top 10 Genres - {label}")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.show()

plot_top_genres(movies_df, "Movies", "Blues_r")
plot_top_genres(series_df, "TV Shows", "Oranges_r")


# In[32]:


# Top 10 countries
top_countries_movies = movies_df['country'].value_counts().head(10)
sns.barplot(x=top_countries_movies.values, y=top_countries_movies.index, palette='Blues')
plt.title("Top Countries Producing Movies")
plt.xlabel("Count")
plt.ylabel("Country")
plt.show()

top_countries_series = series_df['country'].value_counts().head(10)
sns.barplot(x=top_countries_series.values, y=top_countries_series.index, palette='Oranges')
plt.title("Top Countries Producing TV Shows")
plt.xlabel("Count")
plt.ylabel("Country")
plt.show()


# In[33]:


# Ratings
sns.histplot(movies_df['vote_average'], bins=20, kde=True, color='skyblue', label='Movies')
sns.histplot(series_df['vote_average'], bins=20, kde=True, color='orange', label='Series', alpha=0.6)
plt.title("Vote Average Distribution")
plt.xlabel("Vote Average")
plt.ylabel("Count")
plt.legend()
plt.show()


# In[34]:


# Add new log-transformed column
movies_df['log_popularity'] = np.log1p(movies_df['popularity'])
series_df['log_popularity'] = np.log1p(series_df['popularity'])


# In[35]:


sns.histplot(movies_df['log_popularity'], bins=30, color='skyblue', label='Movies', kde=True)
sns.histplot(series_df['log_popularity'], bins=30, color='orange', label='Series', kde=True, alpha=0.6)
plt.title("Log-Transformed Popularity Distribution")
plt.xlabel("log(Popularity + 1)")
plt.ylabel("Count")
plt.legend()
plt.show()


# In[36]:


#Movie Rating Distribution
rating_counts = movies_df['rating'].value_counts().head(6)

colors = sns.color_palette("coolwarm", len(rating_counts))
explode = [0.05] * len(rating_counts)

plt.figure(figsize=(4,4))
plt.pie(rating_counts, labels=rating_counts.index, colors=colors, autopct='%1.1f%%',
        shadow=True, explode=explode, startangle=140)
plt.title("Top Movie Content Ratings")
plt.axis('equal')
plt.show()


# In[37]:


# Top Genres in Series
genres_series = series_df['genres'].dropna().str.lower().str.split(', ')
flat_genres = [g for sublist in genres_series for g in sublist]
genre_count = dict(Counter(flat_genres).most_common(6))

colors = sns.color_palette("YlOrRd", len(genre_count))
explode = [0.05] * len(genre_count)

plt.figure(figsize=(4, 4))
plt.pie(genre_count.values(), labels=genre_count.keys(), colors=colors, autopct='%1.1f%%',
        shadow=True, explode=explode, startangle=90)
plt.title("Top Genres in TV Shows")
plt.axis('equal')
plt.show()


# In[38]:


# Vote Average Comparison Movies vs Series
sns.kdeplot(movies_df['vote_average'], label='Movies', shade=True, color='skyblue')
sns.kdeplot(series_df['vote_average'], label='TV Shows', shade=True, color='orange')
plt.title("Vote Average: Movies vs TV Shows")
plt.xlabel("Vote Average")
plt.legend()
plt.show()


# In[39]:


# Popularity Comparison Movies vs Series
sns.kdeplot(movies_df['log_popularity'], label='Movies', shade=True, color='skyblue')
sns.kdeplot(series_df['log_popularity'], label='TV Shows', shade=True, color='orange')
plt.title("Log-Transformed Popularity: Movies vs TV Shows")
plt.xlabel("log(Popularity + 1)")
plt.legend()
plt.show()


# ----------- COMPARATIVE ANALYSIS - KEY ASPECTS -------------

# In[40]:


# average vote and popularity
avg_metrics = pd.DataFrame({
    'Type': ['Movies', 'TV Shows'],
    'Avg Vote': [movies_df['vote_average'].mean(), series_df['vote_average'].mean()],
    'Avg Popularity': [movies_df['popularity'].mean(), series_df['popularity'].mean()]
})
avg_metrics.set_index('Type').plot(kind='bar', figsize=(8,5), color=['tomato', 'steelblue'])
plt.title("Average Vote & Popularity: Movies vs TV Shows")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


# In[41]:


#Content Volume Over Years
movies_year = movies_df['release_year'].value_counts().sort_index()
series_year = series_df['release_year'].value_counts().sort_index()

plt.figure(figsize=(10,5))
plt.plot(movies_year.index, movies_year.values, label='Movies', color='crimson')
plt.plot(series_year.index, series_year.values, label='TV Shows', color='darkgreen')
plt.title("Content Released Per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[42]:


#Rating Breakdown (G, PG, etc.) – Movies vs TV Shows
# Rating counts
movie_ratings = movies_df['rating'].value_counts().sort_values(ascending=False)
series_ratings = series_df['rating'].value_counts().sort_values(ascending=False)

# Create stacked subplots (1 column, 2 rows)
fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

# Movie ratings
sns.barplot(x=movie_ratings.values, y=movie_ratings.index, ax=ax[0], palette='Reds_r')
ax[0].set_title("🎬 Movie Ratings Distribution", fontsize=14)
ax[0].set_xlabel("Count", fontsize=12)
ax[0].set_ylabel("Rating", fontsize=12)

# TV show ratings
sns.barplot(x=series_ratings.values, y=series_ratings.index, ax=ax[1], palette='Blues_r')
ax[1].set_title("📺 TV Show Ratings Distribution", fontsize=14)
ax[1].set_xlabel("Count", fontsize=12)
ax[1].set_ylabel("Rating", fontsize=12)

plt.tight_layout()
plt.show()


# In[43]:


#verage Cast Size and Duration Comparison

# Handle 'cast' as list of actors
movies_df['cast_size'] = movies_df['cast'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
series_df['cast_size'] = series_df['cast'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

# Extract duration in minutes or seasons
import re
def extract_duration(text):
    if pd.isna(text): return None
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else None

movies_df['duration_mins'] = movies_df['duration'].apply(extract_duration)
series_df['seasons'] = series_df['duration'].apply(extract_duration)

# Average values
avg_cast = [movies_df['cast_size'].mean(), series_df['cast_size'].mean()]
avg_duration = [movies_df['duration_mins'].mean(), series_df['seasons'].mean()]

# Plot
df_compare = pd.DataFrame({
    'Content Type': ['Movies', 'TV Shows'],
    'Avg Cast Size': avg_cast,
    'Avg Duration (mins/seasons)': avg_duration
})

df_compare.set_index('Content Type').plot(kind='bar', figsize=(8,5), color=['mediumvioletred', 'dodgerblue'])
plt.title("Average Cast Size & Duration")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[44]:


# Extracting year from date_added
# Convert to datetime
movies_df['date_added'] = pd.to_datetime(movies_df['date_added'], errors='coerce')
series_df['date_added'] = pd.to_datetime(series_df['date_added'], errors='coerce')

# Extract year and month
movies_df['year_added'] = movies_df['date_added'].dt.year
series_df['year_added'] = series_df['date_added'].dt.year

movies_df['month_added'] = movies_df['date_added'].dt.month
series_df['month_added'] = series_df['date_added'].dt.month


# In[45]:


# Counting number of actors listed in cast
movies_df['cast_size'] = movies_df['cast'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
series_df['cast_size'] = series_df['cast'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)


# In[46]:


# Counting how many countries are listed per content
movies_df['country_count'] = movies_df['country'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
series_df['country_count'] = series_df['country'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)


# In[47]:


import re

# For movies (duration in minutes)
movies_df['duration_min'] = movies_df['duration'].str.extract(r'(\d+)').astype(float)

# For series (convert "X Seasons"/"X Season" to numeric)
series_df['seasons'] = series_df['duration'].str.extract(r'(\d+)').astype(float)


# In[48]:


# description lengths
movies_df['desc_length'] = movies_df['description'].apply(lambda x: len(str(x)))
series_df['desc_length'] = series_df['description'].apply(lambda x: len(str(x)))


# In[49]:


# famous actors/directors
famous_directors = ['Christopher Nolan', 'Martin Scorsese', 'Steven Spielberg', 'Shah Ruk Khan', 'Robert Downy Junior', 'Leonardo DiCaprio', 'Brad Pitt', 'Morgan Freeman', 'Johnny Depp', 'Robert DeNiro', 'Denzel Washington', 'Tom Hanks', 'Hugh Jackman', 'Al Pacino', 'Bradley Cooper', 'Kate Winslet', 'Natalie Portman', 'Scarlet Johansson', 'Charlize Theron', 'Angelina Jolie',  'Anne Hathaway', 'Nicole Kidman', 'Emma Stone', 'Emily Blunt', 'Julia Roberts', 'Sandra Bullock', 'Rachel McAdams', 'Emma Watson', 'Quentin Tarantino','James Cameron', 'David Fincher', 'Francis Ford', 'Stanley Kubrick', 'Woody Allen', 'Robert Zemeckis', 'David Lynch', 'Roman Polanski', 'Akira Kurosawa']
movies_df['famous_director'] = movies_df['director'].apply(lambda x: 1 if str(x) in famous_directors else 0)
series_df['famous_director'] = series_df['director'].apply(lambda x: 1 if str(x) in famous_directors else 0)


# ----------RECOMMENDATION SYSTEM-------------

# In[69]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[70]:


content_df['description'] = content_df['description'].fillna('')
content_df['genres'] = content_df['genres'].fillna('')

# Combine text features
content_df['text_features'] = content_df['genres'] + " " + content_df['description']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_df['text_features'])

# Cosine Similarity Matrix
cosine_sim_combined = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping (lowercase titles for uniformity)
content_indices = pd.Series(content_df.index, index=content_df['title'].str.lower()).drop_duplicates()


# In[71]:


# Recommendation Function (Combined)
def recommend_content(title, num=5):
    idx = content_indices.get(title.lower())
    if idx is None:
        return "Content not found in database."
    
    sim_scores = list(enumerate(cosine_sim_combined[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    recommended_indices = [i[0] for i in sim_scores]
    
    return content_df.iloc[recommended_indices][['title', 'type', 'genres', 'popularity', 'vote_average']]

# Example usage
recommend_content("sector 36", num=5)


# ------------RATING PREDICTION--------------

# In[78]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#dropping the missing values from the selected attributes
rating_df = content_df[['popularity', 'vote_count', 'vote_average']].dropna()

# Features & Target
X = rating_df.drop('vote_average', axis=1)
y = rating_df['vote_average']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[79]:


#model training and evaluation

# LINEAR REGRESSION
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

print("Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))
print("R² Score:", r2_score(y_test, lr_preds))


# In[81]:


# RANDOM FOREST REGRESSOR 

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("Random Forest Regressor:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("R² Score:", r2_score(y_test, rf_preds))


# In[75]:


# Sample prediction
sample = X_test.iloc[0].values.reshape(1, -1)
predicted_rating = rf_model.predict(sample)
print("Predicted Rating:", predicted_rating[0])
print("Actual Rating:", y_test.iloc[0])


# In[82]:


importances = rf_model.feature_importances_
features = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importances")
plt.show()


# In[92]:


import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler


# In[93]:


joblib.dump(rf_model, 'best_rating_predictor.pkl')


# In[91]:


# Load model and any encoders if saved
model = joblib.load('best_rating_predictor.pkl')

st.title("Netflix Content Rating Predictor")

# User inputs
title = st.text_input("Title")
duration = st.slider("Duration (minutes)", 30, 300, 90)
popularity = st.slider("Popularity", 0, 100, 50)
vote_count = st.number_input("Vote Count", 0)
vote_average = st.slider("Vote Average", 0.0, 10.0, 5.0)
budget = st.number_input("Budget", 0)
revenue = st.number_input("Revenue", 0)
cast_size = st.slider("Cast Size", 1, 30, 5)
release_year = st.slider("Release Year", 1950, 2025, 2020)

if st.button("Predict Rating"):
    # Construct DataFrame
    input_df = pd.DataFrame({
        'duration': [duration],
        'popularity': [popularity],
        'vote_count': [vote_count],
        'vote_average': [vote_average],
        'budget': [budget],
        'revenue': [revenue],
        'cast_size': [cast_size],
        'release_year': [release_year]
    })

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted IMDb Rating: {prediction:.2f}")


# In[ ]:




