import streamlit as st
import pandas as pd
import joblib

# Set page layout to wide
st.set_page_config(layout="wide")

# Load preprocessed data
content_df = joblib.load("content_df.pkl")
cosine_sim = joblib.load("cosine_sim.pkl")

# Language code to name mapping
lang_map = {
    'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'as': 'Assamese', 'az': 'Azerbaijani',
    'bg': 'Bulgarian', 'bn': 'Bengali', 'bs': 'Bosnian', 'ca': 'Catalan', 'cn': 'Chinese (Simplified)',
    'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'dz': 'Dzongkha', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian',
    'fi': 'Finnish', 'fr': 'French', 'ga': 'Irish', 'gl': 'Galician', 'he': 'Hebrew',
    'hi': 'Hindi', 'hr': 'Croatian', 'ht': 'Haitian Creole', 'hu': 'Hungarian', 'hy': 'Armenian',
    'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'ja': 'Japanese', 'ka': 'Georgian',
    'kk': 'Kazakh', 'kl': 'Kalaallisut', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean',
    'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lt': 'Lithuanian',
    'lv': 'Latvian', 'mi': 'Māori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian',
    'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'nb': 'Norwegian Bokmål', 'ne': 'Nepali',
    'nl': 'Dutch', 'no': 'Norwegian', 'or': 'Odia', 'pa': 'Punjabi', 'pl': 'Polish',
    'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'si': 'Sinhala',
    'sk': 'Slovak', 'sl': 'Slovene', 'sr': 'Serbian', 'sv': 'Swedish', 'ta': 'Tamil',
    'te': 'Telugu', 'th': 'Thai', 'tl': 'Tagalog', 'tr': 'Turkish', 'uk': 'Ukrainian',
    'ur': 'Urdu', 'vi': 'Vietnamese', 'xx': 'Unknown', 'yo': 'Yoruba', 'za': 'Zhuang',
    'zh': 'Chinese (Traditional)', 'zu': 'Zulu'
}

# Title and navigation
st.title("🎥 Netflix & TV Show Content Explorer")
page = st.radio("Navigate to:", ["Home", "🎬 Recommendation", "🎭 Top by Genre", "🌐 Top by Language"])

# ------------------------ PAGE 1: HOME ------------------------
if page == "Home":
    st.markdown("Welcome to **Netflix Content Explorer**! Choose an option from above to explore:")
    st.markdown("""
    - 🎬 **Recommendation**: Get recommendations based on a movie/TV show you like  
    - 🎭 **Top by Genre**: Select any genre and explore top-rated content globally  
    - 🌐 **Top by Language**: Pick a language to see its best-rated shows/movies
    """)

# ------------------------ PAGE 2: RECOMMENDATION ------------------------
elif page == "🎬 Recommendation":
    user_input = st.text_input("Enter a Movie or TV Show Title:")

    def recommend_by_title(title):
        idx = content_df[content_df['title'].str.lower() == title.lower()].index
        if len(idx) == 0:
            return None
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        return content_df.iloc[[i[0] for i in sim_scores]][['title', 'type', 'genres', 'director', 'vote_average']]

    if user_input:
        idx = content_df[content_df['title'].str.lower() == user_input.lower()].index
        if not idx.empty:
            genre = content_df.loc[idx[0], 'genres']
            rating = content_df.loc[idx[0], 'vote_average']
            st.markdown(f"### 🎯 You selected: `{user_input}`")
            st.markdown(f"- **Genre:** {genre if pd.notna(genre) else 'Unknown'}")
            st.markdown(f"- **Average Rating:** {rating if pd.notna(rating) else 'Not Available'}")
        else:
            st.warning("❌ Title not found. Please check spelling or try another.")

        recommendations = recommend_by_title(user_input)
        if recommendations is not None:
            st.write("✅ **Top 5 Recommendations:**")
            st.dataframe(recommendations.reset_index(drop=True))
        else:
            st.warning("No similar recommendations found.")

    st.markdown("""
    ---
    ℹ️ **Note:** Some entries may show `None` in the fields.  
    This is due to missing or incomplete data in the original dataset.
    """)

# ------------------------ PAGE 3: TOP BY GENRE ------------------------
elif page == "🎭 Top by Genre":
    genre_list = sorted(set(g.strip() for sublist in content_df['genres'].dropna().str.split(',') for g in sublist))
    selected_genre = st.selectbox("Select Genre", genre_list)

    if selected_genre:
        filtered = content_df[content_df['genres'].str.contains(selected_genre, na=False, case=False)]
        top_20 = filtered.sort_values(by="vote_average", ascending=False).head(20)
        st.write(f"🎭 **Top 20 {selected_genre} Movies/TV Shows:**")
        st.dataframe(top_20[['title', 'language', 'vote_average', 'type']].reset_index(drop=True))

        st.markdown("""
        ---
        ℹ️ **Note:** Some entries may show `None` in the fields.  
        This is due to missing or incomplete data in the original dataset.
        """)

# ------------------------ PAGE 4: TOP BY LANGUAGE ------------------------
elif page == "🌐 Top by Language":
    unique_langs = sorted(content_df['language'].dropna().unique())
    lang_display = [f"{lang_map.get(code, code)} ({code})" for code in unique_langs]
    lang_selection = st.selectbox("Select Language", lang_display)

    selected_code = lang_selection.split("(")[-1].strip(")")
    filtered = content_df[content_df['language'] == selected_code]
    top_20 = filtered.sort_values(by="vote_average", ascending=False).head(20)

    st.write(f"🌐 **Top 20 Movies/TV Shows in {lang_selection}:**")
    st.dataframe(top_20[['title', 'genres', 'vote_average', 'type']].reset_index(drop=True))

    st.markdown("""
    ---
    ℹ️ **Note:** Some entries may show `None` in the fields.  
    This is due to missing or incomplete data in the original dataset.
    """)
