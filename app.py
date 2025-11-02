
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Movie Rating & Recommendation Analyzer",
    page_icon="🎬",
    layout="wide"
)

# TMDB API Configuration
TMDB_API_KEY = "e4822e991b31008d38b52890eee2aa8c" 
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .api-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .csv-badge {
        background-color: #17a2b8;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== API FUNCTIONS ====================
def search_movie_api(query):
    """Search for movies using TMDB API"""
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "en-US"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
    except:
        pass
    return []

def get_movie_details_api(movie_id):
    """Get detailed information about a specific movie"""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_trending_movies_api():
    """Get trending movies (Real-time data from API)"""
    url = f"{TMDB_BASE_URL}/trending/movie/week"
    params = {"api_key": TMDB_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
    except:
        pass
    return []

def get_movie_recommendations_api(movie_id):
    """Get movie recommendations from API"""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
    except:
        pass
    return []

def get_popular_movies_api():
    """Get popular movies in real-time"""
    url = f"{TMDB_BASE_URL}/movie/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("results", [])
    except:
        pass
    return []

# ==================== CSV FUNCTIONS ====================
def load_csv_data(file):
    """Load CSV file"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def clean_data(df):
    """Clean and preprocess data"""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    return df

def get_genre_distribution(df):
    """Analyze genre distribution"""
    if 'genre' in df.columns:
        genres = df['genre'].str.split(',|/|\|', expand=True).stack().str.strip()
        return genres.value_counts()
    return None

def recommend_similar_movies_csv(df, movie_title, top_n=5):
    """Recommend similar movies from CSV based on genre and rating"""
    movie = df[df['title'].str.lower() == movie_title.lower()]
    
    if movie.empty:
        return None
    
    movie = movie.iloc[0]
    recommendations = df.copy()
    recommendations = recommendations[recommendations['title'].str.lower() != movie_title.lower()]
    
    if 'genre' in df.columns and pd.notna(movie['genre']):
        movie_genres = set(str(movie['genre']).lower().split(','))
        recommendations['genre_match'] = recommendations['genre'].apply(
            lambda x: len(set(str(x).lower().split(',')) & movie_genres) if pd.notna(x) else 0
        )
        recommendations = recommendations[recommendations['genre_match'] > 0]
    
    if 'rating' in recommendations.columns:
        recommendations = recommendations.sort_values(['genre_match', 'rating'], ascending=[False, False])
    
    return recommendations.head(top_n)

# ==================== MATPLOTLIB VISUALIZATION FUNCTIONS ====================
def create_rating_distribution_plot(df):
    """Create rating distribution using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'rating' in df.columns:
        ax.hist(df['rating'].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Movie Rating Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    return fig

def create_genre_bar_chart(df):
    """Create genre distribution bar chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    genre_dist = get_genre_distribution(df)
    if genre_dist is not None:
        top_genres = genre_dist.head(10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_genres)))
        ax.barh(top_genres.index, top_genres.values, color=colors)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Genre', fontsize=12)
        ax.set_title('Top 10 Movie Genres', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    return fig

def create_year_trend_plot(df):
    """Create year trend plot using matplotlib"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().sort_index()
        ax.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=6, color='purple')
        ax.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='purple')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Movies', fontsize=12)
        ax.set_title('Movies Released Over Years', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    return fig

def create_rating_vs_year_scatter(df):
    """Create scatter plot of rating vs year using matplotlib"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'year' in df.columns and 'rating' in df.columns:
        scatter = ax.scatter(df['year'], df['rating'], c=df['rating'], 
                           cmap='coolwarm', s=100, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title('Movie Ratings Over Time', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Rating')
        ax.grid(True, alpha=0.3)
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap using matplotlib and seaborn"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.2f', linewidths=1)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    return fig

def create_rating_category_pie(df):
    """Create pie chart for rating categories using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'rating' in df.columns:
        df['rating_category'] = pd.cut(df['rating'], 
                                       bins=[0, 5, 7, 9, 10], 
                                       labels=['Poor (0-5)', 'Average (5-7)', 'Good (7-9)', 'Excellent (9-10)'])
        rating_counts = df['rating_category'].value_counts()
        
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
        ax.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
              colors=colors, startangle=90, textprops={'fontsize': 12})
        ax.set_title('Movies by Rating Category', fontsize=14, fontweight='bold')
    
    return fig

# ==================== MAIN APP ====================
st.title("🎬 Movie Rating & Recommendation Analyzer")
st.markdown("### 📊 Real-time API Data + CSV Analysis with Matplotlib Visualizations")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode selection
    mode = st.radio("Select Mode:", 
                    ["🌐 API Mode (Real-time Data)", "📁 CSV Mode (Upload Dataset)", "🔄 Both Modes"])
    
    st.markdown("---")
    
    if mode in ["🌐 API Mode (Real-time Data)", "🔄 Both Modes"]:
        st.subheader("🔑 API Setup")
        if TMDB_API_KEY == "your_api_key_here":
            st.warning("⚠️ Add your TMDB API key!")
            st.info("Get free API key from:\nhttps://www.themoviedb.org/settings/api")
        else:
            st.success("✅ API Key configured!")
    
    if mode in ["📁 CSV Mode (Upload Dataset)", "🔄 Both Modes"]:
        st.subheader("📁 Upload CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write("**API Mode:** Fetches real-time movie data from TMDB")
    st.write("**CSV Mode:** Analyzes your uploaded movie dataset")
    st.write("**Matplotlib:** Advanced visualizations")

# ==================== API MODE ====================
if mode in ["🌐 API Mode (Real-time Data)", "🔄 Both Modes"]:
    st.header("🌐 API Mode - Real-time Movie Data")
    st.markdown('<span class="api-badge">LIVE DATA</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    api_tab1, api_tab2, api_tab3, api_tab4 = st.tabs([
        "🔍 Search Movies", 
        "🔥 Trending Now", 
        "⭐ Popular Movies",
        "💡 Get Recommendations"
    ])
    
    with api_tab1:
        st.subheader("Search Movies (Real-time)")
        search_query = st.text_input("Enter movie name:", placeholder="e.g., Inception")
        
        if st.button("🔍 Search", type="primary"):
            if search_query:
                with st.spinner("Fetching real-time data from TMDB API..."):
                    movies = search_movie_api(search_query)
                    
                    if movies:
                        st.success(f"Found {len(movies)} movies in real-time!")
                        
                        for movie in movies[:5]:
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                poster = movie.get("poster_path")
                                if poster:
                                    st.image(f"{IMAGE_BASE_URL}{poster}", use_container_width=True)
                            
                            with col2:
                                st.markdown(f"### {movie.get('title', 'Unknown')}")
                                st.write(f"⭐ **Rating:** {movie.get('vote_average', 0):.1f}/10")
                                st.write(f"📅 **Release:** {movie.get('release_date', 'N/A')}")
                                st.write(f"📊 **Popularity:** {movie.get('popularity', 0):.1f}")
                                st.write(f"**Overview:** {movie.get('overview', 'No overview')[:200]}...")
                            
                            st.markdown("---")
                    else:
                        st.warning("No movies found!")
    
    with api_tab2:
        st.subheader("🔥 Trending Movies This Week")
        
        if st.button("Load Trending", type="primary"):
            with st.spinner("Fetching trending movies..."):
                trending = get_trending_movies_api()
                
                if trending:
                    st.success(f"✅ Loaded {len(trending)} trending movies!")
                    
                    # Convert to DataFrame for matplotlib
                    trending_df = pd.DataFrame(trending)
                    
                    # Matplotlib visualization
                    st.subheader("📊 Trending Movies Ratings (Matplotlib)")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    titles = [m['title'][:20] for m in trending[:10]]
                    ratings = [m['vote_average'] for m in trending[:10]]
                    colors = plt.cm.plasma(np.linspace(0, 1, len(titles)))
                    
                    ax.barh(titles, ratings, color=colors)
                    ax.set_xlabel('Rating', fontsize=12)
                    ax.set_ylabel('Movie', fontsize=12)
                    ax.set_title('Top 10 Trending Movies', fontsize=14, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    
                    # Show details
                    for movie in trending[:5]:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if movie.get("poster_path"):
                                st.image(f"{IMAGE_BASE_URL}{movie['poster_path']}", use_container_width=True)
                        with col2:
                            st.markdown(f"### {movie['title']}")
                            st.write(f"⭐ {movie['vote_average']:.1f}/10 | 📅 {movie.get('release_date', 'N/A')}")
                        st.markdown("---")
    
    with api_tab3:
        st.subheader("⭐ Popular Movies Right Now")
        
        if st.button("Load Popular", type="primary"):
            with st.spinner("Fetching popular movies..."):
                popular = get_popular_movies_api()
                
                if popular:
                    cols = st.columns(3)
                    for idx, movie in enumerate(popular[:9]):
                        with cols[idx % 3]:
                            if movie.get("poster_path"):
                                st.image(f"{IMAGE_BASE_URL}{movie['poster_path']}", use_container_width=True)
                            st.write(f"**{movie['title']}**")
                            st.write(f"⭐ {movie['vote_average']:.1f}/10")
    
    with api_tab4:
        st.subheader("💡 Get Recommendations (API)")
        search_for_rec = st.text_input("Search movie for recommendations:", placeholder="e.g., Avatar")
        
        if st.button("Get Recommendations", type="primary"):
            if search_for_rec:
                movies = search_movie_api(search_for_rec)
                if movies:
                    movie_id = movies[0]['id']
                    recommendations = get_movie_recommendations_api(movie_id)
                    
                    if recommendations:
                        st.success(f"Recommendations for '{movies[0]['title']}':")
                        cols = st.columns(3)
                        for idx, rec in enumerate(recommendations[:6]):
                            with cols[idx % 3]:
                                if rec.get("poster_path"):
                                    st.image(f"{IMAGE_BASE_URL}{rec['poster_path']}", use_container_width=True)
                                st.write(f"**{rec['title']}**")
                                st.write(f"⭐ {rec['vote_average']:.1f}/10")

# ==================== CSV MODE ====================
if mode in ["📁 CSV Mode (Upload Dataset)", "🔄 Both Modes"]:
    st.header("📁 CSV Mode - Dataset Analysis")
    st.markdown('<span class="csv-badge">OFFLINE ANALYSIS</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    if 'uploaded_file' in locals() and uploaded_file is not None:
        df = load_csv_data(uploaded_file)
        
        if df is not None:
            df = clean_data(df)
            st.success(f"✅ Loaded {len(df)} movies from CSV!")
            
            csv_tab1, csv_tab2, csv_tab3, csv_tab4 = st.tabs([
                "📊 Overview & Stats",
                "📈 Matplotlib Visualizations",
                "🔍 Search & Filter",
                "💡 Recommendations"
            ])
            
            with csv_tab1:
                st.subheader("Dataset Overview")
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Movies", len(df))
                with col2:
                    if 'rating' in df.columns:
                        st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
                with col3:
                    if 'year' in df.columns:
                        st.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")
                with col4:
                    st.metric("Columns", len(df.columns))
                
                # Statistics
                st.subheader("📊 Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)
            
            with csv_tab2:
                st.subheader("📈 Matplotlib Visualizations")
                
                viz_option = st.selectbox("Select Visualization:", [
                    "Rating Distribution",
                    "Genre Bar Chart",
                    "Year Trend",
                    "Rating vs Year Scatter",
                    "Correlation Heatmap",
                    "Rating Category Pie Chart"
                ])
                
                if viz_option == "Rating Distribution":
                    fig = create_rating_distribution_plot(df)
                    st.pyplot(fig)
                
                elif viz_option == "Genre Bar Chart":
                    fig = create_genre_bar_chart(df)
                    st.pyplot(fig)
                
                elif viz_option == "Year Trend":
                    fig = create_year_trend_plot(df)
                    st.pyplot(fig)
                
                elif viz_option == "Rating vs Year Scatter":
                    fig = create_rating_vs_year_scatter(df)
                    st.pyplot(fig)
                
                elif viz_option == "Correlation Heatmap":
                    fig = create_correlation_heatmap(df)
                    st.pyplot(fig)
                
                elif viz_option == "Rating Category Pie Chart":
                    fig = create_rating_category_pie(df)
                    st.pyplot(fig)
            
            with csv_tab3:
                st.subheader("🔍 Search & Filter")
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'title' in df.columns:
                        search_term = st.text_input("Search by Title:", "")
                with col2:
                    if 'rating' in df.columns:
                        min_rating = st.slider("Minimum Rating:", 
                                              float(df['rating'].min()), 
                                              float(df['rating'].max()), 
                                              float(df['rating'].min()))
                
                filtered_df = df.copy()
                if 'title' in df.columns and search_term:
                    filtered_df = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
                if 'rating' in df.columns:
                    filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
                
                st.subheader(f"Found {len(filtered_df)} movies")
                st.dataframe(filtered_df, use_container_width=True)
            
            with csv_tab4:
                st.subheader("💡 Movie Recommendations (CSV)")
                
                if 'title' in df.columns:
                    selected_movie = st.selectbox("Select a movie:", df['title'].unique())
                    num_rec = st.slider("Number of recommendations:", 1, 10, 5)
                    
                    if st.button("Get Similar Movies", type="primary"):
                        recommendations = recommend_similar_movies_csv(df, selected_movie, num_rec)
                        
                        if recommendations is not None and not recommendations.empty:
                            st.success(f"Movies similar to '{selected_movie}':")
                            st.dataframe(recommendations[['title', 'genre', 'rating', 'year'] 
                                                        if all(c in recommendations.columns for c in ['title', 'genre', 'rating', 'year']) 
                                                        else recommendations], 
                                       use_container_width=True)
    else:
        st.info("👈 Please upload a CSV file to begin analysis")
        
        # Sample data
        sample_data = pd.DataFrame({
            'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction'],
            'genre': ['Sci-Fi/Action', 'Sci-Fi/Thriller', 'Sci-Fi/Drama', 'Action/Crime', 'Crime/Drama'],
            'rating': [8.7, 8.8, 8.6, 9.0, 8.9],
            'year': [1999, 2010, 2014, 2008, 1994],
            'director': ['Wachowski', 'Nolan', 'Nolan', 'Nolan', 'Tarantino']
        })
        
        st.subheader("Sample Dataset Format:")
        st.dataframe(sample_data, use_container_width=True)
        
        csv = sample_data.to_csv(index=False)
        st.download_button("📥 Download Sample CSV", data=csv, file_name="sample_movies.csv", mime="text/csv")

st.markdown("---")
st.markdown("**🎬 Movie Analyzer** | API: TMDB | Visualizations: Matplotlib & Plotly | Made with Streamlit")