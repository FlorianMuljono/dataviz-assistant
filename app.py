import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="DataViz Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS - DARK GREEN THEME
# ============================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #1a5f4a;
        --dark-green: #0d3025;
        --light-green: #2d8a6e;
        --accent-green: #4ecca3;
        --bg-light: #f8faf9;
        --text-dark: #1a1a1a;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0d3025 0%, #1a5f4a 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .status-badge {
        background: #4ecca3;
        color: #0d3025;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #0d3025;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Section headers */
    .section-header {
        color: #1a5f4a;
        font-size: 1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-subtext {
        color: #666;
        font-size: 0.85rem;
        font-weight: 400;
    }
    
    /* File chip */
    .file-chip {
        background: #e8f5f0;
        border: 1px solid #4ecca3;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        color: #1a5f4a;
        margin: 0.5rem 0;
    }
    
    .file-chip-icon {
        font-size: 1.2rem;
    }
    
    .file-chip-size {
        color: #666;
        font-size: 0.75rem;
    }
    
    /* Notes panel styling */
    .notes-panel {
        background: #f8faf9;
        border-radius: 10px;
        padding: 1rem;
        border-left: 3px solid #1a5f4a;
        margin-bottom: 1rem;
    }
    
    .notes-title {
        color: #1a5f4a;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .clarification-box {
        background: #e8f5f0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    
    .clarification-status {
        color: #1a5f4a;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.3rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Button styling - for suggestion pills */
    div[data-testid="column"] .stButton > button {
        background: linear-gradient(135deg, #1a5f4a 0%, #2d8a6e 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.3s;
        width: 100%;
        min-height: 80px;
        white-space: normal;
        line-height: 1.3;
    }
    
    div[data-testid="column"] .stButton > button:hover {
        background: linear-gradient(135deg, #0d3025 0%, #1a5f4a 100%);
        box-shadow: 0 4px 15px rgba(26, 95, 74, 0.4);
        transform: translateY(-2px);
    }
    
    /* Primary action button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a5f4a 0%, #2d8a6e 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0d3025 0%, #1a5f4a 100%);
        box-shadow: 0 4px 12px rgba(26, 95, 74, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #666;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1a5f4a;
        border-bottom: 2px solid #1a5f4a;
        background: transparent;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    
    /* Question display */
    .current-question {
        background: linear-gradient(135deg, #e8f5f0 0%, #d4ede5 100%);
        border-left: 4px solid #1a5f4a;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 1rem;
        color: #0d3025;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Text input styling */
    .stTextArea textarea, .stTextInput input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #1a5f4a;
        box-shadow: 0 0 0 1px #1a5f4a;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #1a5f4a;
        font-weight: 700;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #1a5f4a;
        font-weight: 500;
    }
    
    /* Success/info message styling */
    .stSuccess {
        background-color: #e8f5f0;
        color: #1a5f4a;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #1a5f4a !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INIT
# ============================================
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'webhook_url' not in st.session_state:
    st.session_state.webhook_url = ""
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_code_key' not in st.session_state:
    st.session_state.current_code_key = None
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False

# ============================================
# DATASET CONFIGURATION
# ============================================
DATASETS = {
    "HDB Resale Prices (Singapore)": {
        "file_id": "1HUK7Way3F4LUpgh6vA9I7UeBNHLU0khR",
        "description": "Singapore HDB resale flat transactions from 1990, focusing on approvals in 1990-1999. Each row records a specific flat listing with attributes like town, flat type, block, street, storey range, floor area, flat model, lease start year, and the resale price. It can be used to analyze price patterns by location, flat type/model, size, and age of the flat, and to track trends over time.",
        "icon": "🏠",
        "questions": [
            {"category": "Trends Analysis", "question": "How have resale prices changed over the years?", "code_key": "hdb_price_trend"},
            {"category": "Geographical Analysis", "question": "Which town has the highest average resale price?", "code_key": "hdb_town_price"},
            {"category": "Property Analysis", "question": "How do different flat types compare in terms of average resale price?", "code_key": "hdb_flat_type"},
            {"category": "Lease Analysis", "question": "How does the lease commencement date impact the resale price?", "code_key": "hdb_lease_impact"},
            {"category": "Charting", "question": "Show the average resale price by flat type over the years", "code_key": "hdb_flattype_years"}
        ]
    },
    "Airbnb Listings (New Zealand)": {
        "file_id": "1W-peKALdxBzOHx_muetYIz_2Bb9Vg9Gh",
        "description": "New Zealand Airbnb property listings with details on room types, pricing, availability, reviews, and host information. Useful for travelers planning visits, hosts optimizing their listings, and regulators monitoring short-term rental impacts.",
        "icon": "🏡",
        "questions": [
            {"category": "🧳 For Tourists", "question": "What is the average price by neighbourhood - where are the best value areas?", "code_key": "airbnb_neighbourhood_price"},
            {"category": "🧳 For Tourists", "question": "How does pricing vary by room type and what's the price distribution?", "code_key": "airbnb_price_distribution"},
            {"category": "🧳 For Tourists", "question": "How do prices compare between entire homes vs private rooms by area?", "code_key": "airbnb_price_comparison"},
            {"category": "🏠 For Hosts", "question": "What factors correlate with more reviews (proxy for bookings)?", "code_key": "airbnb_success_factors"},
            {"category": "🏠 For Hosts", "question": "Which neighbourhoods have the highest-rated listings?", "code_key": "airbnb_top_rated_areas"},
            {"category": "📋 For Regulators", "question": "Which areas have the highest concentration of listings (housing impact)?", "code_key": "airbnb_listing_concentration"},
            {"category": "📋 For Regulators", "question": "What percentage of hosts own multiple properties (commercial vs individual)?", "code_key": "airbnb_host_portfolio"},
            {"category": "📋 For Regulators", "question": "What is the average availability - are listings actually available?", "code_key": "airbnb_availability_analysis"}
        ]
    }
}

# ============================================
# VISUALIZATION CODE TEMPLATES
# ============================================
VIZ_CODE_TEMPLATES = {
    "hdb_price_trend": """
# Resale Price Trend Over the Years
df['year'] = pd.to_datetime(df['month']).dt.year
yearly_avg = df.groupby('year')['resale_price'].mean().reset_index()

sns.lineplot(data=yearly_avg, x='year', y='resale_price', marker='o', linewidth=2.5, color='#1a5f4a', ax=ax)
ax.fill_between(yearly_avg['year'], yearly_avg['resale_price'], alpha=0.3, color='#4ecca3')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Average Resale Price ($)', fontsize=12)
ax.set_title('HDB Resale Price Trend Over the Years', fontsize=14, fontweight='bold', color='#0d3025')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

latest = yearly_avg.iloc[-1]
ax.annotate(f'${latest["resale_price"]:,.0f}', xy=(latest['year'], latest['resale_price']),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold', color='#1a5f4a')
""",
    
    "hdb_town_price": """
# Average Resale Price by Town
town_avg = df.groupby('town')['resale_price'].mean().sort_values(ascending=True)

colors = ['#4ecca3' if x < town_avg.max() else '#1a5f4a' for x in town_avg.values]
bars = ax.barh(town_avg.index, town_avg.values, color=colors)
ax.set_xlabel('Average Resale Price ($)', fontsize=12)
ax.set_ylabel('Town', fontsize=12)
ax.set_title('Average HDB Resale Price by Town', fontsize=14, fontweight='bold', color='#0d3025')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
""",
    
    "hdb_flat_type": """
# Average Resale Price by Flat Type
flat_avg = df.groupby('flat_type')['resale_price'].mean().sort_values(ascending=True)

colors = sns.color_palette("Greens", len(flat_avg))
bars = ax.barh(flat_avg.index, flat_avg.values, color=colors)
ax.set_xlabel('Average Resale Price ($)', fontsize=12)
ax.set_ylabel('Flat Type', fontsize=12)
ax.set_title('Average HDB Resale Price by Flat Type', fontsize=14, fontweight='bold', color='#0d3025')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, val in zip(bars, flat_avg.values):
    ax.text(val + 5000, bar.get_y() + bar.get_height()/2, 
            f'${val:,.0f}', va='center', fontsize=9, color='#1a5f4a')
""",
    
    "hdb_lease_impact": """
# Impact of Lease Commencement Date on Resale Price
df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
lease_price = df.groupby('lease_commence_date')['resale_price'].mean().reset_index()
lease_price = lease_price.dropna()

sns.scatterplot(data=lease_price, x='lease_commence_date', y='resale_price', 
                s=80, alpha=0.7, color='#4ecca3', ax=ax)
sns.regplot(data=lease_price, x='lease_commence_date', y='resale_price', 
            scatter=False, color='#1a5f4a', line_kws={'linewidth': 2}, ax=ax)

ax.set_xlabel('Lease Commencement Year', fontsize=12)
ax.set_ylabel('Average Resale Price ($)', fontsize=12)
ax.set_title('Impact of Lease Commencement Date on Resale Price', fontsize=14, fontweight='bold', color='#0d3025')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.05, 0.95, 'Newer leases → Higher prices', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', style='italic', color='#1a5f4a',
        bbox=dict(boxstyle='round', facecolor='#e8f5f0', alpha=0.8, edgecolor='#1a5f4a'))
""",
    
    "hdb_flattype_years": """
# Average Resale Price by Flat Type Over the Years
df['year'] = pd.to_datetime(df['month']).dt.year

top_flat_types = df['flat_type'].value_counts().head(5).index.tolist()
df_filtered = df[df['flat_type'].isin(top_flat_types)]

pivot = df_filtered.groupby(['year', 'flat_type'])['resale_price'].mean().reset_index()

palette = ['#0d3025', '#1a5f4a', '#2d8a6e', '#4ecca3', '#a8e6cf']
sns.lineplot(data=pivot, x='year', y='resale_price', hue='flat_type', 
             marker='o', linewidth=2, palette=palette, ax=ax)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Average Resale Price ($)', fontsize=12)
ax.set_title('Average Resale Price by Flat Type Over the Years', fontsize=14, fontweight='bold', color='#0d3025')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.legend(title='Flat Type', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
""",

    # =============================================
    # AIRBNB NEW ZEALAND - 8 MEANINGFUL QUESTIONS
    # =============================================
    
    # 1. For Tourists: Best value areas
    "airbnb_neighbourhood_price": """
# Average Price by Neighbourhood - Best Value Areas
# Clean price column
df['price_clean'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')

neighbourhood_avg = df.groupby('neighbourhood')['price_clean'].agg(['mean', 'count']).reset_index()
neighbourhood_avg.columns = ['neighbourhood', 'avg_price', 'listings']
neighbourhood_avg = neighbourhood_avg[neighbourhood_avg['listings'] >= 10]  # Min 10 listings
neighbourhood_avg = neighbourhood_avg.sort_values('avg_price', ascending=True).head(20)

colors = sns.color_palette("Greens", len(neighbourhood_avg))
bars = ax.barh(neighbourhood_avg['neighbourhood'], neighbourhood_avg['avg_price'], color=colors)
ax.set_xlabel('Average Price per Night ($)', fontsize=12)
ax.set_ylabel('Neighbourhood', fontsize=12)
ax.set_title('Best Value Neighbourhoods (Lowest Average Prices)', fontsize=14, fontweight='bold', color='#0d3025')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add listing count annotation
for bar, (_, row) in zip(bars, neighbourhood_avg.iterrows()):
    ax.text(row['avg_price'] + 2, bar.get_y() + bar.get_height()/2, 
            f'${row["avg_price"]:.0f} ({row["listings"]:.0f} listings)', 
            va='center', fontsize=8, color='#1a5f4a')
""",

    # 2. For Tourists: Price distribution by room type
    "airbnb_price_distribution": """
# Price Distribution by Room Type
# Clean price column
df['price_clean'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
df_clean = df[df['price_clean'] < df['price_clean'].quantile(0.95)]  # Remove top 5% outliers

fig.clear()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Box plot by room type
palette = ['#0d3025', '#1a5f4a', '#2d8a6e', '#4ecca3']
sns.boxplot(data=df_clean, x='room_type', y='price_clean', palette=palette, ax=axes[0])
axes[0].set_xlabel('Room Type', fontsize=12)
axes[0].set_ylabel('Price per Night ($)', fontsize=12)
axes[0].set_title('Price Distribution by Room Type', fontsize=14, fontweight='bold', color='#0d3025')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
axes[0].tick_params(axis='x', rotation=15)

# Right: Overall histogram
sns.histplot(data=df_clean, x='price_clean', bins=50, kde=True, color='#1a5f4a', ax=axes[1])
axes[1].set_xlabel('Price per Night ($)', fontsize=12)
axes[1].set_ylabel('Number of Listings', fontsize=12)
axes[1].set_title('Overall Price Distribution', fontsize=14, fontweight='bold', color='#0d3025')
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

median_price = df_clean['price_clean'].median()
axes[1].axvline(median_price, color='#e63946', linestyle='--', linewidth=2)
axes[1].text(median_price + 5, axes[1].get_ylim()[1]*0.9, f'Median: ${median_price:.0f}', 
             color='#e63946', fontweight='bold')

plt.tight_layout()
ax = axes[0]  # For compatibility
""",

    # 3. For Tourists: Entire home vs Private room by area
    "airbnb_price_comparison": """
# Entire Home vs Private Room Prices by Top Areas
# Clean price column
df['price_clean'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')

# Focus on main room types
df_filtered = df[df['room_type'].isin(['Entire home/apt', 'Private room'])]

# Get top 12 neighbourhoods by listing count
top_areas = df_filtered['neighbourhood'].value_counts().head(12).index.tolist()
df_top = df_filtered[df_filtered['neighbourhood'].isin(top_areas)]

# Calculate average by neighbourhood and room type
pivot = df_top.groupby(['neighbourhood', 'room_type'])['price_clean'].mean().unstack()
pivot = pivot.sort_values('Entire home/apt', ascending=True)

# Plot grouped bar chart
x = range(len(pivot))
width = 0.35

bars1 = ax.barh([i - width/2 for i in x], pivot['Entire home/apt'], width, label='Entire home/apt', color='#1a5f4a')
bars2 = ax.barh([i + width/2 for i in x], pivot['Private room'], width, label='Private room', color='#4ecca3')

ax.set_yticks(x)
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_xlabel('Average Price per Night ($)', fontsize=12)
ax.set_ylabel('Neighbourhood', fontsize=12)
ax.set_title('Price Comparison: Entire Home vs Private Room by Area', fontsize=14, fontweight='bold', color='#0d3025')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
""",

    # 4. For Hosts: Success factors (reviews correlation)
    "airbnb_success_factors": """
# Factors Correlating with More Reviews (Proxy for Bookings)
# Clean price column
df['price_clean'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')

fig.clear()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Reviews vs Price
sample = df.dropna(subset=['price_clean', 'number_of_reviews'])
sample = sample[sample['price_clean'] < sample['price_clean'].quantile(0.95)]
sample = sample.sample(n=min(2000, len(sample)), random_state=42)

sns.scatterplot(data=sample, x='price_clean', y='number_of_reviews', alpha=0.5, color='#4ecca3', ax=axes[0])
axes[0].set_xlabel('Price per Night ($)', fontsize=12)
axes[0].set_ylabel('Number of Reviews', fontsize=12)
axes[0].set_title('Reviews vs Price', fontsize=14, fontweight='bold', color='#0d3025')

corr = sample[['price_clean', 'number_of_reviews']].corr().iloc[0,1]
axes[0].text(0.95, 0.95, f'Correlation: {corr:.3f}', transform=axes[0].transAxes, 
             fontsize=11, ha='right', va='top', color='#1a5f4a',
             bbox=dict(boxstyle='round', facecolor='#e8f5f0', alpha=0.8))

# Right: Average reviews by room type
room_reviews = df.groupby('room_type')['number_of_reviews'].mean().sort_values(ascending=True)
colors = sns.color_palette("Greens", len(room_reviews))
bars = axes[1].barh(room_reviews.index, room_reviews.values, color=colors)
axes[1].set_xlabel('Average Number of Reviews', fontsize=12)
axes[1].set_ylabel('Room Type', fontsize=12)
axes[1].set_title('Average Reviews by Room Type', fontsize=14, fontweight='bold', color='#0d3025')

for bar, val in zip(bars, room_reviews.values):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                 va='center', fontsize=10, fontweight='bold', color='#1a5f4a')

plt.tight_layout()
ax = axes[0]
""",

    # 5. For Hosts: Top rated areas
    "airbnb_top_rated_areas": """
# Neighbourhoods with Highest-Rated Listings
# Find review score columns
score_cols = [col for col in df.columns if 'review' in col.lower() and 'score' in col.lower()]
review_col = [col for col in df.columns if col in ['review_scores_rating', 'reviews_per_month', 'number_of_reviews']]

if score_cols:
    rating_col = score_cols[0]
    df_rated = df[df[rating_col].notna() & (df[rating_col] > 0)]
    
    # Get areas with enough reviews
    area_ratings = df_rated.groupby('neighbourhood').agg({
        rating_col: 'mean',
        'number_of_reviews': 'sum'
    }).reset_index()
    area_ratings.columns = ['neighbourhood', 'avg_rating', 'total_reviews']
    area_ratings = area_ratings[area_ratings['total_reviews'] >= 50]
    area_ratings = area_ratings.sort_values('avg_rating', ascending=True).tail(15)
    
    colors = sns.color_palette("Greens", len(area_ratings))
    bars = ax.barh(area_ratings['neighbourhood'], area_ratings['avg_rating'], color=colors)
    ax.set_xlabel('Average Rating Score', fontsize=12)
    ax.set_ylabel('Neighbourhood', fontsize=12)
    ax.set_title('Top 15 Highest-Rated Neighbourhoods', fontsize=14, fontweight='bold', color='#0d3025')
    ax.set_xlim(left=area_ratings['avg_rating'].min() - 2)
    
    for bar, (_, row) in zip(bars, area_ratings.iterrows()):
        ax.text(row['avg_rating'] + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{row["avg_rating"]:.1f}', va='center', fontsize=9, fontweight='bold', color='#1a5f4a')
else:
    # Fallback: use number of reviews as proxy
    area_reviews = df.groupby('neighbourhood')['number_of_reviews'].agg(['mean', 'sum']).reset_index()
    area_reviews.columns = ['neighbourhood', 'avg_reviews', 'total_reviews']
    area_reviews = area_reviews[area_reviews['total_reviews'] >= 100]
    area_reviews = area_reviews.sort_values('avg_reviews', ascending=True).tail(15)
    
    colors = sns.color_palette("Greens", len(area_reviews))
    bars = ax.barh(area_reviews['neighbourhood'], area_reviews['avg_reviews'], color=colors)
    ax.set_xlabel('Average Reviews per Listing', fontsize=12)
    ax.set_ylabel('Neighbourhood', fontsize=12)
    ax.set_title('Most Popular Neighbourhoods (by Reviews)', fontsize=14, fontweight='bold', color='#0d3025')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
""",

    # 6. For Regulators: Listing concentration
    "airbnb_listing_concentration": """
# Listing Concentration by Neighbourhood (Housing Impact)
listing_counts = df['neighbourhood'].value_counts().head(20)
listing_counts = listing_counts.sort_values(ascending=True)

total_listings = len(df)
percentages = (listing_counts / total_listings * 100)

colors = sns.color_palette("Greens", len(listing_counts))
bars = ax.barh(listing_counts.index, listing_counts.values, color=colors)
ax.set_xlabel('Number of Listings', fontsize=12)
ax.set_ylabel('Neighbourhood', fontsize=12)
ax.set_title('Listing Concentration by Neighbourhood\\n(Potential Housing Impact Areas)', fontsize=14, fontweight='bold', color='#0d3025')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add percentage annotations
for bar, (area, count) in zip(bars, listing_counts.items()):
    pct = count / total_listings * 100
    ax.text(count + 5, bar.get_y() + bar.get_height()/2, 
            f'{count:,} ({pct:.1f}%)', va='center', fontsize=8, color='#1a5f4a')

# Add summary text
top5_pct = listing_counts.tail(5).sum() / total_listings * 100
ax.text(0.95, 0.05, f'Top 5 areas = {top5_pct:.1f}% of all listings', 
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8, edgecolor='#ffc107'))
""",

    # 7. For Regulators: Commercial vs Individual hosts
    "airbnb_host_portfolio": """
# Commercial vs Individual Hosts Analysis
host_listings = df.groupby('host_id').size().reset_index(name='listing_count')

# Categorize hosts
def categorize_host(count):
    if count == 1:
        return '1 listing (Individual)'
    elif count <= 3:
        return '2-3 listings (Small)'
    elif count <= 10:
        return '4-10 listings (Medium)'
    else:
        return '10+ listings (Commercial)'

host_listings['category'] = host_listings['listing_count'].apply(categorize_host)

# Count hosts in each category
category_counts = host_listings['category'].value_counts()
category_order = ['1 listing (Individual)', '2-3 listings (Small)', '4-10 listings (Medium)', '10+ listings (Commercial)']
category_counts = category_counts.reindex(category_order)

fig.clear()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart of host distribution
colors = ['#4ecca3', '#2d8a6e', '#1a5f4a', '#0d3025']
wedges, texts, autotexts = axes[0].pie(category_counts.values, labels=category_counts.index, 
                                        autopct='%1.1f%%', colors=colors, explode=[0.02]*4)
axes[0].set_title('Distribution of Hosts by Portfolio Size', fontsize=14, fontweight='bold', color='#0d3025')
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

# Right: Listings controlled by each category
listings_by_cat = host_listings.groupby('category')['listing_count'].sum()
listings_by_cat = listings_by_cat.reindex(category_order)
total_listings = listings_by_cat.sum()

bars = axes[1].bar(range(len(listings_by_cat)), listings_by_cat.values, color=colors)
axes[1].set_xticks(range(len(listings_by_cat)))
axes[1].set_xticklabels(['Individual\\n(1)', 'Small\\n(2-3)', 'Medium\\n(4-10)', 'Commercial\\n(10+)'], fontsize=10)
axes[1].set_ylabel('Number of Listings Controlled', fontsize=12)
axes[1].set_title('Listings Controlled by Host Type', fontsize=14, fontweight='bold', color='#0d3025')

for bar, val in zip(bars, listings_by_cat.values):
    pct = val / total_listings * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                 f'{val:,.0f}\\n({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold', color='#1a5f4a')

plt.tight_layout()
ax = axes[0]
""",

    # 8. For Regulators: Availability analysis
    "airbnb_availability_analysis": """
# Availability Analysis - Are Listings Actually Available?
availability_col = [col for col in df.columns if 'availability' in col.lower() and '365' in col]
if not availability_col:
    availability_col = [col for col in df.columns if 'availability' in col.lower()]

if availability_col:
    avail_col = availability_col[0]
    df['availability'] = pd.to_numeric(df[avail_col], errors='coerce')
    
    fig.clear()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Distribution of availability
    sns.histplot(data=df, x='availability', bins=50, kde=True, color='#1a5f4a', ax=axes[0])
    axes[0].set_xlabel('Days Available per Year', fontsize=12)
    axes[0].set_ylabel('Number of Listings', fontsize=12)
    axes[0].set_title('Distribution of Listing Availability', fontsize=14, fontweight='bold', color='#0d3025')
    
    # Add availability categories
    low_avail = (df['availability'] < 30).sum() / len(df) * 100
    med_avail = ((df['availability'] >= 30) & (df['availability'] < 180)).sum() / len(df) * 100
    high_avail = (df['availability'] >= 180).sum() / len(df) * 100
    
    axes[0].text(0.95, 0.95, f'Low (<30 days): {low_avail:.1f}%\\nMedium (30-180): {med_avail:.1f}%\\nHigh (>180): {high_avail:.1f}%', 
                 transform=axes[0].transAxes, fontsize=10, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='#e8f5f0', alpha=0.8))
    
    # Right: Average availability by room type
    room_avail = df.groupby('room_type')['availability'].mean().sort_values(ascending=True)
    colors = sns.color_palette("Greens", len(room_avail))
    bars = axes[1].barh(room_avail.index, room_avail.values, color=colors)
    axes[1].set_xlabel('Average Days Available per Year', fontsize=12)
    axes[1].set_ylabel('Room Type', fontsize=12)
    axes[1].set_title('Average Availability by Room Type', fontsize=14, fontweight='bold', color='#0d3025')
    
    for bar, val in zip(bars, room_avail.values):
        axes[1].text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f} days', 
                     va='center', fontsize=10, fontweight='bold', color='#1a5f4a')
    
    plt.tight_layout()
    ax = axes[0]
else:
    ax.text(0.5, 0.5, 'Availability data not found in dataset', ha='center', va='center', 
            transform=ax.transAxes, fontsize=14, color='#1a5f4a')
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
"""
}

# ============================================
# STYLING FOR PLOTS
# ============================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelcolor'] = '#0d3025'
plt.rcParams['text.color'] = '#0d3025'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ============================================
# LOAD DATA
# ============================================
@st.cache_data(ttl=3600)
def load_dataset(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df, None
    except Exception as e:
        return None, str(e)

# ============================================
# HELPER: Execute visualization code
# ============================================
def execute_viz_code(code, df):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    
    exec_globals = {
        'df': df.copy(),
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'ax': ax,
        'fig': fig
    }
    
    try:
        exec(code, exec_globals)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close(fig)
        return None, str(e)

# ============================================
# HELPER: Get visualization code
# ============================================
def get_viz_code(code_key):
    return VIZ_CODE_TEMPLATES.get(code_key, "ax.text(0.5, 0.5, 'Visualization not available', ha='center', va='center', transform=ax.transAxes)")

# ============================================
# ADMIN PAGE
# ============================================
def render_admin_page():
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Admin Settings</h1>
        <div class="status-badge"><div class="status-dot"></div> Configuration</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Main App"):
        st.session_state.admin_mode = False
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("🔗 Make.com Webhook Configuration")
    
    webhook_input = st.text_input(
        "Webhook URL",
        value=st.session_state.webhook_url,
        type="password",
        help="Enter your Make.com webhook URL for AI-powered custom questions"
    )
    
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.webhook_url = webhook_input
        st.success("✅ Configuration saved!")
    
    st.markdown("---")
    
    st.subheader("📊 Status")
    
    if st.session_state.webhook_url:
        st.success("✅ Make.com webhook configured - Custom questions enabled")
    else:
        st.info("ℹ️ Running in Demo Mode - Pre-built visualizations only")
    
    st.markdown("---")
    
    st.subheader("📁 Configured Datasets")
    
    for name, info in DATASETS.items():
        with st.expander(f"{info['icon']} {name}"):
            st.write(f"**Description:** {info['description'][:200]}...")
            st.write(f"**Questions:** {len(info['questions'])}")

# ============================================
# MAIN APP PAGE
# ============================================
def render_main_app():
    # Custom Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 DataViz Assistant</h1>
        <div class="status-badge"><div class="status-dot"></div> Connected</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset selector at top
    dataset_options = list(DATASETS.keys())
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=dataset_options,
        format_func=lambda x: f"{DATASETS[x]['icon']} {x}",
        label_visibility="collapsed"
    )
    
    dataset_config = DATASETS[selected_dataset]
    file_id = dataset_config["file_id"]
    
    # Load data
    with st.spinner("Loading data..."):
        df, error = load_dataset(file_id)
    
    if error:
        st.error(f"❌ Failed to load data: {error}")
        st.stop()
    
    # Main layout with notes panel on the right
    main_col, notes_col = st.columns([3, 1])
    
    with main_col:
        # Tabs
        tab_overview, tab_ask = st.tabs(["📋 Overview", "💬 Ask Your Own Question"])
        
        with tab_overview:
            # Data Preview Section
            st.markdown("""
            <div class="section-header">
                📄 Data Preview
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df.head(50), use_container_width=True, height=250)
            
            # File chip
            file_size = len(df) * len(df.columns) * 8 / (1024 * 1024)
            st.markdown(f"""
            <div class="file-chip">
                <span class="file-chip-icon">📄</span>
                <span>{selected_dataset.split('(')[0].strip()}</span>
                <span class="file-chip-size">• {len(df):,} rows • ~{file_size:.1f} MB</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Quick Plot Suggestions
            st.markdown(f"""
            <div class="section-header">
                ⚡ Quick Plot Suggestions
                <span class="section-subtext">{len(dataset_config['questions'])} suggestions available</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("Click any suggestion to instantly generate the visualization")
            
            # Suggestion buttons in columns
            cols = st.columns(len(dataset_config['questions']))
            for i, (col, q) in enumerate(zip(cols, dataset_config['questions'])):
                with col:
                    btn_label = q['question']
                    if len(btn_label) > 35:
                        btn_label = btn_label[:35] + "..."
                    
                    if st.button(btn_label, key=f"suggestion_{i}", use_container_width=True):
                        st.session_state.current_question = q['question']
                        st.session_state.current_code_key = q['code_key']
                        st.session_state.show_chart = True
            
            # Show chart if a suggestion was clicked
            if st.session_state.show_chart and st.session_state.current_code_key:
                st.markdown(f"""
                <div class="current-question">
                    🤔 <strong>{st.session_state.current_question}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Generating visualization..."):
                    code = get_viz_code(st.session_state.current_code_key)
                    fig, exec_error = execute_viz_code(code, df)
                    
                    if exec_error:
                        st.error(f"❌ Error: {exec_error}")
                    else:
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        with st.expander("💻 View Generated Code"):
                            st.code(code, language="python")
        
        with tab_ask:
            st.markdown("""
            <div class="section-header">
                💬 Ask Your Own Question
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("Type your question about the data and get a custom visualization.")
            
            user_question = st.text_area(
                "Your question:",
                placeholder="e.g., What is the average price by location? Show me the trend over time...",
                height=100,
                label_visibility="collapsed"
            )
            
            if st.session_state.webhook_url:
                if st.button("🎨 Generate Visualization", type="primary", disabled=not user_question):
                    with st.spinner("AI is analyzing your question..."):
                        # TODO: Call Make.com webhook here
                        st.info("🚧 Custom question feature coming soon! For now, please use the Quick Plot Suggestions.")
            else:
                st.warning("⚠️ To use custom questions, please configure the Make.com webhook in Admin Settings.")
                st.caption("Click the ⚙️ button at the bottom right to access Admin Settings.")
            
            st.markdown("---")
            
            st.markdown("**💡 Sample questions you could ask:**")
            for q in dataset_config['questions'][:3]:
                st.markdown(f"- {q['question']}")
    
    with notes_col:
        # Notes Panel
        st.markdown("""
        <div class="notes-panel">
            <div class="notes-title">📝 Notes</div>
            <p style="font-size: 0.8rem; color: #666; margin: 0;">Preliminary information about your dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clarification box
        st.markdown(f"""
        <div class="clarification-box">
            <div class="clarification-status">✅ File Loaded</div>
            <p style="font-size: 0.8rem; color: #333; line-height: 1.5; margin: 0;">
                {dataset_config['description'][:300]}{'...' if len(dataset_config['description']) > 300 else ''}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("""
        <div class="notes-title">📊 Quick Stats</div>
        """, unsafe_allow_html=True)
        
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        st.metric("Numeric Fields", len(numeric_cols))
        
        # Column list
        with st.expander("📋 View All Columns"):
            for col in df.columns:
                dtype = str(df[col].dtype)
                st.caption(f"• {col} ({dtype})")
    
    # Footer with admin link
    st.markdown("---")
    footer_cols = st.columns([10, 1])
    
    with footer_cols[0]:
        st.caption("Built with Streamlit • Powered by AI • Data Analytics Made Simple")
    
    with footer_cols[1]:
        if st.button("⚙️", help="Admin Settings", key="admin_btn"):
            st.session_state.admin_mode = True
            st.rerun()

# ============================================
# MAIN ROUTER
# ============================================
def main():
    query_params = st.query_params
    if query_params.get("admin") == "true":
        st.session_state.admin_mode = True
    
    if st.session_state.admin_mode:
        render_admin_page()
    else:
        render_main_app()

if __name__ == "__main__":
    main()
