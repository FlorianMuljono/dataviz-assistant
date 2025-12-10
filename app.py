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
    layout="wide"
)

# ============================================
# SESSION STATE INIT
# ============================================
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'webhook_url' not in st.session_state:
    st.session_state.webhook_url = ""

# ============================================
# DATASET CONFIGURATION WITH CURATED QUESTIONS
# ============================================
DATASETS = {
    "HDB Resale Prices (Singapore)": {
        "file_id": "1HUK7Way3F4LUpgh6vA9I7UeBNHLU0khR",
        "description": "Singapore HDB resale flat transactions",
        "icon": "🏠",
        "questions": [
            {
                "category": "Trends Analysis",
                "question": "How have resale prices changed over the years?",
                "code_key": "hdb_price_trend"
            },
            {
                "category": "Geographical Analysis",
                "question": "Which town has the highest average resale price?",
                "code_key": "hdb_town_price"
            },
            {
                "category": "Property Analysis",
                "question": "How do different flat types compare in terms of average resale price?",
                "code_key": "hdb_flat_type"
            },
            {
                "category": "Lease Analysis",
                "question": "How does the lease commencement date impact the resale price?",
                "code_key": "hdb_lease_impact"
            },
            {
                "category": "Charting",
                "question": "Show the average resale price by flat type over the years",
                "code_key": "hdb_flattype_years"
            }
        ]
    },
    "Airbnb Listings (New Zealand)": {
        "file_id": "1W-peKALdxBzOHx_muetYIz_2Bb9Vg9Gh",
        "description": "New Zealand Airbnb property listings",
        "icon": "🏡",
        "questions": [
            {
                "category": "Pricing Analysis",
                "question": "What is the average price by room type?",
                "code_key": "airbnb_room_price"
            },
            {
                "category": "Geographical Analysis", 
                "question": "Which areas have the highest average listing prices?",
                "code_key": "airbnb_area_price"
            },
            {
                "category": "Availability Analysis",
                "question": "How does availability vary across different room types?",
                "code_key": "airbnb_availability"
            },
            {
                "category": "Reviews Analysis",
                "question": "What is the relationship between number of reviews and price?",
                "code_key": "airbnb_reviews_price"
            },
            {
                "category": "Host Analysis",
                "question": "Which hosts have the most listings?",
                "code_key": "airbnb_top_hosts"
            }
        ]
    }
}

# ============================================
# VISUALIZATION CODE TEMPLATES
# ============================================
VIZ_CODE_TEMPLATES = {
    # HDB Templates
    "hdb_price_trend": """
# Resale Price Trend Over the Years
df['year'] = pd.to_datetime(df['month']).dt.year
yearly_avg = df.groupby('year')['resale_price'].mean().reset_index()

sns.lineplot(data=yearly_avg, x='year', y='resale_price', marker='o', linewidth=2.5, color='#2563eb', ax=ax)
ax.fill_between(yearly_avg['year'], yearly_avg['resale_price'], alpha=0.3, color='#2563eb')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Average Resale Price ($)', fontsize=12)
ax.set_title('HDB Resale Price Trend Over the Years', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add annotation for latest price
latest = yearly_avg.iloc[-1]
ax.annotate(f'${latest["resale_price"]:,.0f}', xy=(latest['year'], latest['resale_price']),
            xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')
""",
    
    "hdb_town_price": """
# Average Resale Price by Town
town_avg = df.groupby('town')['resale_price'].mean().sort_values(ascending=True)

colors = sns.color_palette("viridis", len(town_avg))
bars = ax.barh(town_avg.index, town_avg.values, color=colors)
ax.set_xlabel('Average Resale Price ($)', fontsize=12)
ax.set_ylabel('Town', fontsize=12)
ax.set_title('Average HDB Resale Price by Town', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Highlight highest
bars[-1].set_color('#e63946')
""",
    
    "hdb_flat_type": """
# Average Resale Price by Flat Type
flat_avg = df.groupby('flat_type')['resale_price'].mean().sort_values(ascending=True)

colors = sns.color_palette("rocket", len(flat_avg))
bars = ax.barh(flat_avg.index, flat_avg.values, color=colors)
ax.set_xlabel('Average Resale Price ($)', fontsize=12)
ax.set_ylabel('Flat Type', fontsize=12)
ax.set_title('Average HDB Resale Price by Flat Type', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add value labels
for bar, val in zip(bars, flat_avg.values):
    ax.text(val + 5000, bar.get_y() + bar.get_height()/2, 
            f'${val:,.0f}', va='center', fontsize=9)
""",
    
    "hdb_lease_impact": """
# Impact of Lease Commencement Date on Resale Price
df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
lease_price = df.groupby('lease_commence_date')['resale_price'].mean().reset_index()
lease_price = lease_price.dropna()

sns.scatterplot(data=lease_price, x='lease_commence_date', y='resale_price', 
                s=80, alpha=0.7, color='#2563eb', ax=ax)
sns.regplot(data=lease_price, x='lease_commence_date', y='resale_price', 
            scatter=False, color='#e63946', line_kws={'linewidth': 2}, ax=ax)

ax.set_xlabel('Lease Commencement Year', fontsize=12)
ax.set_ylabel('Average Resale Price ($)', fontsize=12)
ax.set_title('Impact of Lease Commencement Date on Resale Price', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add trend annotation
ax.text(0.05, 0.95, 'Newer leases → Higher prices', transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
""",
    
    "hdb_flattype_years": """
# Average Resale Price by Flat Type Over the Years
df['year'] = pd.to_datetime(df['month']).dt.year

# Get top flat types
top_flat_types = df['flat_type'].value_counts().head(5).index.tolist()
df_filtered = df[df['flat_type'].isin(top_flat_types)]

pivot = df_filtered.groupby(['year', 'flat_type'])['resale_price'].mean().reset_index()

sns.lineplot(data=pivot, x='year', y='resale_price', hue='flat_type', 
             marker='o', linewidth=2, ax=ax)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Average Resale Price ($)', fontsize=12)
ax.set_title('Average Resale Price by Flat Type Over the Years', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.legend(title='Flat Type', bbox_to_anchor=(1.02, 1), loc='upper left')
""",

    # Airbnb Templates
    "airbnb_room_price": """
# Average Price by Room Type
room_avg = df.groupby('room_type')['price'].mean().sort_values(ascending=True)

colors = sns.color_palette("viridis", len(room_avg))
bars = ax.barh(room_avg.index, room_avg.values, color=colors)
ax.set_xlabel('Average Price ($)', fontsize=12)
ax.set_ylabel('Room Type', fontsize=12)
ax.set_title('Average Airbnb Price by Room Type', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add value labels
for bar, val in zip(bars, room_avg.values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, 
            f'${val:,.0f}', va='center', fontsize=10, fontweight='bold')
""",
    
    "airbnb_area_price": """
# Top 15 Areas by Average Price
area_avg = df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(15)
area_avg = area_avg.sort_values(ascending=True)

colors = sns.color_palette("rocket", len(area_avg))
bars = ax.barh(area_avg.index, area_avg.values, color=colors)
ax.set_xlabel('Average Price ($)', fontsize=12)
ax.set_ylabel('Neighbourhood', fontsize=12)
ax.set_title('Top 15 Areas with Highest Average Airbnb Prices', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
""",
    
    "airbnb_availability": """
# Availability by Room Type
availability_col = [col for col in df.columns if 'availability' in col.lower()]
if availability_col:
    avail_col = availability_col[0]
    room_avail = df.groupby('room_type')[avail_col].mean().sort_values(ascending=True)
    
    colors = sns.color_palette("Set2", len(room_avail))
    bars = ax.barh(room_avail.index, room_avail.values, color=colors)
    ax.set_xlabel('Average Availability (days)', fontsize=12)
    ax.set_ylabel('Room Type', fontsize=12)
    ax.set_title('Average Availability by Room Type', fontsize=14, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Availability column not found', ha='center', va='center', transform=ax.transAxes)
""",
    
    "airbnb_reviews_price": """
# Relationship between Reviews and Price
sample = df.sample(n=min(3000, len(df)), random_state=42)

sns.scatterplot(data=sample, x='number_of_reviews', y='price', 
                alpha=0.5, s=30, color='#2563eb', ax=ax)
                
ax.set_xlabel('Number of Reviews', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.set_title('Relationship: Number of Reviews vs Price', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add correlation
corr = sample[['number_of_reviews', 'price']].corr().iloc[0,1]
ax.text(0.95, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
        fontsize=11, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
""",
    
    "airbnb_top_hosts": """
# Top 15 Hosts by Number of Listings
host_col = [col for col in df.columns if 'host' in col.lower() and 'name' in col.lower()]
if host_col:
    host_counts = df[host_col[0]].value_counts().head(15)
else:
    host_counts = df['host_id'].value_counts().head(15)

host_counts = host_counts.sort_values(ascending=True)

colors = sns.color_palette("viridis", len(host_counts))
bars = ax.barh(range(len(host_counts)), host_counts.values, color=colors)
ax.set_yticks(range(len(host_counts)))
ax.set_yticklabels(host_counts.index.astype(str), fontsize=9)
ax.set_xlabel('Number of Listings', fontsize=12)
ax.set_ylabel('Host', fontsize=12)
ax.set_title('Top 15 Hosts with Most Listings', fontsize=14, fontweight='bold')

# Add count labels
for bar, val in zip(bars, host_counts.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{val}', va='center', fontsize=9)
"""
}

# ============================================
# STYLING
# ============================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# ============================================
# LOAD DATA
# ============================================
@st.cache_data(ttl=3600)
def load_dataset(file_id):
    """Load dataset from Google Drive"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df, None
    except Exception as e:
        return None, str(e)

# ============================================
# HELPER: Analyze dataset structure
# ============================================
def analyze_dataset(df):
    """Analyze dataset and return structured info"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }

# ============================================
# HELPER: Get data summary for LLM
# ============================================
def get_data_summary(df, dataset_name):
    """Create a concise summary for the LLM"""
    analysis = analyze_dataset(df)
    
    summary = f"""
Dataset: {dataset_name}
Total Rows: {len(df):,}
Total Columns: {len(df.columns)}

Columns: {', '.join(df.columns.tolist())}

Numeric columns: {', '.join(analysis['numeric'])}
Categorical columns: {', '.join(analysis['categorical'])}

Sample data (first 5 rows):
{df.head().to_string()}
"""
    return summary

# ============================================
# HELPER: Call Make.com webhook
# ============================================
def call_makecom_webhook(webhook_url, payload):
    """Send request to Make.com webhook and get response"""
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        return response.text, None
    except Exception as e:
        return None, str(e)

# ============================================
# HELPER: Execute generated code safely
# ============================================
def execute_viz_code(code, df):
    """Execute the generated visualization code"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
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
# GET VISUALIZATION CODE
# ============================================
def get_viz_code(code_key, question, df):
    """Get visualization code from templates or generate fallback"""
    
    if code_key in VIZ_CODE_TEMPLATES:
        return VIZ_CODE_TEMPLATES[code_key]
    
    # Fallback for custom questions - basic visualization
    analysis = analyze_dataset(df)
    numeric_cols = analysis['numeric']
    
    if numeric_cols:
        col = numeric_cols[0]
        return f"""
# Distribution of {col}
sns.histplot(data=df, x='{col}', bins=50, kde=True, ax=ax, color='#2563eb')
ax.set_xlabel('{col.replace('_', ' ').title()}', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of {col.replace('_', ' ').title()}', fontsize=14, fontweight='bold')
"""
    
    return "ax.text(0.5, 0.5, 'Unable to generate visualization', ha='center', va='center', transform=ax.transAxes)"


# ============================================
# ADMIN PAGE
# ============================================
def render_admin_page():
    """Render the admin configuration page"""
    st.title("⚙️ Admin Settings")
    
    st.markdown("---")
    
    # Back to main app link
    if st.button("← Back to Main App"):
        st.session_state.admin_mode = False
        st.rerun()
    
    st.markdown("---")
    
    st.header("🔗 Make.com Configuration")
    
    webhook_input = st.text_input(
        "Make.com Webhook URL",
        value=st.session_state.webhook_url,
        type="password",
        help="Enter your Make.com webhook URL for AI-powered features"
    )
    
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.webhook_url = webhook_input
        st.success("✅ Configuration saved!")
    
    st.markdown("---")
    
    # Status display
    st.header("📊 Current Status")
    
    if st.session_state.webhook_url:
        st.success("✅ Make.com webhook configured")
        st.code(st.session_state.webhook_url[:20] + "..." if len(st.session_state.webhook_url) > 20 else st.session_state.webhook_url)
    else:
        st.info("ℹ️ Running in Demo Mode - Using pre-built visualizations")
    
    st.markdown("---")
    
    # Dataset info
    st.header("📁 Configured Datasets")
    
    for name, info in DATASETS.items():
        with st.expander(f"{info['icon']} {name}"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**File ID:** `{info['file_id']}`")
            st.write(f"**Questions:** {len(info['questions'])}")
            for q in info['questions']:
                st.write(f"  - {q['question']}")
    
    st.markdown("---")
    st.caption("Admin Settings • DataViz Assistant")


# ============================================
# MAIN APP PAGE
# ============================================
def render_main_app():
    """Render the main application page"""
    
    # Header
    st.title("📊 DataViz Assistant")
    st.markdown("*Select a dataset, ask questions, get beautiful visualizations*")
    
    # ============================================
    # STEP 1: SELECT DATASET
    # ============================================
    st.header("📁 Step 1: Select Dataset")
    
    dataset_options = list(DATASETS.keys())
    selected_dataset = st.selectbox(
        "Choose a dataset to analyze:",
        options=dataset_options,
        format_func=lambda x: f"{DATASETS[x]['icon']} {x}"
    )
    
    # Load selected dataset
    dataset_config = DATASETS[selected_dataset]
    file_id = dataset_config["file_id"]
    
    with st.spinner(f"Loading {selected_dataset}..."):
        df, error = load_dataset(file_id)
    
    if error:
        st.error(f"❌ Failed to load data: {error}")
        st.stop()
    
    st.success(f"✅ Loaded **{len(df):,}** records")
    
    # Data preview
    with st.expander("👀 Preview Data", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            analysis = analyze_dataset(df)
            st.metric("Numeric Columns", len(analysis['numeric']))
        
        st.markdown("**Column Names:**")
        st.code(", ".join(df.columns.tolist()))
    
    st.markdown("---")
    
    # ============================================
    # STEP 2: SELECT QUESTION
    # ============================================
    st.header("🤔 Step 2: Choose a Question")
    
    # Get curated questions for this dataset
    questions = dataset_config["questions"]
    
    # Display as nice cards/buttons
    st.markdown("**Select a business question:**")
    
    # Create formatted options
    question_options = []
    for q in questions:
        question_options.append(f"**({q['category']})** {q['question']}")
    
    question_options.append("✍️ Ask my own question...")
    
    selected_idx = st.radio(
        "Questions:",
        options=range(len(question_options)),
        format_func=lambda x: question_options[x],
        label_visibility="collapsed"
    )
    
    # Handle selection
    if selected_idx == len(questions):  # Custom question
        selected_question = st.text_input(
            "Type your question:",
            placeholder="e.g., What is the average price by location?"
        )
        code_key = "custom"
    else:
        selected_question = questions[selected_idx]["question"]
        code_key = questions[selected_idx]["code_key"]
    
    st.markdown("---")
    
    # ============================================
    # STEP 3: GENERATE VISUALIZATION
    # ============================================
    st.header("📊 Step 3: Generate Visualization")
    
    if st.button("🎨 Generate Chart", type="primary", disabled=not selected_question):
        with st.spinner("Creating your visualization..."):
            
            if st.session_state.webhook_url and code_key == "custom":
                # Use Make.com for custom questions
                data_summary = get_data_summary(df, selected_dataset)
                
                payload = {
                    "action": "generate_code",
                    "question": selected_question,
                    "data_summary": data_summary,
                    "columns": df.columns.tolist()
                }
                
                code, error = call_makecom_webhook(st.session_state.webhook_url, payload)
                
                if error:
                    st.error(f"Webhook error: {error}")
                    st.info("Using fallback visualization...")
                    code = get_viz_code(code_key, selected_question, df)
            else:
                # Use pre-built templates
                code = get_viz_code(code_key, selected_question, df)
            
            # Show generated code
            with st.expander("💻 View Generated Code", expanded=False):
                st.code(code, language="python")
            
            # Execute and display
            fig, exec_error = execute_viz_code(code, df)
            
            if exec_error:
                st.error(f"❌ Visualization error: {exec_error}")
                st.markdown("**Debug info:**")
                st.code(code, language="python")
            else:
                st.pyplot(fig)
                plt.close(fig)
                st.success("✅ Visualization generated!")
    
    # ============================================
    # FOOTER WITH SUBTLE ADMIN LINK
    # ============================================
    st.markdown("---")
    
    footer_cols = st.columns([8, 1])
    
    with footer_cols[0]:
        st.caption("Built with Streamlit • Data from Google Drive • AI-powered analysis")
    
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


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    main()
