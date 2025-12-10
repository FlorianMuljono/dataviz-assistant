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
# DATASET CONFIGURATION
# Add new datasets here easily!
# ============================================
DATASETS = {
    "HDB Resale Prices (Singapore)": {
        "file_id": "1HUK7Way3F4LUpgh6vA9I7UeBNHLU0khR",
        "description": "Singapore HDB resale flat transactions",
        "icon": "🏠"
    },
    "Airbnb Listings (New Zealand)": {
        "file_id": "1W-peKALdxBzOHx_muetYIz_2Bb9Vg9Gh",
        "description": "New Zealand Airbnb property listings",
        "icon": "🏡"
    }
    # ADD MORE DATASETS HERE:
    # "Dataset Name": {
    #     "file_id": "your_google_drive_file_id",
    #     "description": "Brief description",
    #     "icon": "emoji"
    # }
}

# ============================================
# STYLING
# ============================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# ============================================
# LOAD DATA (cached for performance)
# ============================================
@st.cache_data(ttl=3600)
def load_dataset(file_id):
    """Load dataset from Google Drive"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    
    try:
        df = pd.read_csv(url)
        # Clean column names - lowercase, replace spaces with underscores
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        return df, None
    except Exception as e:
        return None, str(e)

# ============================================
# HELPER: Analyze dataset structure
# ============================================
def analyze_dataset(df):
    """Analyze dataset and return structured info"""
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to detect date columns stored as strings
    for col in categorical_cols:
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else ""
        if isinstance(sample, str) and any(x in sample for x in ['-', '/']) and len(sample) <= 10:
            try:
                pd.to_datetime(df[col].iloc[:100])
                datetime_cols.append(col)
            except:
                pass
    
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

Numeric columns statistics:
{df[analysis['numeric']].describe().to_string() if analysis['numeric'] else 'No numeric columns'}

Categorical columns unique values:
"""
    for col in analysis['categorical'][:8]:
        unique_count = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:5].tolist()
        summary += f"\n- {col}: {unique_count} unique values. Examples: {sample_vals}"
    
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
# DEMO MODE: Generate smart questions based on data
# ============================================
def generate_smart_questions(df, dataset_name):
    """Generate relevant questions based on actual dataset structure"""
    
    analysis = analyze_dataset(df)
    questions = []
    
    numeric_cols = analysis['numeric']
    categorical_cols = analysis['categorical']
    
    # Pattern 1: Distribution of main numeric column
    if numeric_cols:
        main_numeric = numeric_cols[0]
        questions.append(f"What is the distribution of {main_numeric.replace('_', ' ')}?")
    
    # Pattern 2: Average numeric by category
    if numeric_cols and categorical_cols:
        questions.append(f"What is the average {numeric_cols[0].replace('_', ' ')} by {categorical_cols[0].replace('_', ' ')}?")
    
    # Pattern 3: Top categories
    if categorical_cols and numeric_cols:
        questions.append(f"Which {categorical_cols[0].replace('_', ' ')} has the highest {numeric_cols[0].replace('_', ' ')}?")
    
    # Pattern 4: Correlation (if multiple numeric)
    if len(numeric_cols) >= 2:
        questions.append(f"What is the relationship between {numeric_cols[0].replace('_', ' ')} and {numeric_cols[1].replace('_', ' ')}?")
    
    # Pattern 5: Category distribution
    if categorical_cols:
        questions.append(f"What is the distribution of {categorical_cols[0].replace('_', ' ')}?")
    
    return questions[:5]  # Return max 5 questions

# ============================================
# DEMO MODE: Generate visualization code
# ============================================
def generate_demo_code(question, df):
    """Generate visualization code based on question and actual data structure"""
    
    analysis = analyze_dataset(df)
    numeric_cols = analysis['numeric']
    categorical_cols = analysis['categorical']
    
    question_lower = question.lower()
    
    # Detect what type of visualization is needed
    
    # DISTRIBUTION of numeric
    if "distribution" in question_lower and numeric_cols:
        for col in numeric_cols:
            if col.replace('_', ' ') in question_lower:
                target_col = col
                break
        else:
            target_col = numeric_cols[0]
        
        return f"""
# Distribution of {target_col}
sns.histplot(data=df, x='{target_col}', bins=50, kde=True, ax=ax, color='#2563eb')
ax.set_xlabel('{target_col.replace('_', ' ').title()}', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of {target_col.replace('_', ' ').title()}', fontsize=14, fontweight='bold')

# Add median line
median_val = df['{target_col}'].median()
ax.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {{median_val:,.2f}}')
ax.legend()
"""
    
    # AVERAGE by category
    if "average" in question_lower and numeric_cols and categorical_cols:
        # Find which columns are mentioned
        target_numeric = numeric_cols[0]
        target_cat = categorical_cols[0]
        
        for col in numeric_cols:
            if col.replace('_', ' ') in question_lower:
                target_numeric = col
                break
        
        for col in categorical_cols:
            if col.replace('_', ' ') in question_lower:
                target_cat = col
                break
        
        return f"""
# Average {target_numeric} by {target_cat}
grouped = df.groupby('{target_cat}')['{target_numeric}'].mean().sort_values(ascending=True)

# Limit to top 20 if too many categories
if len(grouped) > 20:
    grouped = grouped.tail(20)

colors = sns.color_palette("viridis", len(grouped))
bars = ax.barh(grouped.index.astype(str), grouped.values, color=colors)
ax.set_xlabel('Average {target_numeric.replace('_', ' ').title()}', fontsize=12)
ax.set_ylabel('{target_cat.replace('_', ' ').title()}', fontsize=12)
ax.set_title('Average {target_numeric.replace('_', ' ').title()} by {target_cat.replace('_', ' ').title()}', fontsize=14, fontweight='bold')
"""
    
    # HIGHEST / TOP
    if ("highest" in question_lower or "top" in question_lower) and numeric_cols and categorical_cols:
        target_numeric = numeric_cols[0]
        target_cat = categorical_cols[0]
        
        for col in numeric_cols:
            if col.replace('_', ' ') in question_lower:
                target_numeric = col
                break
        
        for col in categorical_cols:
            if col.replace('_', ' ') in question_lower:
                target_cat = col
                break
        
        return f"""
# Top 10 {target_cat} by {target_numeric}
grouped = df.groupby('{target_cat}')['{target_numeric}'].mean().sort_values(ascending=False).head(10)

colors = sns.color_palette("rocket", len(grouped))
bars = ax.bar(range(len(grouped)), grouped.values, color=colors)
ax.set_xticks(range(len(grouped)))
ax.set_xticklabels(grouped.index.astype(str), rotation=45, ha='right')
ax.set_xlabel('{target_cat.replace('_', ' ').title()}', fontsize=12)
ax.set_ylabel('Average {target_numeric.replace('_', ' ').title()}', fontsize=12)
ax.set_title('Top 10 {target_cat.replace('_', ' ').title()} by {target_numeric.replace('_', ' ').title()}', fontsize=14, fontweight='bold')
"""
    
    # RELATIONSHIP / CORRELATION
    if ("relationship" in question_lower or "correlation" in question_lower) and len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        
        for col in numeric_cols:
            if col.replace('_', ' ') in question_lower:
                if col1 == numeric_cols[0]:
                    col1 = col
                else:
                    col2 = col
        
        return f"""
# Relationship between {col1} and {col2}
sample = df.sample(n=min(5000, len(df)), random_state=42)

sns.regplot(data=sample, x='{col1}', y='{col2}', 
            scatter_kws={{'alpha':0.3, 's':10}}, 
            line_kws={{'color':'red', 'linewidth':2}}, ax=ax)
ax.set_xlabel('{col1.replace('_', ' ').title()}', fontsize=12)
ax.set_ylabel('{col2.replace('_', ' ').title()}', fontsize=12)
ax.set_title('Relationship: {col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()}', fontsize=14, fontweight='bold')

# Add correlation coefficient
corr = df[['{col1}', '{col2}']].corr().iloc[0,1]
ax.text(0.05, 0.95, f'Correlation: {{corr:.3f}}', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
"""
    
    # CATEGORY DISTRIBUTION (pie/bar)
    if "distribution" in question_lower and categorical_cols:
        target_cat = categorical_cols[0]
        
        for col in categorical_cols:
            if col.replace('_', ' ') in question_lower:
                target_cat = col
                break
        
        return f"""
# Distribution of {target_cat}
counts = df['{target_cat}'].value_counts().head(10)

colors = sns.color_palette("Set2", len(counts))
bars = ax.bar(range(len(counts)), counts.values, color=colors)
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(counts.index.astype(str), rotation=45, ha='right')
ax.set_xlabel('{target_cat.replace('_', ' ').title()}', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of {target_cat.replace('_', ' ').title()}', fontsize=14, fontweight='bold')

# Add count labels
for i, (bar, val) in enumerate(zip(bars, counts.values)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{{val:,}}', ha='center', va='bottom', fontsize=9)
"""
    
    # DEFAULT: First numeric column distribution
    if numeric_cols:
        col = numeric_cols[0]
        return f"""
# Overview: Distribution of {col}
sns.histplot(data=df, x='{col}', bins=50, kde=True, ax=ax, color='#2563eb')
ax.set_xlabel('{col.replace('_', ' ').title()}', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of {col.replace('_', ' ').title()}', fontsize=14, fontweight='bold')
"""
    
    # FALLBACK: Just show value counts of first categorical
    if categorical_cols:
        col = categorical_cols[0]
        return f"""
# Overview: {col} breakdown
counts = df['{col}'].value_counts().head(10)
ax.bar(range(len(counts)), counts.values, color=sns.color_palette("Set2", len(counts)))
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(counts.index.astype(str), rotation=45, ha='right')
ax.set_title('{col.replace('_', ' ').title()} Breakdown', fontsize=14, fontweight='bold')
"""

    return "ax.text(0.5, 0.5, 'Unable to generate visualization', ha='center', va='center', transform=ax.transAxes)"

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.title("📊 DataViz Assistant")
    st.markdown("*Select a dataset, ask questions, get beautiful visualizations*")
    
    # ============================================
    # SIDEBAR
    # ============================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Make.com webhook URL (optional)
        webhook_url = st.text_input(
            "Make.com Webhook URL (optional)",
            type="password",
            help="Leave empty for Demo Mode"
        )
        
        if not webhook_url:
            st.info("🎮 Running in Demo Mode")
        
        st.markdown("---")
        st.markdown("### 📁 Available Datasets")
        for name, info in DATASETS.items():
            st.markdown(f"{info['icon']} **{name}**")
            st.caption(info['description'])
    
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
    file_id = DATASETS[selected_dataset]["file_id"]
    
    with st.spinner(f"Loading {selected_dataset}..."):
        df, error = load_dataset(file_id)
    
    if error:
        st.error(f"❌ Failed to load data: {error}")
        st.stop()
    
    # Success message
    st.success(f"✅ Loaded **{len(df):,}** records from {selected_dataset}")
    
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
    # STEP 2: SELECT OR ASK QUESTION
    # ============================================
    st.header("🤔 Step 2: Choose a Question")
    
    # Generate smart questions based on THIS dataset
    suggested_questions = generate_smart_questions(df, selected_dataset)
    
    st.markdown("**Suggested questions for this dataset:**")
    
    # Radio buttons for suggested questions
    question_choice = st.radio(
        "Select a question:",
        options=suggested_questions + ["✍️ Ask my own question..."],
        label_visibility="collapsed"
    )
    
    # Custom question input
    if question_choice == "✍️ Ask my own question...":
        selected_question = st.text_input(
            "Type your question:",
            placeholder="e.g., What is the average price by location?"
        )
    else:
        selected_question = question_choice
    
    st.markdown("---")
    
    # ============================================
    # STEP 3: GENERATE VISUALIZATION
    # ============================================
    st.header("📊 Step 3: Generate Visualization")
    
    if st.button("🎨 Generate Chart", type="primary", disabled=not selected_question):
        with st.spinner("Creating your visualization..."):
            
            if webhook_url:
                # Use Make.com for AI-generated code
                data_summary = get_data_summary(df, selected_dataset)
                
                payload = {
                    "action": "generate_code",
                    "question": selected_question,
                    "data_summary": data_summary,
                    "columns": df.columns.tolist(),
                    "numeric_columns": analyze_dataset(df)['numeric'],
                    "categorical_columns": analyze_dataset(df)['categorical']
                }
                
                code, error = call_makecom_webhook(webhook_url, payload)
                
                if error:
                    st.error(f"Webhook error: {error}")
                    st.info("Falling back to Demo Mode...")
                    code = generate_demo_code(selected_question, df)
            else:
                # Demo mode
                code = generate_demo_code(selected_question, df)
            
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
    # FOOTER
    # ============================================
    st.markdown("---")
    st.caption("Built with Streamlit • Data from Google Drive • AI-powered analysis")

# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    main()
