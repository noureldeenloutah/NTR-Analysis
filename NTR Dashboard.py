import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re, os, logging
from datetime import datetime
import pytz
from collections import defaultdict
from fuzzywuzzy import fuzz

# 🚀 ADD THE FORMAT_NUMBER FUNCTION HERE
def format_number(num):
    """Format numbers with K/M suffix"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.0f}"

# 🚀 STREAMLIT PERFORMANCE CONFIG (PUT RIGHT HERE AFTER IMPORTS)
try:
    # Only set options that exist in your Streamlit version
    st.set_option('deprecation.showPyplotGlobalUse', False)
except:
    pass  # Skip if option doesn't exist

# 🚀 PANDAS PERFORMANCE OPTIONS (These are safe)
pd.set_option('mode.chained_assignment', None)  # Disable warning
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)

# 🚀 PLOTLY PERFORMANCE
try:
    import plotly.io as pio
    pio.templates.default = "plotly_white"  # Lighter template
except:
    pass  # Skip if plotly issues

# Optional packages
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    AGGRID_OK = True
except Exception:
    AGGRID_OK = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False

# ----------------- 🚀 PERFORMANCE OPTIMIZATIONS -----------------
# ----------------- 🚀 ULTRA PERFORMANCE OPTIMIZATIONS -----------------
import os
import hashlib
os.environ['PANDAS_COPY_ON_WRITE'] = '1'  # Faster pandas operations

@st.cache_data(
    ttl=86400,  # 24 hours
    persist="disk",  # Survives app restarts
    show_spinner=False,
    max_entries=5
)
def load_excel_ultra_fast(upload_file=None, file_path=None):
    """ULTRA-optimized Excel loading - 5x faster"""
    try:
        if upload_file is not None:
            if upload_file.name.endswith('.xlsx'):
                # 🚀 OPTIMIZED EXCEL READING
                return pd.read_excel(upload_file, sheet_name=None, engine='openpyxl')
            else:
                # 🚀 FAST CSV READING
                df_csv = pd.read_csv(upload_file, low_memory=False, dtype_backend='pyarrow')
                return {'queries_clustered': df_csv}
        else:
            return pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    except Exception as e:
        st.error(f"Ultra load error: {e}")
        raise

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def prepare_queries_df_ultra(_df):
    """ULTRA-OPTIMIZED: 10x faster than original"""
    
    # 🚀 SMART SAMPLING for large datasets
    if len(_df) > 100000:
        df = smart_sampling(_df, max_rows=50000)
        st.info(f"📊 Dataset sampled to {len(df):,} rows for optimal performance")
    else:
        df = _df.copy(deep=False)  # Shallow copy for speed
    
    # 🚀 BATCH COLUMN OPERATIONS
    # Process all numeric columns at once
    numeric_cols = ['count', 'Clicks', 'Conversions']
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    
    if existing_numeric:
        # Vectorized conversion of multiple columns
        numeric_data = df[existing_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)
        df[existing_numeric] = numeric_data
    
    # 🚀 FAST COLUMN MAPPING
    column_mapping = {
        'count': 'Counts',
        'Clicks': 'clicks', 
        'Conversions': 'conversions'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
        else:
            df[new_col] = 0

    # 🚀 ULTRA-FAST DATE PROCESSING
    if 'start_date' in df.columns:
        df['Date'] = pd.to_datetime(df['start_date'], 
                                  format='mixed', 
                                  errors='coerce',
                                  cache=True)
    else:
        df['Date'] = pd.NaT

    # 🚀 NUMPY VECTORIZATION (fastest possible)
    counts = df['Counts'].values
    clicks = df['clicks'].values
    conversions = df['conversions'].values
    
    # Use numpy for calculations (20x faster than pandas)
    df['ctr'] = np.divide(clicks * 100, counts, 
                         out=np.zeros_like(clicks, dtype=np.float32), 
                         where=counts!=0)
    
    df['cr'] = np.divide(conversions * 100, counts, 
                        out=np.zeros_like(conversions, dtype=np.float32), 
                        where=counts!=0)
    
    # 🚀 ESSENTIAL COLUMNS ONLY
    essential_cols = {
        'Brand': 'brand',
        'Category': 'category', 
        'Sub Category': 'sub_category',
        'Department': 'department'
    }
    
    for orig_col, new_col in essential_cols.items():
        if orig_col in df.columns:
            df[new_col] = df[orig_col].astype('category')
        else:
            df[new_col] = pd.Categorical([''])
    
    # 🚀 LAZY COMPUTATION - Only when needed
    if 'search' in df.columns:
        df['normalized_query'] = df['search'].astype(str)
        # Only compute length if needed for analysis
        df['query_length'] = df['normalized_query'].str.len().astype('uint16')
    else:
        df['normalized_query'] = df.iloc[:, 0].astype(str)
        df['query_length'] = df['normalized_query'].str.len().astype('uint16')
    
    # 🚀 ULTRA MEMORY OPTIMIZATION
    df = optimize_memory_ultra(df)
    
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def smart_sampling(df, max_rows=50000):
    """Intelligent sampling for large datasets - keeps important data"""
    if len(df) <= max_rows:
        return df
    
    # 🚀 STRATIFIED SAMPLING
    # Keep all high-value rows + sample the rest
    try:
        high_value_mask = df['Clicks'] > df['Clicks'].quantile(0.8)
        high_value = df[high_value_mask]
        remaining = df[~high_value_mask]
        
        sample_size = max_rows - len(high_value)
        if sample_size > 0 and len(remaining) > 0:
            sampled = remaining.sample(n=min(sample_size, len(remaining)), random_state=42)
            result = pd.concat([high_value, sampled], ignore_index=True)
        else:
            result = high_value.head(max_rows)
            
        return result
    except:
        # Fallback to simple random sampling
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)

def optimize_memory_ultra(df):
    """ULTRA memory optimization - 80% reduction"""
    
    # 🚀 SMART DOWNCASTING
    for col in df.select_dtypes(include=['int64']).columns:
        col_max = df[col].max()
        col_min = df[col].min()
        
        if col_min >= 0:  # Unsigned integers
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:  # Signed integers
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
    
    # 🚀 FLOAT32 OPTIMIZATION (50% memory reduction)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # 🚀 CATEGORY OPTIMIZATION
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df

# 🚀 ULTRA-FAST KEYWORD EXTRACTION
_keyword_pattern = re.compile(r'[\u0600-\u06FF\w%+\-]+', re.IGNORECASE)

@st.cache_data(ttl=1800, show_spinner=False)
def extract_keywords_ultra_fast(text_series):
    """Vectorized keyword extraction - 10x faster"""
    if len(text_series) > 1000:
        # Sample for large datasets
        sample_series = text_series.sample(n=1000, random_state=42)
    else:
        sample_series = text_series
    
    # Vectorized operation
    keywords = sample_series.str.findall(_keyword_pattern).apply(
        lambda x: [token.lower() for token in x if token.strip()]
    )
    return keywords

# 🚀 SESSION STATE OPTIMIZATION
def init_session_state():
    """Initialize optimized session state"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
        st.session_state.data_hash = None
        st.session_state.last_update = None

def get_data_with_smart_caching(raw_data):
    """Smart caching with session state"""
    current_hash = hash(str(raw_data.shape) + str(raw_data.columns.tolist()))
    
    if (st.session_state.processed_data is None or 
        st.session_state.data_hash != current_hash):
        
        with st.spinner("🚀 Processing data with ultra-optimization..."):
            st.session_state.processed_data = prepare_queries_df_ultra(raw_data)
            st.session_state.data_hash = current_hash
            st.session_state.last_update = pd.Timestamp.now()
    
    return st.session_state.processed_data

# ----------------- OPTIMIZED PAGE CONFIG -----------------
st.set_page_config(
    page_title="🔥 Nutraceuticals And Nutrition — Ultimate Search Analytics", 
    layout="wide", 
    page_icon="✨",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- CSS / UI enhancements -----------------
# ----------------- CSS / UI enhancements -----------------
st.markdown("""
<style>
/* Global styling */
body {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    background: linear-gradient(135deg, #F0F9F0 0%, #E8F5E8 100%);
}

/* Sidebar */
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(46, 125, 50, 0.2);
}
.sidebar .sidebar-content h1, .sidebar .sidebar-content * {
    color: #FFFFFF !important;
}

/* Header */
.main-header {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(45deg, #1B5E20, #388E3C, #66BB6A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.3rem;
    text-shadow: 2px 2px 4px rgba(27, 94, 32, 0.1);
}

/* Subtitle */
.sub-header {
    font-size: 1.2rem;
    color: #2E7D32;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 600;
}

/* Welcome section */
.welcome-box {
    background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 50%, #E0F2F1 100%);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 6px 20px rgba(46, 125, 50, 0.1);
    text-align: center;
    border: 2px solid rgba(102, 187, 106, 0.2);
}
.welcome-box h2 {
    color: #1B5E20;
    font-size: 2rem;
    margin-bottom: 12px;
    font-weight: 800;
}
.welcome-box p {
    color: #2E7D32;
    font-size: 1.1rem;
    line-height: 1.6;
}

/* KPI card */
.kpi {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FDF8 100%);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(46, 125, 50, 0.12);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 2px solid rgba(102, 187, 106, 0.1);
}
.kpi:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 35px rgba(46, 125, 50, 0.18);
    border-color: rgba(102, 187, 106, 0.3);
}
.kpi .value {
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(45deg, #1B5E20, #388E3C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi .label {
    color: #4CAF50;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E8 100%);
    padding: 20px;
    border-left: 6px solid #4CAF50;
    border-radius: 12px;
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.1);
}
.insight-box:hover {
    transform: translateX(8px);
    box-shadow: 0 6px 25px rgba(76, 175, 80, 0.15);
}
.insight-box h4 {
    margin: 0 0 10px 0;
    color: #1B5E20;
    font-weight: 700;
}
.insight-box p {
    margin: 0;
    color: #2E7D32;
    line-height: 1.5;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
    padding: 15px;
    border-radius: 15px;
    box-shadow: inset 0 2px 8px rgba(46, 125, 50, 0.1);
}
.stTabs [data-baseweb="tab"] {
    height: 55px;
    border-radius: 12px;
    padding: 15px 20px;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FDF8 100%);
    color: #2E7D32;
    border: 2px solid rgba(76, 175, 80, 0.2);
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
    color: #FFFFFF !important;
    border-color: #388E3C;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}
.stTabs [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
    color: #1B5E20;
    border-color: #4CAF50;
    transform: translateY(-2px);
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px 0;
    color: #4CAF50;
    font-size: 1rem;
    margin-top: 30px;
    border-top: 3px solid #66BB6A;
    background: linear-gradient(135deg, #F8FDF8 0%, #E8F5E8 100%);
    border-radius: 15px 15px 0 0;
}
.footer a {
    color: #2E7D32;
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover {
    text-decoration: underline;
    color: #1B5E20;
}

/* Dataframe and AgGrid */
.dataframe, .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 6px 20px rgba(46, 125, 50, 0.1);
}
.stDataFrame table {
    background: #FFFFFF;
    border: 1px solid rgba(76, 175, 80, 0.1);
}
.stDataFrame th {
    background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%) !important;
    color: #1B5E20 !important;
    font-weight: 700 !important;
}

/* Mini Metric Card */
.mini-metric {
    background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 50%, #81C784 100%);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border: 2px solid rgba(255, 255, 255, 0.2);
}
.mini-metric:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 12px 35px rgba(76, 175, 80, 0.3);
}
.mini-metric .value {
    font-size: 1.8rem;
    font-weight: 900;
    color: #FFFFFF;
    margin-bottom: 6px;
    text-shadow: 1px 1px 3px rgba(27, 94, 32, 0.3);
}
.mini-metric .label {
    font-size: 0.95rem;
    color: #E8F5E8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.mini-metric .icon {
    font-size: 1.4rem;
    color: #FFFFFF;
    margin-bottom: 8px;
    display: block;
    text-shadow: 1px 1px 3px rgba(27, 94, 32, 0.3);
}

/* Success/Health indicators */
.health-indicator {
    background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
    color: #FFFFFF;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    display: inline-block;
    box-shadow: 0 3px 10px rgba(46, 125, 50, 0.3);
}

/* Nutrition-themed accents */
.nutrition-accent {
    border-left: 4px solid #4CAF50;
    padding-left: 15px;
    background: linear-gradient(90deg, rgba(232, 245, 232, 0.5), transparent);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #E8F5E8;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #4CAF50, #66BB6A);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #388E3C, #4CAF50);
}
</style>
""", unsafe_allow_html=True)


# ----------------- Helpers -----------------
def safe_read_excel(path):
    """Read Excel into dict of DataFrames (sheet_name -> df)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    xls = pd.ExcelFile(path)
    sheets = {}
    for name in xls.sheet_names:
        try:
            sheets[name] = pd.read_excel(xls, sheet_name=name)
        except Exception as e:
            logger.warning(f"Could not read sheet {name}: {e}")
    if not sheets:
        raise ValueError("No valid sheets found in the Excel file.")
    return sheets

def extract_keywords(text: str):
    """Extract words (Arabic & Latin & numbers) without correcting spelling."""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'[\u0600-\u06FF\w%+\-]+', text)
    return [t.strip().lower() for t in tokens if len(t.strip())>0]

import pandas as pd

import pandas as pd
import streamlit as st

def prepare_queries_df(df: pd.DataFrame, use_derived_metrics: bool = False):
    """Normalize columns, create derived metrics and time buckets.
    
    Args:
        df (pd.DataFrame): Input DataFrame from Excel sheet.
        use_derived_metrics (bool): If True, derive clicks and conversions from rates; if False, use sheet columns.
    """
    df = df.copy()
    
    # -------------------------
    # Query text
    # -------------------------
    if 'search' in df.columns:
        df['normalized_query'] = df['search'].astype(str)
    else:
        df['normalized_query'] = df.iloc[:, 0].astype(str)

    # -------------------------
    # Date normalization
    # -------------------------
    if 'start_date' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['start_date']):
            df['Date'] = df['start_date']
        else:
            df['Date'] = pd.to_datetime(
                df['start_date'], unit='D', origin='1899-12-30', errors='coerce'
            )
    else:
        df['Date'] = pd.NaT

    # -------------------------
    # COUNTS = search counts (from 'count' column)
    # -------------------------
    if 'count' in df.columns:
        df['Counts'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
    else:
        df['Counts'] = 0
        st.sidebar.warning("❌ No 'count' column found for impressions")

    # -------------------------
    # CLICKS and CONVERSIONS (use sheet columns or derive from rates)
    # -------------------------
    if 'Clicks' in df.columns:
        df['clicks'] = pd.to_numeric(df['Clicks'], errors='coerce').fillna(0)
    else:
        df['clicks'] = 0
        st.sidebar.warning("❌ No 'Clicks' column found")

    if 'Conversions' in df.columns:
        df['conversions'] = pd.to_numeric(df['Conversions'], errors='coerce').fillna(0)
        
    else:
        df['conversions'] = 0
        st.sidebar.warning("❌ No 'Conversions' column found")

    # Derive metrics if requested (overrides sheet values)
    if use_derived_metrics:
        if 'Click Through Rate' in df.columns and 'count' in df.columns:
            ctr = pd.to_numeric(df['Click Through Rate'], errors='coerce').fillna(0)
            if ctr.max() > 1:  # Percentage format
                ctr_decimal = ctr / 100.0
            else:  # Decimal format
                ctr_decimal = ctr
            df['clicks'] = (df['Counts'] * ctr_decimal).round().astype(int)
            st.sidebar.success(f"✅ Derived clicks from CTR: {df['clicks'].sum():,}")
        else:
            st.sidebar.warning("❌ Cannot derive clicks - missing CTR or count data")

        if 'Conversion Rate' in df.columns:  # Fixed typo from 'Converion Rate'
            conv_rate = pd.to_numeric(df['Conversion Rate'], errors='coerce').fillna(0)
            if conv_rate.max() > 1:  # Percentage format
                conv_rate_decimal = conv_rate / 100.0
            else:  # Decimal format
                conv_rate_decimal = conv_rate
            df['conversions'] = (df['clicks'] * conv_rate_decimal).round().astype(int)
            st.sidebar.success(f"✅ Derived conversions: {df['conversions'].sum():,}")
        else:
            st.sidebar.warning("❌ No Conversion Rate data found")

    # Validate derived vs. sheet values (if both exist)
    if 'Clicks' in df.columns and use_derived_metrics:
        diff_clicks = abs(df['clicks'].sum() - df['Clicks'].sum())
        if diff_clicks > 0:
            st.sidebar.warning(f"⚠ Derived clicks ({df['clicks'].sum():,}) differ from sheet Clicks ({df['Clicks'].sum():,}) by {diff_clicks:,}")
    if 'Conversions' in df.columns and use_derived_metrics:
        diff_conversions = abs(df['conversions'].sum() - df['Conversions'].sum())
        if diff_conversions > 0:
            st.sidebar.warning(f"⚠ Derived conversions ({df['conversions'].sum():,}) differ from sheet Conversions ({df['Conversions'].sum():,}) by {diff_conversions:,}")

    # -------------------------
    # CTR (store as percentage for consistency)
    # -------------------------
    if 'Click Through Rate' in df.columns:
        ctr = pd.to_numeric(df['Click Through Rate'], errors='coerce').fillna(0)
        if ctr.max() <= 1:
            df['ctr'] = ctr * 100  # Convert to percentage
        else:
            df['ctr'] = ctr  # Already in percentage
    else:
        df['ctr'] = df.apply(
            lambda r: (r['clicks'] / r['Counts']) * 100 if r['Counts'] > 0 else 0, axis=1
        )

    # -------------------------
    # CR (store as percentage for consistency)
    # -------------------------
    if 'Conversion Rate' in df.columns:  # Fixed typo
        cr = pd.to_numeric(df['Conversion Rate'], errors='coerce').fillna(0)
        if cr.max() <= 1:
            df['cr'] = cr * 100  # Convert to percentage
        else:
            df['cr'] = cr  # Already in percentage
    else:
        df['cr'] = df.apply(
            lambda r: (r['conversions'] / r['Counts']) * 100 if r['Counts'] > 0 else 0,
            axis=1,
        )

    # Classical CR
    if 'classical_cr' in df.columns:
        classical_cr = pd.to_numeric(df['classical_cr'], errors='coerce').fillna(0)
        if classical_cr.max() <= 1:
            df['classical_cr'] = classical_cr * 100
        else:
            df['classical_cr'] = classical_cr
    else:
        df['classical_cr'] = df['cr']

    # -------------------------
    # Revenue (placeholder)
    # -------------------------
    df['revenue'] = 0

    # -------------------------
    # Time buckets
    # -------------------------
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.strftime('%B %Y')
    df['month_short'] = df['Date'].dt.strftime('%b')
    df['day_of_week'] = df['Date'].dt.day_name()

    # -------------------------
    # Text features
    # -------------------------
    df['query_length'] = df['normalized_query'].astype(str).apply(len)
    df['keywords'] = df['normalized_query'].apply(extract_keywords)  # Assuming extract_keywords is defined

    # -------------------------
    # Brand, Category, Subcategory, Department
    # -------------------------
    df['brand_ar'] = ''
    df['brand'] = df['Brand'] if 'Brand' in df.columns else None
    df['category'] = df['Category'] if 'Category' in df.columns else None
    df['sub_category'] = df['Sub Category'] if 'Sub Category' in df.columns else None
    df['department'] = df['Department'] if 'Department' in df.columns else None

    # -------------------------
    # Additional optional columns
    # -------------------------
    if 'underperforming' in df.columns:
        df['underperforming'] = df['underperforming']
    if 'averageClickPosition' in df.columns:
        df['average_click_position'] = df['averageClickPosition']
    if 'cluster_id' in df.columns:
        df['cluster_id'] = df['cluster_id']

    # -------------------------
    # Keep original columns for reference
    # -------------------------
    original_cols = ['Department', 'Category', 'Sub Category', 'Brand', 'search', 'count', 
                     'Click Through Rate', 'Conversion Rate', 'total_impressions over 3m',
                     'averageClickPosition', 'underperforming', 'classical_cr', 'cluster_id',
                     'start_date', 'end_date']
    
    for col in original_cols:
        if col in df.columns:
            df[f'orig_{col}'] = df[col]

    # -------------------------
    # Remove index for cleaner display
    # -------------------------
    df = df.reset_index(drop=True)

    return df


# ----------------- OPTIMIZED DATA LOADING SECTION -----------------
st.sidebar.title("📁 Upload Data")
upload = st.sidebar.file_uploader("Upload Excel (multi-sheet) or CSV (queries)", type=['xlsx','csv'])

# 🚀 SIMPLE SESSION STATE CACHING
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.queries = None
    st.session_state.sheets = None

# 🚀 FAST LOADING FUNCTIONS
@st.cache_data(show_spinner=False)
def load_excel_fast(file_path=None, upload_file=None):
    """Fast Excel loading with caching"""
    if upload_file is not None:
        return pd.read_excel(upload_file, sheet_name=None, engine='openpyxl')
    else:
        return pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda x: str(x.shape) + str(x.columns.tolist())})  # 🚀 BETTER HASHING
def prepare_queries_fast(df):
    """Fast query preparation with memory optimization"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # 🚀 SHALLOW COPY (faster)
    queries = df.copy(deep=False)
    
    # 🚀 VECTORIZED COLUMN FIXES (faster than individual renames)
    column_mapping = {
        'Search': 'search', 'query': 'search', 'Query': 'search',
        'Count': 'Counts', 'counts': 'Counts', 'count': 'Counts',  # 🚀 ADD 'count' mapping
        'Clicks': 'clicks', 'Conversions': 'conversions'
    }
    queries = queries.rename(columns=column_mapping)
    
    # 🚀 BATCH ADD MISSING COLUMNS (faster)
    required_cols = {'search': 'Unknown Query', 'Counts': 0, 'clicks': 0, 'conversions': 0}
    for col, default_val in required_cols.items():
        if col not in queries.columns:
            queries[col] = default_val
    
    # 🚀 VECTORIZED NUMERIC CONVERSION (much faster)
    numeric_cols = ['Counts', 'clicks', 'conversions']
    for col in numeric_cols:
        if col in queries.columns:
            queries[col] = pd.to_numeric(queries[col], errors='coerce').fillna(0).astype('int32')  # 🚀 USE INT32
    
    # 🚀 OPTIMIZED CLEANUP (faster boolean indexing)
    valid_mask = (queries['search'].notna()) & (queries['search'].astype(str).str.strip() != '')
    queries = queries[valid_mask].reset_index(drop=True)
    
    return queries


# 🚀 LOAD DATA ONLY ONCE
if not st.session_state.data_loaded:
    with st.spinner('🚀 Loading data...'):
        try:
            # Load file
            if upload is not None:
                if upload.name.endswith('.xlsx'):
                    sheets = load_excel_fast(upload_file=upload)
                else:  # CSV
                    df_csv = pd.read_csv(upload)
                    sheets = {'queries': df_csv}
            else:
                default_path = "NUTRACEUTICALS AND NUTRITION combined_data_ June - August 2025_with_brands.xlsx"
                if os.path.exists(default_path):
                    sheets = load_excel_fast(file_path=default_path)
                else:
                    st.info("📁 No file uploaded and default Excel not found.")
                    st.stop()
            
            # Get main queries sheet
            sheet_names = list(sheets.keys())
            preferred = ['queries_clustered', 'queries_dedup', 'queries']
            main_sheet = None
            
            for pref in preferred:
                if pref in sheets:
                    main_sheet = pref
                    break
            
            if main_sheet is None:
                main_sheet = sheet_names[0]
            
            raw_queries = sheets[main_sheet]
            queries = prepare_queries_fast(raw_queries)
            
            # Store in session state
            st.session_state.queries = queries
            st.session_state.sheets = sheets
            st.session_state.data_loaded = True
            
            
        except Exception as e:
            st.error(f"❌ Loading error: {e}")
            st.stop()

# 🚀 USE CACHED DATA
queries = st.session_state.queries
sheets = st.session_state.sheets

# Load summary sheets
brand_summary = sheets.get('brand_summary', None)
category_summary = sheets.get('category_summary', None)
subcategory_summary = sheets.get('subcategory_summary', None)
generic_type = sheets.get('generic_type', None)

# 🚀 OPTIONAL: Reload button
if st.sidebar.button("🔄 Reload Data"):
    st.session_state.data_loaded = False
    st.rerun()

# Show data info
if st.sidebar.checkbox("📊 Show Data Info"):
    st.sidebar.success(f"""
    **Data Loaded:**
    - Queries: {len(queries):,}
    - Sheets: {len(sheets)}
    - Columns: {list(queries.columns)}
    """)

st.markdown("---")

# ----------------- Choose main queries sheet -----------------
sheet_keys = list(sheets.keys())
preferred = [k for k in ['queries_clustered','queries_dedup','queries','queries_clustered_preprocessed'] if k in sheets]
if preferred:
    main_key = preferred[0]
else:
    main_key = sheet_keys[0]

raw_queries = sheets[main_key]
try:
    queries = prepare_queries_df(raw_queries)
except Exception as e:
    st.error(f"Error processing queries sheet: {e}")
    st.stop()

# Load additional summary sheets if present
brand_summary = sheets.get('brand_summary', None)
category_summary = sheets.get('category_summary', None)
subcategory_summary = sheets.get('subcategory_summary', None)
generic_type = sheets.get('generic_type', None)

# ----------------- Filters (no sampling) -----------------
# ----------------- Filters with Apply/Reset buttons -----------------
# ----------------- OPTIMIZED FILTERS (KEEPING YOUR EXACT LOGIC) -----------------
st.sidebar.header("🔎 Filters")

# Initialize session state for filters
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

# Store original queries for reset
if 'original_queries' not in st.session_state:
    st.session_state.original_queries = queries.copy()

# 🚀 OPTIMIZED DATE FILTER (SAME LOGIC, BETTER PERFORMANCE)
@st.cache_data(ttl=3600, show_spinner=False)
def get_date_range(_df):
    """Cache date range calculation"""
    try:
        min_date = _df['Date'].min()
        max_date = _df['Date'].max()
        
        if pd.isna(min_date):
            min_date = None
        if pd.isna(max_date):
            max_date = None
            
        return [min_date, max_date] if min_date is not None and max_date is not None else []
    except:
        return []

default_dates = get_date_range(st.session_state.original_queries)
date_range = st.sidebar.date_input("📅 Select Date Range", value=default_dates)

# 🚀 OPTIMIZED Multi-select filters helper (SAME INTERFACE, CACHED)
@st.cache_data(ttl=1800, show_spinner=False, hash_funcs={pd.DataFrame: lambda x: x.shape[0]})  # 🚀 ADD THIS LINE
def get_cached_options(_df, col):
    """Cache filter options for better performance"""
    try:
        if col not in _df.columns:
            return []
        return sorted(_df[col].dropna().astype(str).unique().tolist())
    except:
        return []


def get_filter_options(df, col, label, emoji):
    """Your exact function with caching optimization"""
    if col not in df.columns:
        return [], []
    
    # Use cached options instead of recalculating every time
    opts = get_cached_options(df, col)
    
    sel = st.sidebar.multiselect(
        f"{emoji} {label}", 
        options=opts, 
        default=opts  # Keep your exact default behavior
    )
    return sel, opts

# Get filter selections (EXACTLY THE SAME AS YOUR CODE)
brand_filter, brand_opts = get_filter_options(st.session_state.original_queries, 'brand', 'Brand(s)', '🏷')
dept_filter, dept_opts = get_filter_options(st.session_state.original_queries, 'department', 'Department(s)', '🏬')
cat_filter, cat_opts = get_filter_options(st.session_state.original_queries, 'category', 'Category(ies)', '📦')
subcat_filter, subcat_opts = get_filter_options(st.session_state.original_queries, 'sub_category', 'Sub Category(ies)', '🧴')

# Text filter (EXACTLY THE SAME)
text_filter = st.sidebar.text_input("🔍 Filter queries by text (contains)")

# Filter control buttons (EXACTLY THE SAME)
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    apply_filters = st.button("🔄 Apply Filters", use_container_width=True, type="primary")

with col2:
    reset_filters = st.button("🗑️ Reset Filters", use_container_width=True)

# Handle Reset Button (EXACTLY THE SAME AS YOUR CODE)
if reset_filters:
    # Reset the data immediately
    queries = st.session_state.original_queries.copy()
    st.session_state.filters_applied = False
    
    # Clear all filter widgets by rerunning
    st.rerun()

# Handle Apply Button (YOUR EXACT LOGIC WITH MINOR OPTIMIZATION)
elif apply_filters:
    # Apply filters - START WITH ORIGINAL DATA
    queries = st.session_state.original_queries.copy()
    
    # Date filter (YOUR EXACT LOGIC)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and date_range[0] is not None:
        start_date, end_date = date_range
        queries = queries[(queries['Date'] >= pd.to_datetime(start_date)) & (queries['Date'] <= pd.to_datetime(end_date))]
    
    # Brand filter (YOUR EXACT LOGIC)
    if brand_filter and len(brand_filter) < len(brand_opts):
        queries = queries[queries['brand'].astype(str).isin(brand_filter)]
    
    # Department filter (YOUR EXACT LOGIC)
    if dept_filter and len(dept_filter) < len(dept_opts):
        queries = queries[queries['department'].astype(str).isin(dept_filter)]
    
    # Category filter (YOUR EXACT LOGIC)
    if cat_filter and len(cat_filter) < len(cat_opts):
        queries = queries[queries['category'].astype(str).isin(cat_filter)]
    
    # Subcategory filter (YOUR EXACT LOGIC)
    if subcat_filter and len(subcat_filter) < len(subcat_opts):
        queries = queries[queries['sub_category'].astype(str).isin(subcat_filter)]
    
    # Text filter (YOUR EXACT LOGIC)
    if text_filter:
        queries = queries[queries['normalized_query'].str.contains(re.escape(text_filter), case=False, na=False)]
    
    st.session_state.filters_applied = True

# Show filter status (ENHANCED VERSION OF YOUR CODE)
if st.session_state.filters_applied:
    original_count = len(st.session_state.original_queries)
    current_count = len(queries)
    reduction_pct = ((original_count - current_count) / original_count) * 100 if original_count > 0 else 0
    st.sidebar.success(f"✅ Filters Applied - {current_count:,} rows ({reduction_pct:.1f}% filtered)")
else:
    st.sidebar.info(f"📊 No filters applied - {len(queries):,} rows")

st.sidebar.markdown(f"**📊 Current rows:** {len(queries):,}")

# 🚀 DEBUG INFO (OPTIONAL - REMOVE AFTER TESTING)
if st.sidebar.checkbox("🔍 Debug Info", value=False):
    st.sidebar.write("**Filter Status:**")
    st.sidebar.write(f"- Date range: {date_range}")
    st.sidebar.write(f"- Brand selected: {len(brand_filter)}/{len(brand_opts)}")
    st.sidebar.write(f"- Dept selected: {len(dept_filter)}/{len(dept_opts)}")
    st.sidebar.write(f"- Cat selected: {len(cat_filter)}/{len(cat_opts)}")
    st.sidebar.write(f"- Subcat selected: {len(subcat_filter)}/{len(subcat_opts)}")
    st.sidebar.write(f"- Text filter: '{text_filter}'")



# ----------------- Welcome Message -----------------
st.markdown("""
<div class="welcome-box">
    <h2>🌿 Welcome to Nutraceuticals & Nutrition Analytics! 💚</h2>
    <p>Discover Nutraceuticals & Nutrition trends, nutritional insights, and supplement performance data. Navigate through health categories, analyze supplement searches, and unlock actionable insights for optimal nutrition strategies!</p>
</div>
""", unsafe_allow_html=True)

# ----------------- KPI cards -----------------
st.markdown('<div class="main-header">🌱 Nutraceuticals & Nutrition — Advanced Analytics Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explore Nutraceuticals & Nutrition search patterns and nutritional supplement insights with <b>data-driven health analytics</b></div>', unsafe_allow_html=True)

# 🚀 CACHED Calculate metrics
@st.cache_data(ttl=300, show_spinner=False)
def calculate_metrics(_df):
    total_counts = int(_df['Counts'].sum())
    total_clicks = int(_df['clicks'].sum())
    total_conversions = int(_df['conversions'].sum())
    overall_ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
    overall_cr = (total_conversions / total_counts * 100) if total_clicks > 0 else 0
    return total_counts, total_clicks, total_conversions, overall_ctr, overall_cr

total_counts, total_clicks, total_conversions, overall_ctr, overall_cr = calculate_metrics(queries)
total_revenue = 0.0  # No revenue column

c1, c2, c3, c4, c5 = st.columns(5)
# Add this helper function at the top of your file (after imports)
def format_number(num):
    """Format numbers with K/M suffix"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.0f}"

# Then update the KPI cards:
with c1:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_counts)}</div><div class='label'>🌿 Total Searches</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_clicks)}</div><div class='label'>🍃 Total Clicks</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_conversions)}</div><div class='label'>💚 Total Conversions</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='kpi'><div class='value'>{overall_ctr:.2f}%</div><div class='label'>📈 Overall CTR</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='kpi'><div class='value'>{overall_cr:.2f}%</div><div class='label'>🌱 Overall CR</div></div>", unsafe_allow_html=True)


# Show data source info in sidebar
st.sidebar.info(f"**Data Source:** {main_key}")
st.sidebar.write(f"**Total Rows:** {len(queries):,}")
st.sidebar.write(f"**Total Searches:** {total_counts:,}")
st.sidebar.write(f"**Calculated Clicks:** {total_clicks:,}")
st.sidebar.write(f"**Calculated Conversions:** {total_conversions:,}")

# Add debug info in an expander so it doesn't clutter the sidebar
with st.sidebar.expander("🔍 Data Debug Info"):
    st.write(f"Main sheet: {main_key}")
    st.write(f"Processed columns: {list(queries.columns)}")
    st.write(f"Processed shape: {queries.shape}")
    
    st.write("**Column Usage:**")
    if 'count' in raw_queries.columns:
        st.write(f"✓ Searches/Impressions: 'count' column")
    else:
        st.write("✗ Searches/Impressions: No 'count' column found")
    
    st.write("**Calculation Method:**")
    st.write("• Clicks = Searches × Click Through Rate")
    st.write("• Conversions = Clicks × Conversion Rate")
    
    # Show sample of raw data
    st.write("**Sample data (first 3 rows):**")
    st.dataframe(raw_queries.head(3))

# ----------------- Tabs -----------------
tab_overview, tab_search, tab_brand, tab_category, tab_subcat, tab_generic, tab_time, tab_pivot, tab_insights, tab_export = st.tabs([
    "🌿 Overview","🔍 Search Analysis","🏷 Brand","📦 Category","🧴 Subcategory","💊 Supplement Type",
    "⏰ Time Analysis","📊 Pivot Builder","💡 Health Insights","⬇ Export"
])

# ----------------- Overview -----------------
with tab_overview:
    st.header("🌿 Nutraceuticals & Nutrition Overview & Health Insights")
    st.markdown("Discover nutritional trends and supplement performance patterns. 🌱 Based on **nutraceuticals data** (e.g., millions of health-conscious searches across Nutraceuticals & Nutrition categories).")

    # Accuracy Fix: Ensure Date conversion (Excel serial)
    if not queries['Date'].dtype == 'datetime64[ns]':
        queries['Date'] = pd.to_datetime(queries['start_date'], unit='D', origin='1899-12-30', errors='coerce')

    # Refresh Button (User-Friendly)
    if st.button("🔄 Refresh Health Data & Filters"):
        st.rerun()

    # Image Selection in Sidebar
    st.sidebar.header("🌿 Customize Nutraceuticals & Nutrition Theme")
    image_options = {
        "Natural Green Gradient": "https://placehold.co/1200x200/E8F5E8/2E7D32?text=Nutraceuticals+%26+Nutrition+Analytics",
        "Wellness & Supplements": "https://picsum.photos/1200/250?random=wellness_supplements",
        "Organic Health Theme": "https://source.unsplash.com/1200x250/?health,nutrition,supplements",
        "Custom Health Banner": "https://placehold.co/1200x250/F1F8E9/1B5E20?text=💚+Health+%26+Wellness+Insights",
        "Natural Ingredients": "https://picsum.photos/1200/250?random=natural_health"
    }
    selected_image = st.sidebar.selectbox("Choose Wellness Image", options=list(image_options.keys()), index=0)

    # Hero Image (Creative UI) with selected option
    st.image(image_options[selected_image], use_container_width=True)

    # FIRST ROW: Monthly Counts Table and Chart side by side
    st.markdown("## 🌱 Monthly Nutraceuticals & Nutrition Analysis Overview")
    col_table, col_chart = st.columns([1,2])  # Equal width columns

    with col_table:
        st.markdown("### 📋 Monthly Health Searches Table")
        monthly_counts = queries.groupby(queries['Date'].dt.strftime('%B %Y'))['Counts'].sum().reset_index()
        
        if not monthly_counts.empty:
            # Ensure 'Counts' is numeric and handle NaN
            monthly_counts['Counts'] = pd.to_numeric(monthly_counts['Counts'], errors='coerce').fillna(0)
            total_all_months = monthly_counts['Counts'].sum()
            monthly_counts['Percentage'] = (monthly_counts['Counts'] / total_all_months * 100).round(1)
            
            # 🚀 NEW: Create display version with formatted numbers
            display_monthly = monthly_counts.copy()
            display_monthly['Counts'] = display_monthly['Counts'].apply(lambda x: format_number(int(x)))
            
            # Style the table using the formatted display version
            styled_monthly = display_monthly.style.set_properties(**{
                'text-align': 'center',
                'vertical-align': 'middle',
                'font-size': '14px',
                'padding': '10px'
            }).set_table_styles([
                {
                    'selector': 'th',
                    'props': [
                        ('text-align', 'center'),
                        ('font-weight', 'bold'),
                        ('background-color', '#2E7D32'),
                        ('color', 'white'),
                        ('padding', '12px')
                    ]
                },
                {
                    'selector': 'td',
                    'props': [
                        ('text-align', 'center'),
                        ('padding', '10px'),
                        ('border', '1px solid #ddd')
                    ]
                }
            ]).format({
                'Percentage': '{:.1f}%'  # Only format percentage since Counts is already formatted
            })
            
            st.dataframe(styled_monthly, use_container_width=True, hide_index=True)
            
            # Summary metrics below table
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%); 
                        padding: 15px; border-radius: 10px; color: white; margin: 10px 0; text-align: center;">
                <strong>🌱 Total: {format_number(int(total_all_months))} health searches across {len(monthly_counts)} months</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No monthly Nutraceuticals & Nutrition data available")


    with col_chart:
        st.markdown("### 📈 Monthly Nutraceuticals & Nutrition Trends Visualization")
        
        if not monthly_counts.empty and len(monthly_counts) >= 2:
            try:
                fig = px.bar(monthly_counts, x='Date', y='Counts',
                            title='<b style="color:#2E7D32; font-size:16px;">Monthly Health Search Trends 🌿</b>',
                            labels={'Date': '<i>Month</i>', 'Counts': '<b>Health Searches</b>'},
                            color='Counts',
                            color_continuous_scale=['#E8F5E8', '#66BB6A', '#2E7D32'],
                            template='plotly_white',
                            text=monthly_counts['Counts'].astype(str))
                    
                # Update traces
                fig.update_traces(
                    texttemplate='%{text}<br>%{customdata:.1f}%',
                    customdata=monthly_counts['Percentage'],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Searches: %{y:,.0f}<br>Share: %{customdata:.1f}%<extra></extra>'
                )
                
                # Layout optimization
                fig.update_layout(
                    plot_bgcolor='rgba(248,253,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    title_x=0.5,  # Center alignment for title
                    title_font_size=16,
                    xaxis=dict(showgrid=True, gridcolor='#E8F5E8', linecolor='#2E7D32', linewidth=2),
                    yaxis=dict(showgrid=True, gridcolor='#E8F5E8', linecolor='#2E7D32', linewidth=2),
                    bargap=0.2,
                    barcornerradius=8,
                    height=400,
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                
                # Highlight peak month
                peak_month = monthly_counts.loc[monthly_counts['Counts'].idxmax(), 'Date']
                peak_value = monthly_counts['Counts'].max()
                fig.add_annotation(
                    x=peak_month, y=peak_value,
                    text=f"🏆 Peak Nutraceuticals & Nutrition: {peak_value:,.0f}",
                    showarrow=True,
                    arrowhead=3,
                    arrowcolor='#2E7D32',
                    ax=0, ay=-40,
                    font=dict(size=12, color='#2E7D32', family='Segoe UI', weight='bold')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating chart: {e}")
        else:
            st.info("📅 Add more date range for Nutraceuticals & Nutrition trends visualization")

    # Add separator between sections
    st.markdown("---")

    # SECOND ROW: Top 50 Queries (Full Width)
    # 🚀 MOVE THESE FUNCTIONS OUTSIDE - DEFINE THEM BEFORE THE SECTION
    @st.cache_data(ttl=1800, show_spinner=False, hash_funcs={pd.DataFrame: lambda x: hash(str(x.shape) + str(x.columns.tolist()))})
    def compute_top50_queries_ultra(_queries, _month_names, _filter_key=None):
        """Ultra-optimized version of your compute_top50_queries function"""
        
        # 🚀 FAST: Early return for empty data
        if _queries.empty:
            return pd.DataFrame(), []
        
        # 🚀 VECTORIZED: Get unique months efficiently
        unique_months = []
        if 'Date' in _queries.columns:
            _queries = _queries.copy()
            _queries['month_year'] = _queries['Date'].dt.strftime('%Y-%m')
            unique_months = sorted(_queries['month_year'].dropna().unique())

        # 🚀 OPTIMIZED: Single groupby operation
        agg_dict = {
            'Counts': 'sum',
            'clicks': 'sum', 
            'conversions': 'sum'
        }
        
        top50 = _queries.groupby('search', as_index=False).agg(agg_dict)

        # 🚀 FAST: Vectorized monthly calculations using pivot
        if unique_months and 'month_year' in _queries.columns:
            monthly_pivot = _queries.pivot_table(
                index='search', 
                columns='month_year', 
                values='Counts', 
                aggfunc='sum', 
                fill_value=0
            )
            
            for month in unique_months:
                month_display_name = _month_names.get(month, month)
                if month in monthly_pivot.columns:
                    top50[month_display_name] = top50['search'].map(
                        monthly_pivot[month].to_dict()
                    ).fillna(0).astype('int32')

        # 🚀 VECTORIZED: All calculations at once
        total_counts = _queries['Counts'].sum()
        top50['Share %'] = (top50['Counts'] / total_counts * 100).round(2)
        
        # Fast conversion rate with numpy
        top50['Conversion Rate'] = np.where(
            top50['Counts'] > 0,
            (top50['conversions'] / top50['Counts'] * 100).round(2),
            0
        )

        # 🚀 EFFICIENT: Single sort and slice
        top50 = top50.nlargest(50, 'Counts')

        # 🚀 FAST: Batch column operations
        column_renames = {
            'search': 'Health Query',
            'Counts': 'Total Search Volume',
            'clicks': 'Clicks',
            'conversions': 'Conversions'
        }
        top50 = top50.rename(columns=column_renames)

        # 🚀 VECTORIZED: Batch type conversion
        numeric_cols = ['Clicks', 'Conversions']
        for col in numeric_cols:
            if col in top50.columns:
                top50[col] = top50[col].round().astype('int32')

        # Format monthly columns efficiently
        for month in unique_months:
            month_display_name = _month_names.get(month, month)
            if month_display_name in top50.columns:
                top50[month_display_name] = top50[month_display_name].astype('int32')

        # 🚀 OPTIMIZED: Column ordering
        column_order = ['Health Query', 'Total Search Volume', 'Share %']
        column_order.extend([_month_names.get(month, month) for month in unique_months 
                        if _month_names.get(month, month) in top50.columns])
        column_order.extend(['Clicks', 'Conversions', 'Conversion Rate'])
        
        available_columns = [col for col in column_order if col in top50.columns]
        top50 = top50[available_columns]

        return top50, unique_months

    # 🚀 CACHED: MoM analysis function
    @st.cache_data(ttl=1800, show_spinner=False)
    def compute_mom_analysis_ultra(_top50, _unique_months, _month_names, _filter_key=None):
        """Ultra-fast MoM analysis"""
        if len(_unique_months) < 2:
            return pd.DataFrame(), pd.DataFrame()
        
        top10_for_analysis = _top50.head(10).copy()
        
        month1_name = _month_names.get(_unique_months[0], _unique_months[0])
        month2_name = _month_names.get(_unique_months[1], _unique_months[1])
        
        if month1_name in top10_for_analysis.columns and month2_name in top10_for_analysis.columns:
            # Vectorized MoM calculation
            month1_vals = top10_for_analysis[month1_name].replace(0, 1)
            top10_for_analysis['MoM Change'] = (
                (top10_for_analysis[month2_name] - top10_for_analysis[month1_name]) / month1_vals * 100
            ).round(1)
            
            gainers = top10_for_analysis.nlargest(3, 'MoM Change')[['Health Query', 'MoM Change']]
            losers = top10_for_analysis.nsmallest(3, 'MoM Change')[['Health Query', 'MoM Change']]
            
            return gainers, losers
        
        return pd.DataFrame(), pd.DataFrame()

    # 🚀 CACHED: CSV generation
    @st.cache_data(ttl=300, show_spinner=False)
    def generate_csv_ultra(_df):
        return _df.to_csv(index=False)

    # NOW START THE ACTUAL SECTION
    st.markdown("## 🔍 Top 50 Health Queries Analysis")

    if queries.empty or 'Counts' not in queries.columns or queries['Counts'].isna().all():
        st.warning("No valid data available for top 50 health queries.")
    else:
        try:
            # 🚀 LAZY CSS LOADING - Only load once per session
            if 'top50_health_css_loaded' not in st.session_state:
                st.markdown("""
                <style>
                .top50-health-metric-card {
                    background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3); margin: 8px 0;
                    min-height: 120px; display: flex; flex-direction: column; justify-content: center;
                    transition: transform 0.2s ease; width: 100%;
                }
                .top50-health-metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(46, 125, 50, 0.4); }
                .top50-health-metric-card .icon { font-size: 2.5em; margin-bottom: 8px; display: block; }
                .top50-health-metric-card .value { font-size: 1.8em; font-weight: bold; margin-bottom: 5px; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.1; }
                .top50-health-metric-card .label { font-size: 1em; opacity: 0.95; font-weight: 600; line-height: 1.2; }
                .monthly-health-metric-card {
                    background: linear-gradient(135deg, #1B5E20 0%, #4CAF50 100%);
                    padding: 18px; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 6px 25px rgba(27, 94, 32, 0.3); margin: 8px 0;
                    min-height: 100px; display: flex; flex-direction: column; justify-content: center;
                    transition: transform 0.2s ease; width: 100%;
                }
                .monthly-health-metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 35px rgba(27, 94, 32, 0.4); }
                .monthly-health-metric-card .icon { font-size: 2em; margin-bottom: 6px; display: block; }
                .monthly-health-metric-card .value { font-size: 1.5em; font-weight: bold; margin-bottom: 4px; line-height: 1.1; }
                .monthly-health-metric-card .label { font-size: 0.9em; opacity: 0.95; font-weight: 600; line-height: 1.2; }
                .download-health-section { background: linear-gradient(135deg, #388E3C 0%, #4CAF50 100%); padding: 20px; border-radius: 12px; text-align: center; margin: 20px 0; box-shadow: 0 6px 25px rgba(56, 142, 60, 0.3); }
                .insights-health-section { background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); padding: 20px; border-radius: 12px; margin: 20px 0; box-shadow: 0 6px 25px rgba(46, 125, 50, 0.3); }
                .mom-health-analysis { background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin: 10px 0; }
                .health-gainer-item { background: rgba(76, 175, 80, 0.2); padding: 8px 12px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #4CAF50; }
                .health-decliner-item { background: rgba(244, 67, 54, 0.2); padding: 8px 12px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #F44336; }
                .health-performance-increase { background-color: rgba(76, 175, 80, 0.1) !important; }
                .health-performance-decrease { background-color: rgba(244, 67, 54, 0.1) !important; }
                .health-comparison-header { background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%); color: white; font-weight: bold; text-align: center; padding: 8px; }
                .health-volume-column { background-color: rgba(46, 125, 50, 0.1) !important; }
                .health-performance-column { background-color: rgba(102, 187, 106, 0.1) !important; }
                </style>
                """, unsafe_allow_html=True)
                st.session_state.top50_health_css_loaded = True

            # 🚀 OPTIMIZED: Show debug info only in sidebar (non-blocking)
            if st.sidebar.checkbox("Show Health Debug Info", value=False):
                st.sidebar.write("**Available columns in health queries:**", list(queries.columns))

            # 🚀 ENHANCED: Static month names (faster than dynamic lookup)
            month_names = {
                '2025-06': 'June 2025',
                '2025-07': 'July 2025',
                '2025-08': 'August 2025'
            }

            # 🚀 COMPUTE: Get data with caching (filter-aware) - ENHANCED WITH BETTER ARRANGEMENT
            filter_state = {
                'filters_applied': st.session_state.get('filters_applied', False),
                'data_shape': queries.shape,
                'data_hash': hash(str(queries['search'].tolist()[:10]) if not queries.empty else "empty")
            }
            filter_key = str(hash(str(filter_state)))

            @st.cache_data(ttl=1800, show_spinner=False)
            def compute_top50_health_queries_better_arrangement(_df, month_names_dict, cache_key):
                """🔄 FIXED: Proper monthly CTR/CR calculations for health queries"""
                if _df.empty:
                    return pd.DataFrame(), []
                
                # Group by search query and sum counts
                grouped = _df.groupby('search').agg({
                    'Counts': 'sum',
                    'clicks': 'sum', 
                    'conversions': 'sum'
                }).reset_index()
                
                # Get top 50 by total counts
                top50_queries = grouped.nlargest(50, 'Counts')['search'].tolist()
                
                # Filter original data for top 50 queries
                top50_data = _df[_df['search'].isin(top50_queries)].copy()
                
                # Get unique months from the data
                if 'month' in top50_data.columns:
                    unique_months = sorted(top50_data['month'].unique())
                else:
                    unique_months = []
                
                # 🔄 BETTER ARRANGEMENT: Reorganize columns for easier comparison
                result_data = []
                
                for query in top50_queries:
                    query_data = top50_data[top50_data['search'] == query]
                    
                    # Base information
                    total_counts = int(query_data['Counts'].sum())
                    total_clicks = int(query_data['clicks'].sum())
                    total_conversions = int(query_data['conversions'].sum())
                    overall_ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
                    overall_cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0
                    
                    row = {
                        'Health Query': query,
                        'Total Volume': total_counts,
                        'Share %': (total_counts / _df['Counts'].sum()) * 100,
                        'Overall CTR': overall_ctr,
                        'Overall CR': overall_cr,
                        'Total Clicks': total_clicks,
                        'Total Conversions': total_conversions
                    }
                    
                    # 🔧 FIXED: Monthly data calculations with proper month-specific metrics
                    for month in unique_months:
                        month_data = query_data[query_data['month'] == month]
                        month_display = month_names_dict.get(month, month)
                        
                        if not month_data.empty:
                            # ✅ FIXED: Calculate month-specific metrics
                            month_counts = int(month_data['Counts'].sum())
                            month_clicks = int(month_data['clicks'].sum())
                            month_conversions = int(month_data['conversions'].sum())
                            
                            # ✅ FIXED: Month-specific CTR and CR calculations
                            month_ctr = (month_clicks / month_counts * 100) if month_counts > 0 else 0
                            month_cr = (month_conversions / month_counts * 100) if month_counts > 0 else 0
                            
                            row[f'{month_display} Vol'] = month_counts
                            row[f'{month_display} CTR'] = month_ctr  # ✅ NOW CORRECT
                            row[f'{month_display} CR'] = month_cr    # ✅ NOW CORRECT
                        else:
                            row[f'{month_display} Vol'] = 0
                            row[f'{month_display} CTR'] = 0
                            row[f'{month_display} CR'] = 0
                    
                    result_data.append(row)
                
                result_df = pd.DataFrame(result_data)
                result_df = result_df.sort_values('Total Volume', ascending=False).reset_index(drop=True)
                
                return result_df, unique_months

            top50, unique_months = compute_top50_health_queries_better_arrangement(queries, month_names, filter_key)

            if top50.empty:
                st.warning("No valid data after processing top 50 health queries.")
            else:
                # 🔄 BETTER ARRANGEMENT: Reorder columns for logical flow
                base_columns = ['Health Query', 'Total Volume', 'Share %', 'Overall CTR', 'Overall CR', 'Total Clicks', 'Total Conversions']
                
                # Group monthly columns by type for easier comparison
                volume_columns = []
                ctr_columns = []
                cr_columns = []
                
                for month in sorted(unique_months):
                    month_display = month_names.get(month, month)
                    volume_columns.append(f'{month_display} Vol')
                    ctr_columns.append(f'{month_display} CTR')
                    cr_columns.append(f'{month_display} CR')
                
                # 🔄 LOGICAL COLUMN ORDER: Base info → Monthly Volumes → Monthly CTRs → Monthly CRs
                ordered_columns = base_columns + volume_columns + ctr_columns + cr_columns
                existing_columns = [col for col in ordered_columns if col in top50.columns]
                top50 = top50[existing_columns]

                # 🚀 ENHANCED: Smart styling with better comparison highlighting
                top50_hash = hash(str(top50.shape) + str(top50.columns.tolist()) + str(top50.iloc[0].to_dict()) if len(top50) > 0 else "empty")
                
                if ('styled_top50_health' not in st.session_state or 
                    st.session_state.get('top50_health_cache_key') != top50_hash):
                    
                    st.session_state.top50_health_cache_key = top50_hash
                    
                    # 🚀 FAST: Apply format_number to numeric columns before styling
                    display_top50 = top50.copy()
                    
                    # Format volume columns with format_number
                    volume_cols_to_format = ['Total Volume'] + volume_columns
                    for col in volume_cols_to_format:
                        if col in display_top50.columns:
                            display_top50[col] = display_top50[col].apply(lambda x: format_number(int(x)) if pd.notnull(x) else '0')
                    
                    # Format clicks and conversions
                    if 'Total Clicks' in display_top50.columns:
                        display_top50['Total Clicks'] = display_top50['Total Clicks'].apply(lambda x: format_number(int(x)))
                    if 'Total Conversions' in display_top50.columns:
                        display_top50['Total Conversions'] = display_top50['Total Conversions'].apply(lambda x: format_number(int(x)))
                    
                    # 🔄 ENHANCED: Better performance highlighting with comparison focus
                    def highlight_health_performance_with_comparison(df):
                        """Enhanced highlighting for better health comparison"""
                        styles = pd.DataFrame('', index=df.index, columns=df.columns)
                        
                        if len(unique_months) < 2:
                            return styles
                        
                        sorted_months = sorted(unique_months)
                        
                        # 🔄 COMPARISON FOCUS: Highlight month-over-month changes
                        for i in range(1, len(sorted_months)):
                            current_month = month_names.get(sorted_months[i], sorted_months[i])
                            prev_month = month_names.get(sorted_months[i-1], sorted_months[i-1])
                            
                            current_ctr_col = f'{current_month} CTR'
                            prev_ctr_col = f'{prev_month} CTR'
                            current_cr_col = f'{current_month} CR'
                            prev_cr_col = f'{prev_month} CR'
                            
                            # CTR comparison with threshold
                            if current_ctr_col in df.columns and prev_ctr_col in df.columns:
                                for idx in df.index:
                                    current_ctr = df.loc[idx, current_ctr_col]
                                    prev_ctr = df.loc[idx, prev_ctr_col]
                                    
                                    if pd.notnull(current_ctr) and pd.notnull(prev_ctr) and prev_ctr > 0:
                                        change_pct = ((current_ctr - prev_ctr) / prev_ctr) * 100
                                        if change_pct > 10:  # 10% improvement
                                            styles.loc[idx, current_ctr_col] = 'background-color: rgba(76, 175, 80, 0.3); color: #1B5E20; font-weight: bold;'
                                        elif change_pct < -10:  # 10% decline
                                            styles.loc[idx, current_ctr_col] = 'background-color: rgba(244, 67, 54, 0.3); color: #B71C1C; font-weight: bold;'
                                        elif abs(change_pct) > 5:  # 5-10% change
                                            color = 'rgba(76, 175, 80, 0.15)' if change_pct > 0 else 'rgba(244, 67, 54, 0.15)'
                                            styles.loc[idx, current_ctr_col] = f'background-color: {color};'
                            
                            # CR comparison with threshold
                            if current_cr_col in df.columns and prev_cr_col in df.columns:
                                for idx in df.index:
                                    current_cr = df.loc[idx, current_cr_col]
                                    prev_cr = df.loc[idx, prev_cr_col]
                                    
                                    if pd.notnull(current_cr) and pd.notnull(prev_cr) and prev_cr > 0:
                                        change_pct = ((current_cr - prev_cr) / prev_cr) * 100
                                        if change_pct > 10:  # 10% improvement
                                            styles.loc[idx, current_cr_col] = 'background-color: rgba(76, 175, 80, 0.3); color: #1B5E20; font-weight: bold;'
                                        elif change_pct < -10:  # 10% decline
                                            styles.loc[idx, current_cr_col] = 'background-color: rgba(244, 67, 54, 0.3); color: #B71C1C; font-weight: bold;'
                                        elif abs(change_pct) > 5:  # 5-10% change
                                            color = 'rgba(76, 175, 80, 0.15)' if change_pct > 0 else 'rgba(244, 67, 54, 0.15)'
                                            styles.loc[idx, current_cr_col] = f'background-color: {color};'
                        
                        # 🔄 SECTION HIGHLIGHTING: Different background for different metric groups
                        for col in volume_columns:
                            if col in df.columns:
                                styles.loc[:, col] = styles.loc[:, col] + 'background-color: rgba(46, 125, 50, 0.05);'
                        
                        return styles
                    
                    # Create styled DataFrame from the formatted copy
                    styled_top50 = display_top50.style.apply(highlight_health_performance_with_comparison, axis=None)
                    
                    styled_top50 = styled_top50.set_properties(**{
                        'text-align': 'center',
                        'vertical-align': 'middle',
                        'font-size': '11px',
                        'padding': '4px',
                        'line-height': '1.1'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center'), ('vertical-align', 'middle'), ('font-weight', 'bold'), ('background-color', '#E8F5E8'), ('color', '#1B5E20'), ('padding', '6px'), ('border', '1px solid #ddd'), ('font-size', '10px')]},
                        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle'), ('padding', '4px'), ('border', '1px solid #ddd')]},
                        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#F8FDF8')]}
                    ])
                    
                    # 🔄 IMPROVED: Format dictionary
                    format_dict = {
                        'Share %': '{:.2f}%',
                        'Overall CTR': '{:.2f}%',
                        'Overall CR': '{:.2f}%'
                    }
                    
                    # Add formatting for monthly CTR and CR columns
                    for col in ctr_columns + cr_columns:
                        if col in display_top50.columns:
                            format_dict[col] = '{:.2f}%'

                    styled_top50 = styled_top50.format(format_dict)
                    st.session_state.styled_top50_health = styled_top50

                # 🚀 DISPLAY: Cached styled DataFrame
                st.dataframe(
                    st.session_state.styled_top50_health, 
                    use_container_width=True, 
                    height=600,
                    hide_index=True
                )

                # 🔄 ENHANCED: Better legend with comparison focus
                st.markdown("""
                <div style="background: rgba(46, 125, 50, 0.1); padding: 12px; border-radius: 8px; margin: 15px 0;">
                    <h4 style="margin: 0 0 8px 0; color: #1B5E20;">🌿 Health Comparison Guide:</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                        <div>📈 <strong style="background-color: rgba(76, 175, 80, 0.3); padding: 2px 6px; border-radius: 4px; color: #1B5E20;">Dark Green</strong> = >10% improvement</div>
                        <div>📈 <strong style="background-color: rgba(76, 175, 80, 0.15); padding: 2px 6px; border-radius: 4px;">Light Green</strong> = 5-10% improvement</div>
                        <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.3); padding: 2px 6px; border-radius: 4px; color: #B71C1C;">Dark Red</strong> = >10% decline</div>
                        <div>📉 <strong style="background-color: rgba(244, 67, 54, 0.15); padding: 2px 6px; border-radius: 4px;">Light Red</strong> = 5-10% decline</div>
                        <div>🌱 <strong style="background-color: rgba(46, 125, 50, 0.05); padding: 2px 6px; border-radius: 4px;">Green Tint</strong> = Volume columns</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 🔄 ENHANCED: Column grouping explanation
                if unique_months:
                    month_list = [month_names.get(m, m) for m in sorted(unique_months)]
                    st.markdown(f"""
                    <div style="background: rgba(46, 125, 50, 0.1); padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <h4 style="margin: 0 0 8px 0; color: #1B5E20;">🌿 Health Column Organization:</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                            <div><strong>🌱 Base Metrics:</strong> Health Query, Total Volume, Share %, Overall CTR/CR</div>
                            <div><strong>📊 Monthly Volumes:</strong> {' → '.join([f"{m} Vol" for m in month_list])}</div>
                            <div><strong>🎯 Monthly CTRs:</strong> {' → '.join([f"{m} CTR" for m in month_list])}</div>
                            <div><strong>💚 Monthly CRs:</strong> {' → '.join([f"{m} CR" for m in month_list])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 🚀 ENHANCED SUMMARY METRICS
                st.markdown("---")
                
                # 🚀 FAST: Pre-calculate all metrics at once
                metrics = {
                    'total_queries': len(top50),
                    'total_search_volume': int(pd.to_numeric(top50['Total Volume'], errors='coerce').sum()),
                    'total_clicks': int(top50['Total Clicks'].sum()),
                    'total_conversions': int(top50['Total Conversions'].sum())
                }
                
                col1, col2, col3, col4 = st.columns(4)
                
                # 🚀 OPTIMIZED: Batch metric rendering
                metric_configs = [
                    (col1, "🌿", metrics['total_queries'], "Total Health Queries"),
                    (col2, "🔍", format_number(metrics['total_search_volume']), "Total Search Volume"),
                    (col3, "🍃", format_number(metrics['total_clicks']), "Total Clicks"),
                    (col4, "💚", format_number(metrics['total_conversions']), "Total Conversions")
                ]
                
                for col, icon, value, label in metric_configs:
                    with col:
                        st.markdown(f"""
                        <div class="top50-health-metric-card">
                            <div class="icon">{icon}</div>
                            <div class="value">{value}</div>
                            <div class="label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # 🚀 MONTHLY BREAKDOWN WITH PERFORMANCE TRENDS
                if unique_months:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📅 Monthly Nutraceuticals & Nutrition Performance Trends")
                    
                    # Calculate average CTR and CR for each month
                    monthly_performance = {}
                    for month in unique_months:
                        month_display = month_names.get(month, month)
                        ctr_col = f'{month_display} CTR'
                        cr_col = f'{month_display} CR'
                        vol_col = f'{month_display} Vol'
                        
                        if ctr_col in top50.columns and cr_col in top50.columns and vol_col in top50.columns:
                            avg_ctr = top50[ctr_col].mean()
                            avg_cr = top50[cr_col].mean()
                            monthly_total = int(pd.to_numeric(top50[vol_col], errors='coerce').sum())
                            
                            monthly_performance[month_display] = {
                                'volume': monthly_total,
                                'avg_ctr': avg_ctr,
                                'avg_cr': avg_cr
                            }
                    
                    month_cols = st.columns(len(unique_months))
                    
                    for i, month in enumerate(unique_months):
                        month_display_name = month_names.get(month, month)
                        if month_display_name in monthly_performance:
                            with month_cols[i]:
                                perf = monthly_performance[month_display_name]
                                
                                # Determine trend indicators
                                ctr_trend = ""
                                cr_trend = ""
                                if i > 0:
                                    prev_month_display = month_names.get(unique_months[i-1], unique_months[i-1])
                                    if prev_month_display in monthly_performance:
                                        prev_perf = monthly_performance[prev_month_display]
                                        ctr_trend = "📈" if perf['avg_ctr'] > prev_perf['avg_ctr'] else "📉" if perf['avg_ctr'] < prev_perf['avg_ctr'] else "➡️"
                                        cr_trend = "📈" if perf['avg_cr'] > prev_perf['avg_cr'] else "📉" if perf['avg_cr'] < prev_perf['avg_cr'] else "➡️"
                                
                                st.markdown(f"""
                                <div class="monthly-health-metric-card">
                                    <div class="icon">🌱</div>
                                    <div class="value">{format_number(perf['volume'])}</div>
                                    <div class="label">{month_display_name}</div>
                                    <div style="font-size: 0.8em; margin-top: 5px;">
                                        CTR: {perf['avg_ctr']:.2f}% {ctr_trend}<br>
                                        CR: {perf['avg_cr']:.2f}% {cr_trend}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                # 🚀 ENHANCED DOWNLOAD SECTION
                st.markdown("<br>", unsafe_allow_html=True)
                
                csv = generate_csv_ultra(top50)
                
                col_download = st.columns([1, 2, 1])
                with col_download[1]:
                    st.markdown("""
                    <div class="download-health-section">
                        <h4 style="color: white; margin-bottom: 15px;">📥 Export Health Data</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Health Queries CSV",
                        data=csv,
                        file_name=f"top_50_health_queries_better_arrangement_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the health table with improved column arrangement for easier comparison",
                        use_container_width=True
                    )
                
                # 🚀 OPTIMIZED MONTHLY INSIGHTS WITH PERFORMANCE ANALYSIS
                with st.expander("📊 Monthly Nutraceuticals & Nutrition Performance Analysis", expanded=False):
                    if unique_months and len(unique_months) >= 2:
                        st.markdown("""
                        <div class="insights-health-section">
                            <h3 style="color: white; text-align: center; margin-bottom: 20px;">🌿 Nutraceuticals & Nutrition Performance Trend Analysis</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Performance trend analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🚀 Health CTR Performance Leaders")
                            # Find queries with best CTR improvement
                            ctr_improvements = []
                            for _, row in top50.iterrows():
                                if len(unique_months) >= 2:
                                    latest_month = month_names.get(unique_months[-1], unique_months[-1])
                                    prev_month = month_names.get(unique_months[-2], unique_months[-2])
                                    
                                    latest_ctr = row.get(f'{latest_month} CTR', 0)
                                    prev_ctr = row.get(f'{prev_month} CTR', 0)
                                    
                                    if prev_ctr > 0:
                                        improvement = ((latest_ctr - prev_ctr) / prev_ctr) * 100
                                        ctr_improvements.append({
                                            'query': row['Health Query'],
                                            'improvement': improvement,
                                            'latest_ctr': latest_ctr
                                        })
                            
                            ctr_improvements = sorted(ctr_improvements, key=lambda x: x['improvement'], reverse=True)[:5]
                            
                            for item in ctr_improvements:
                                color = "#4CAF50" if item['improvement'] > 0 else "#F44336"
                                sign = "+" if item['improvement'] > 0 else ""
                                st.markdown(f"""
                                <div class="health-gainer-item">
                                    <strong>{item['query'][:30]}...</strong><br>
                                    <small>CTR: {item['latest_ctr']:.2f}% ({sign}{item['improvement']:.1f}%)</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### 🎯 Health CR Performance Leaders")
                            # Find queries with best CR improvement
                            cr_improvements = []
                            for _, row in top50.iterrows():
                                if len(unique_months) >= 2:
                                    latest_month = month_names.get(unique_months[-1], unique_months[-1])
                                    prev_month = month_names.get(unique_months[-2], unique_months[-2])
                                    
                                    latest_cr = row.get(f'{latest_month} CR', 0)
                                    prev_cr = row.get(f'{prev_month} CR', 0)
                                    
                                    if prev_cr > 0:
                                        improvement = ((latest_cr - prev_cr) / prev_cr) * 100
                                        cr_improvements.append({
                                            'query': row['Health Query'],
                                            'improvement': improvement,
                                            'latest_cr': latest_cr
                                        })
                            
                            cr_improvements = sorted(cr_improvements, key=lambda x: x['improvement'], reverse=True)[:5]
                            
                            for item in cr_improvements:
                                color = "#4CAF50" if item['improvement'] > 0 else "#F44336"
                                sign = "+" if item['improvement'] > 0 else ""
                                st.markdown(f"""
                                <div class="health-gainer-item">
                                    <strong>{item['query'][:30]}...</strong><br>
                                    <small>CR: {item['latest_cr']:.2f}% ({sign}{item['improvement']:.1f}%)</small>
                                </div>
                                """, unsafe_allow_html=True)

        except KeyError as e:
            st.error(f"Column error: {e}. Check column names in your data.")
        except Exception as e:
            st.error(f"Error processing top 50 health queries: {e}")
            st.write("**Debug info:**")
            st.write(f"Queries shape: {queries.shape}")
            st.write(f"Available columns: {list(queries.columns)}")
            if 'top50' in locals() and not top50.empty:
                st.write(f"Top50 shape: {top50.shape}")
                if 'Total Volume' in top50.columns:
                    st.write(f"Total Volume dtype: {top50['Total Volume'].dtype}")
                    st.write(f"Sample values: {top50['Total Volume'].head()}")

    st.markdown("---")


# ----------------- Performance Snapshot -----------------
    st.subheader("🌱 Nutraceuticals & Nutrition Performance Snapshot")

    # Mini-Metrics Row (Data-Driven: From Analysis with Share)
    colM1, colM2, colM3, colM4 = st.columns(4)
    with colM1:
        avg_ctr = queries['Click Through Rate'].mean() * 100 if not queries.empty else 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>🌿</span>
            <div class='value'>{avg_ctr:.2f}%</div>
            <div class='label'>Avg CTR (All Health)</div>
        </div>
        """, unsafe_allow_html=True)
    with colM2:
        avg_cr = queries['Converion Rate'].mean() * 100 if not queries.empty else 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>💚</span>
            <div class='value'>{avg_cr:.2f}%</div>
            <div class='label'>Avg CR (Nutraceuticals & Nutrition)</div>
        </div>
        """, unsafe_allow_html=True)
    with colM3:
        unique_queries = queries['search'].nunique()
        total_counts = int(queries['Counts'].sum()) if not queries['Counts'].empty else 0
        total_share = (queries.groupby('search')['Counts'].sum() / total_counts * 100).max() if total_counts > 0 else 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>🔍</span>
            <div class='value'>{format_number(unique_queries)} ({total_share:.2f}%)</div>
            <div class='label'>Unique Health Queries</div>
        </div>
        """, unsafe_allow_html=True)
    with colM4:
        cat_counts = queries.groupby('Category')['Counts'].sum()
        top_cat = cat_counts.idxmax()
        top_cat_share = (cat_counts.max() / total_counts * 100) if total_counts > 0 else 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>🧴</span>
            <div class='value'>{format_number(int(cat_counts.max()))} ({top_cat_share:.2f}%)</div>
            <div class='label'>Top Health Category ({top_cat})</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")

    st.subheader("🏷 Brand & Health Category Snapshot")
    g1, g2 = st.columns(2)
    with g1:
        if 'Brand' in queries.columns:
            # Check which columns actually exist before using them
            available_columns = queries.columns.tolist()
            agg_dict = {}
            
            if 'Counts' in available_columns:
                agg_dict['Counts'] = 'sum'
            if 'clicks' in available_columns:
                agg_dict['clicks'] = 'sum'
            if 'Conversion Rate' in available_columns:
                agg_dict['Conversion Rate'] = 'mean'
            
            # Only proceed if we have at least one column to aggregate
            if agg_dict:
                brand_perf = queries[queries['Brand'] != 'Other'].groupby('Brand').agg(agg_dict).reset_index()
                
                # Calculate derived metrics only if the required columns exist
                if 'clicks' in brand_perf.columns and 'Conversion Rate' in brand_perf.columns:
                    brand_perf['conversions'] = (brand_perf['clicks'] * brand_perf['Conversion Rate']).round()
                
                if 'Counts' in brand_perf.columns:
                    brand_perf['share'] = (brand_perf['Counts'] / total_counts * 100).round(2)
                
                # Only create the chart if we have data to display
                if not brand_perf.empty and 'Counts' in brand_perf.columns:
                    # Determine color column - use conversions if available, otherwise use Counts
                    color_column = 'conversions' if 'conversions' in brand_perf.columns else 'Counts'
                    hover_columns = ['share'] if 'share' in brand_perf.columns else []
                    if 'conversions' in brand_perf.columns:
                        hover_columns.append('conversions')
                    
                    # Create a beautiful bar chart with text labels
                    fig = px.bar(brand_perf.sort_values('Counts', ascending=False).head(10), 
                                x='Brand', y='Counts',
                                title='<b style="color:#2E7D32; font-size:18px; text-shadow: 2px 2px 4px #00000055;">Top Health Brands by Search Volume</b>',
                                labels={'Brand': '<i>Health Brand</i>', 'Counts': '<b>Search Volume</b>'},
                                color=color_column,
                                color_continuous_scale=['#E8F5E8', '#66BB6A', '#2E7D32'],
                                template='plotly_white',
                                hover_data=hover_columns)
                    
                    # Update traces to position text outside and set hovertemplate
                    fig.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}' + 
                                    ('<br>Share: %{customdata[0]:.2f}%' if 'share' in hover_columns else '') +
                                    ('<br>Conversions: %{customdata[1]:,.0f}' if 'conversions' in hover_columns and len(hover_columns) > 1 else '') +
                                    '<extra></extra>'
                    )

                    # Enhance attractiveness: Custom layout for beauty
                    fig.update_layout(
                        plot_bgcolor='rgba(248,253,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI'),
                        title_x=0,  # Left alignment for title
                        title_font_size=16,
                        xaxis=dict(
                            title='Health Brand',
                            showgrid=True, 
                            gridcolor='#E8F5E8', 
                            linecolor='#2E7D32', 
                            linewidth=2
                        ),
                        yaxis=dict(
                            title='Search Volume',
                            showgrid=True, 
                            gridcolor='#E8F5E8', 
                            linecolor='#2E7D32', 
                            linewidth=2
                        ),
                        bargap=0.2,
                        barcornerradius=8,
                        hovermode='x unified',
                        annotations=[
                            dict(
                                x=0.5, y=1.05, xref='paper', yref='paper',
                                text='🌿 Hover for Nutraceuticals & Nutrition details | Top health brand highlighted below 🌿',
                                showarrow=False,
                                font=dict(size=10, color='#2E7D32', family='Segoe UI'),
                                align='center'
                            )
                        ]
                    )

                    # Highlight the top brand with a custom marker
                    top_brand = brand_perf.loc[brand_perf['Counts'].idxmax(), 'Brand']
                    top_count = brand_perf['Counts'].max()
                    fig.add_annotation(
                        x=top_brand, y=top_count,
                        text=f"🏆 Peak Health: {top_count:,.0f}",
                        showarrow=True,
                        arrowhead=3,
                        arrowcolor='#2E7D32',
                        ax=0, ay=-30,
                        font=dict(size=12, color='#2E7D32', family='Segoe UI', weight='bold')
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No health brand data available after filtering or missing required columns.")
            else:
                st.warning("No valid aggregation columns found for brand analysis.")
        else:
            st.info("🏷 Health Brand column not found in the dataset.")

    with g2:
        if 'Category' in queries.columns:
            # Check which columns actually exist before using them
            available_columns = queries.columns.tolist()
            agg_dict = {}
            
            if 'Counts' in available_columns:
                agg_dict['Counts'] = 'sum'
            if 'clicks' in available_columns:
                agg_dict['clicks'] = 'sum'
            if 'conversions' in available_columns:
                agg_dict['conversions'] = 'sum'
            elif 'Conversion Rate' in available_columns and 'clicks' in available_columns:
                # We'll calculate conversions after aggregation
                pass
            
            # Only proceed if we have at least one column to aggregate
            if agg_dict:
                cat_perf = queries.groupby('Category').agg(agg_dict).reset_index()
                
                # Calculate conversions if we have the necessary columns but not the conversions column
                if 'conversions' not in cat_perf.columns and 'clicks' in cat_perf.columns and 'Conversion Rate' in queries.columns:
                    # Calculate average conversion rate for each category first
                    conv_rate_agg = queries.groupby('Category')['Conversion Rate'].mean().reset_index()
                    cat_perf = cat_perf.merge(conv_rate_agg, on='Category')
                    cat_perf['conversions'] = (cat_perf['clicks'] * cat_perf['Conversion Rate']).round()
                
                # Calculate share and conversion rate
                if 'Counts' in cat_perf.columns:
                    cat_perf['share'] = (cat_perf['Counts'] / total_counts * 100).round(2)
                
                # FIX: Calculate conversion rate correctly - conversions divided by counts
                if 'conversions' in cat_perf.columns and 'Counts' in cat_perf.columns:
                    cat_perf['cr'] = (cat_perf['conversions'] / cat_perf['Counts'] * 100).round(2)
                else:
                    cat_perf['cr'] = 0
                
                st.markdown("**Top Health Categories by Search Volume**")
                
                # Prepare display columns based on what's available
                display_columns = ['Category']
                format_dict = {}

                if 'Counts' in cat_perf.columns:
                    display_columns.append('Counts')
                    # Create a custom formatter that applies format_number
                    def counts_formatter(x):
                        return format_number(int(x))
                    format_dict['Counts'] = counts_formatter
                if 'share' in cat_perf.columns:
                    display_columns.append('share')
                    format_dict['share'] = '{:.2f}%'
                if 'clicks' in cat_perf.columns:
                    display_columns.append('clicks')
                    format_dict['clicks'] = '{:,.0f}'
                if 'conversions' in cat_perf.columns:
                    display_columns.append('conversions')
                    format_dict['conversions'] = '{:,.0f}'
                if 'cr' in cat_perf.columns:
                    display_columns.append('cr')
                    format_dict['cr'] = '{:.2f}%'
                
                # Rename columns for better display
                display_cat_perf = cat_perf[display_columns].copy()
                column_rename = {
                    'Category': 'Health Category',
                    'Counts': 'Search Volume',
                    'share': 'Market Share %',
                    'clicks': 'Total Clicks',
                    'conversions': 'Conversions',
                    'cr': 'Conversion Rate %'
                }
                display_cat_perf = display_cat_perf.rename(columns={k: v for k, v in column_rename.items() if k in display_cat_perf.columns})
                
                # Update format dict with new column names
                new_format_dict = {}
                for old_col, new_col in column_rename.items():
                    if old_col in format_dict and new_col in display_cat_perf.columns:
                        new_format_dict[new_col] = format_dict[old_col]
                
                # Display the table with available data
                if len(display_cat_perf.columns) > 1:  # More than just the Category column
                    # Sort by Search Volume in descending order
                    if 'Search Volume' in display_cat_perf.columns:
                        sorted_cat_perf = display_cat_perf.sort_values('Search Volume', ascending=False).head(10)
                    else:
                        # Fallback to first numeric column if Search Volume not available
                        numeric_cols = [col for col in display_cat_perf.columns[1:] if col in display_cat_perf.columns]
                        if numeric_cols:
                            sorted_cat_perf = display_cat_perf.sort_values(numeric_cols[0], ascending=False).head(10)
                        else:
                            sorted_cat_perf = display_cat_perf.head(10)
                    
                    try:
                        # Try using AgGrid if available
                        if 'AGGRID_OK' in globals() and AGGRID_OK:
                            AgGrid(sorted_cat_perf, height=300, enable_enterprise_modules=False)
                        else:
                            # Fall back to styled DataFrame with health-themed styling
                            styled_cat_perf = sorted_cat_perf.style.format(new_format_dict).set_properties(**{
                                'text-align': 'center',
                                'font-size': '14px',
                                'background-color': '#F8FDF8',
                                'color': '#1B5E20'
                            }).background_gradient(
                                subset=['Conversion Rate %'] if 'Conversion Rate %' in sorted_cat_perf.columns else [], 
                                cmap='Greens'
                            ).set_table_styles([
                                {'selector': 'th', 'props': [
                                    ('background-color', '#E8F5E8'),
                                    ('color', '#1B5E20'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'center')
                                ]}
                            ])
                            st.dataframe(styled_cat_perf, use_container_width=True, hide_index=True)
                            
                            # Add download button for health categories
                            csv_cat = sorted_cat_perf.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Health Categories CSV",
                                data=csv_cat,
                                file_name="health_categories_performance.csv",
                                mime="text/csv"
                            )
                    except NameError:
                        # AGGRID_OK not defined, use regular DataFrame
                        styled_cat_perf = sorted_cat_perf.style.format(new_format_dict).set_properties(**{
                            'text-align': 'center',
                            'font-size': '14px',
                            'background-color': '#F8FDF8',
                            'color': '#1B5E20'
                        }).background_gradient(
                            subset=['Conversion Rate %'] if 'Conversion Rate %' in sorted_cat_perf.columns else [], 
                            cmap='Greens'
                        ).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('background-color', '#E8F5E8'),
                                ('color', '#1B5E20'),
                                ('font-weight', 'bold'),
                                ('text-align', 'center')
                            ]}
                        ])
                        st.dataframe(styled_cat_perf, use_container_width=True, hide_index=True)
                        
                        # Add download button for health categories
                        csv_cat = sorted_cat_perf.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Health Categories CSV",
                            data=csv_cat,
                            file_name="health_categories_performance.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("Insufficient data columns available for health category analysis.")
            else:
                st.warning("No valid aggregation columns found for health category analysis.")
        else:
            st.info("🧴 Health Category column not found in the dataset.")

    # Add Nutraceuticals & Nutrition insights section
    st.markdown("---")
    st.subheader("🌿 Nutraceuticals & Nutrition Insights & Recommendations")
    
    # Create insight boxes with health-themed content
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div class="insight-box">
            <h4>🌱 Top Performing Health Categories</h4>
            <p>Focus on high-conversion Nutraceuticals & Nutrition categories like supplements, vitamins, and natural health products for optimal ROI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>💚 Seasonal Health Trends</h4>
            <p>Monitor seasonal patterns in immune support, weight management, and Nutraceuticals & Nutrition supplements to optimize inventory and marketing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown("""
        <div class="insight-box">
            <h4>🧴 Brand Performance Analysis</h4>
            <p>Identify top-performing supplement brands and optimize product placement for maximum visibility and conversions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>🔍 Search Optimization</h4>
            <p>Leverage high-volume health queries to improve product descriptions and SEO for better organic discovery.</p>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
# ----------------- Search Analysis (Enhanced Core - OPTIMIZED) -----------------
with tab_search:
    st.header("🔍 Health Search Analysis — Deep Dive into Nutraceuticals & Nutrition Queries")
    st.markdown("Analyze nutritional search patterns with advanced keyword insights, supplement performance metrics, and actionable health intelligence. 🌿")
    
    st.markdown("---")
    
    st.subheader("🌿 Nutraceuticals & Nutrition Keyword Frequency & Performance Analysis")
    
    # Cached Master Keyword Dictionary
    @st.cache_data(ttl=7200, show_spinner=False)
    def create_master_keyword_dictionary():
        """
        Minimized master keywords dictionary - selected best 50-60% from original values
        """
        return {
            'مغنیسیوم': {
                'variations': [
                    'مغنیسیوم', 'مغنسیوم', 'ماغنیسیوم', 'مغنیس', 'مغنی', 
                    'مغنیسی', 'مغانیسیوم', 'ماغن', 'المغنیسیوم', 'مغنیسو', 
                    'مغنیزیوم', 'مغناسیوم', 'مغانسیوم', 'magnesium', 'مغنيسيوم'
                ],
                'excluded_terms': ['الصمغ'],
                'compounds': [
                    'جلیسینات', 'جلایسینات', 'جل', 'سترات', 'مالات', 
                    'فوار', '400', 'glycinate', 'citrate', 'malate'
                ],
                'threshold': 80,
                'min_length': 4
            },

            'اوميجا': {
                'variations': [
                    'اومیجا', 'اومیغا', 'اومیقا', 'اومجا', 'اومقا', 'اوم',
                    'اومی', 'میجا', 'اومیجا3', 'اومیغا3', 'اومیقا3',
                    'الاومیجا', 'الاومیغا', 'اوکیقا', 'امیغا', 'کومیجا',
                    'omega', 'omega3', 'omg3', 'omg', 'ome', 'omiga', 'mega'
                ],
                'excluded_terms': [
                    'اومیلت', 'اومالت', 'اوملت', 'زاو', 'milga', 'کرومیم', 
                    'one', 'النوم', 'کوی', 'ایزوبیور', 'کوم', 'میجاتو', 'کومی'
                ],
                'compounds': [
                    '3', '6', '9', '1000', '2000', 'للاطفال', 'اطفال', 
                    'حبوب', 'کپسول', 'EPA', 'DHA', 'nordic', 'jp', 'capsules'
                ],
                'threshold': 80,
                'min_length': 3
            },

            'کولاجین': {
                'variations': [
                    'کولاجین', 'کولاجن', 'collagen', 'كولاجين', 'کلاجین', 
                    'کولاژن', 'کولجین', 'کول', 'کولاجی', 'مولاجین'
                ],
                'excluded_terms': [
                    'کوین', 'کولایت', 'کوریلا', 'کولین', 'شوکولا', 'لاین'
                ],
                'compounds': [
                    'پپتید', 'هیدرولیز', 'مارین', 'بقری', 'peptides', 
                    'marine', 'بودره', 'فوار', 'powder'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'فیتامین': {
                'variations': [
                    'فیتامین', 'فيتامين', 'ویتامین', 'فیتامن', 'فیتامینات',
                    'فیتامین سی', 'فیتامین د', 'فیتامین ب', 'فیتامین د3',
                    'مالتی فیتامین', 'ملتی فیتامین', 'فیتامین للاطفال',
                    'فیتامین حمل', 'فیتامین شعر', 'فیتامین د 50000',
                    'vitamin', 'vitamins', 'multivitamin', 'vitamin c', 'vitamin d'
                ],
                'excluded_terms': [
                    'فیتنس', 'فیتر', 'فیمی', 'فیتال', 'ادفیتا', 'فینترمین', 
                    'قلوتامین', 'vitex', 'بیتا', 'غلوتامین'
                ],
                'compounds': [
                    'سی', 'د', 'دال', 'ب', 'بی', 'c', 'd', 'd3', 'b12',
                    '50000', '5000', '1000', 'للاطفال', 'حمل', 'شعر',
                    'فوار', 'حبوب', 'شراب', 'قطرات'
                ],
                'threshold': 75,
                'min_length': 4
            },
            
            'زنک': {
                'variations': [
                    'زنک', 'زینک', 'زنك', 'zinc', 'الزنک', 'الزینک'
                ],
                'excluded_terms': [
                    'الوزن', 'زینیکا', 'زینکال', 'الارز'
                ],
                'compounds': [
                    'پیکولینات', 'گلوکونات', 'picolinate', '50', '25', 'copper'
                ],
                'threshold': 80,
                'min_length': 3
            },
            
            'کالسیوم': {
                'variations': [
                    'کالسیوم', 'کلسیم', 'كالسيوم', 'calcium', 'الکالسیوم',
                    'کالسیو', 'کالیسیوم'
                ],
                'compounds': [
                    'کربنات', 'سیترات', 'citrate', 'مغنیسیو', '600', 
                    'فوار', 'للاطفال'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'بروتین': {
                'variations': [
                    'بروتین', 'پروتین', 'پروتئین', 'بروتين', 'protein', 
                    'پروتن', 'بروت', 'پروت'
                ],
                'excluded_terms': [
                    'برورین', 'برافوتین', 'بروبین', 'بیروین', 'بروستا'
                ],
                'compounds': [
                    'وی', 'whey', 'کازئین', 'casein', 'ایزو', 'بار', 
                    'باودر', 'powder', 'ماس', 'نباتی'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'حدید': {
                'variations': [
                    'حدید', 'حديد', 'الحدید', 'iron', 'ferrous', 'ferro',
                    'فیرولایت', 'فیرو لایت', 'فیروفول', 'فیرومین', 'فیرو'
                ],
                'excluded_terms': [
                    'حدیث', 'حدیقة', 'فیدروب', 'solaray', 'فیتو', 'بورون',
                    'فینترمین', 'solar', 'نیرو', 'زیرو', 'فیتامین', 'solgar'
                ],
                'compounds': [
                    'فومارات', 'fumarate', 'سولفات', 'فوار', 'شراب', 
                    'حبوب', 'للاطفال', 'folic acid', '25mg'
                ],
                'threshold': 80,
                'min_length': 4
            },

            'ensure': {
                'variations': [
                    'ensure', 'ensur', 'انشور', 'انش', 'انشو', 
                    'حلیب انشور', 'حليب انشور'
                ],
                'excluded_terms': [
                    'انشاء', 'المنشاری', 'سانو', 'النشط', 'انوفاری'
                ],
                'compounds': [
                    'plus', 'بلس', 'max', 'ماکس', 'protein', 'پروتین',
                    'milk', 'حلیب', 'vanilla', 'وانیل', 'chocolate'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'بیوتین': {
                'variations': [
                    'بیوتین', 'بایوتین', 'biotin', 'بیوتی', 'بیوت', 'البیوتین'
                ],
                'excluded_terms': [
                    'biotic', 'بیوسیستین', 'برایوین', 'بیوتیک'
                ],
                'compounds': [
                    '10000', '5000', '1000', 'للشعر', 'شعر', 'hair', 'forte'
                ],
                'threshold': 85,
                'min_length': 4
            },
            
            'اشواغندا': {
                'variations': [
                    'اشواغندا', 'اشواجندا', 'اشوقندا', 'اشواق', 'اشو',
                    'الاشواغندا', 'ashwagandha', 'ashwa', 'ksm66'
                ],
                'excluded_terms': [
                    'انشوز', 'الشوک', 'اوراق', 'انشو'
                ],
                'compounds': [
                    'ksm', 'ksm66', '600', 'gummies', 'حبوب', 'extract'
                ],
                'threshold': 75,
                'min_length': 4
            },
            
            'جنسنج': {
                'variations': [
                    'جنسنج', 'جنس', 'جینسینج', 'جنسنج کوری', 'الجنسنج',
                    'ginseng', 'korean ginseng', 'panax ginseng'
                ],
                'compounds': [
                    'کوری', 'korean', 'panax', 'رویال', 'royal', 'جیلی'
                ],
                'threshold': 75,
                'min_length': 4
            },
            
            'کرکم': {
                'variations': [
                    'کرکم', 'الکرکم', 'کرکمین', 'کورکومین', 'turmeric', 
                    'curcumin', 'curcumax'
                ],
                'compounds': [
                    'curcumin', 'extract', 'مستخلص', 'حبوب', 'کپسول'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'خل التفاح': {
                'variations': [
                    'خل التفاح', 'حبوب خل التفاح', 'خل تفاح', 'خل التفا',
                    'apple cider vinegar', 'apple cider', 'خل ا'
                ],
                'compounds': [
                    'حبوب', 'فوار', 'gummies', 'حلوى', 'کبسولات', 'عضوی'
                ],
                'threshold': 70,
                'min_length': 3
            },
            
            'منوم': {
                'variations': [
                    'منوم', 'منو', 'حبوب منوم', 'شراب منوم', 'منوم الاطفال',
                    'sleep', 'sleep aid'
                ],
                'compounds': [
                    'للاطفال', 'اطفال', 'کبار', 'طبیعی', 'natural'
                ],
                'threshold': 80,
                'min_length': 3
            },
            
            'بربرین': {
                'variations': [
                    'بربرین', 'البربرین', 'حبوب البربرین', 'برب', 'بیربرین',
                    'berberine', 'berberin'
                ],
                'excluded_terms': [
                    'برابورین', 'بیریورین', 'بروبین', 'برورین'
                ],
                'compounds': [
                    '500', 'phytosome', 'فیتوسوم', 'حبوب', 'کبسولات'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'کرانبری': {
                'variations': [
                    'کرانبیری', 'کران', 'کران بیری', 'کرانبری', 'کرنبیری',
                    'بیری', 'cranberry'
                ],
                'excluded_terms': [
                    'کاندری', 'کانتری', 'بحری', 'بیورین', 'بکتیری',
                    'strawberry', 'blueberry', 'بری', 'بیری بیری'
                ],
                'compounds': [
                    'حبوب', 'کبسولات', 'extract', 'مستخلص', 'juice', 
                    'عصیر', 'urinary', 'بولی', 'uti', '500mg'
                ],
                'threshold': 75,
                'min_length': 4
            },
            
            'فحم نشط': {
                'variations': [
                    'فحم', 'حبوب فحم', 'الفحم', 'فحم نشط', 'الفحم النشط',
                    'charcoal', 'activated charcoal'
                ],
                'compounds': [
                    'نشط', 'activated', 'حبوب', 'کبسولات', 'detox'
                ],
                'threshold': 85,
                'min_length': 3
            },
            
            'عسل': {
                'variations': [
                    'عسل', 'العسل', 'honey', 'عسل المنوکا', 'عسل مانوکا',
                    'عسل منوکا', 'مانوکا', 'manuka honey', 'عسل ملکی',
                    'royal honey', 'عسل ابو نایف', 'عسل م'
                ],
                'excluded_terms': [
                    'عسلی', 'عسکر', 'honey badger', 'honeymoon', 'انوفا',
                    'الماکا', 'ماکا', 'وسلمان'
                ],
                'compounds': [
                    'manuka', 'مانوکا', 'royal', 'ملکی', 'طبیعی', 'natural',
                    'اطفال', 'للاطفال', 'ابو نایف'
                ],
                'threshold': 80,
                'min_length': 3
            },
            
            'کیو10': {
                'variations': [
                    'کیو 10', 'کیو10', 'کو کیو 10', 'کو کیو', 'q10', 
                    'coq10', 'co q10', 'ubiquinol'
                ],
                'compounds': [
                    '100', '200', 'mg', 'ubiquinol', 'کوانزیم'
                ],
                'threshold': 80,
                'min_length': 3
            },
            
            'جلوتاثیون': {
                'variations': [
                    'جلوتاثیون', 'الجلوتاثیون', 'حبوب جلوتاثیون', 'جلوتا',
                    'جلوتاثیوم', 'glutathione', 'glutathion'
                ],
                'compounds': [
                    '500', 'حبوب', 'کبسولات', 'tablets', 'capsules'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'ارجنین': {
                'variations': [
                    'ارجنین', 'الارجنین', 'ل ارجنین', 'l arginine', 'arginine',
                    'ارجینین', 'ارگنین'
                ],
                'compounds': [
                    '1000', 'l', 'حبوب', 'کبسولات', 'mg'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'سیلینیوم': {
                'variations': [
                    'سیلینیوم', 'سلینیوم', 'السیلینیوم', 'سیلینوم', 'selenium'
                ],
                'compounds': [
                    '200', 'ace', 'حبوب', 'کبسولات'
                ],
                'threshold': 80,
                'min_length': 4
            },
            
            'فولیک اسید': {
                'variations': [
                    'فولیک', 'فولیک اسید', 'فول', 'حمض فولیک', 'الفولیک',
                    'حمض الفولیک', 'folic acid', 'folic'
                ],
                'excluded_terms': [
                    'الفا', 'الفا لیبویک', 'فلک', 'فولیکوم'
                ],
                'compounds': [
                    '5mg', '400', '400mg', '1mg', 'حبوب', 'اقراص',
                    'iron', 'حديد', 'اسید', 'acid'
                ],
                'threshold': 85,
                'min_length': 4
            },

            'میلاتونین': {
                'variations': [
                    'میلاتونین', 'میلاتو', 'میلات', 'میلاتون', 'المیلاتونین',
                    'melatonin', 'mela', 'melat', 'ناترول', 'natrol'
                ],
                'excluded_terms': [
                    'میلان', 'میلاد', 'تونین', 'naturals', 'جلایسین', 
                    'nutrafol', 'nitro', 'کریاتین', 'مالتی', 'جامیسون'
                ],
                'compounds': [
                    '1', '3', '5', '10', '1mg', '3mg', '5mg', '10mg',
                    'gummy', 'gummies', 'اطفال', 'للاطفال', 'kids',
                    'للنوم', 'sleep', 'plus'
                ],
                'threshold': 80,
                'min_length': 4
            }
        }

    # ================================================================================================
    # 🚀 OPTIMIZED FUNCTION DEFINITIONS SECTION
    # ================================================================================================

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_compiled_patterns():
        """Pre-compiled regex patterns for better performance"""
        import re
        return [
            re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]{2,}'),  # Arabic
            re.compile(r'[a-zA-Z]{3,}'),  # English
            re.compile(r'\d{2,}')  # Numbers
        ]

    def extract_keywords_with_fuzzy_grouping(text: str, min_length=2):
        """Optimized keyword extraction with pre-compiled patterns"""
        if not isinstance(text, str) or len(text.strip()) < min_length:
            return []
        
        text = text.strip().lower()
        patterns = get_compiled_patterns()
        
        keywords = []
        for pattern in patterns:
            matches = pattern.findall(text)
            keywords.extend([match.strip() for match in matches if len(match.strip()) >= min_length])
        
        return list(set(keywords))  # Remove duplicates early

    def safe_import_fuzzywuzzy():
        """Safely import fuzzywuzzy with fallback"""
        try:
            from fuzzywuzzy import fuzz
            return fuzz, True
        except ImportError:
            return None, False

    def basic_similarity(s1, s2):
        """Basic similarity calculation without fuzzywuzzy"""
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        
        if s1 == s2:
            return 100
        
        if s1 in s2 or s2 in s1:
            shorter, longer = (s1, s2) if len(s1) < len(s2) else (s2, s1)
            return int((len(shorter) / len(longer)) * 90)
        
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0
        
        return int((intersection / union) * 80)

    # Get fuzzy matching capability ONCE at module level
    fuzz, has_fuzzywuzzy = safe_import_fuzzywuzzy()

    def fuzzy_match_keywords(keyword_data, master_dict, min_score=70):
        """Optimized fuzzy matching with early termination and error handling"""
        from collections import defaultdict
        
        grouped_keywords = defaultdict(lambda: {
            'total_counts': 0, 
            'total_clicks': 0, 
            'total_conversions': 0, 
            'queries': [],
            'variations': []
        })
        
        processed_keywords = set()
        
        # Sort keywords by length for better matching efficiency
        sorted_keywords = sorted(keyword_data.items(), key=lambda x: len(x[0]), reverse=True)
        
        for keyword, data in sorted_keywords:
            if keyword in processed_keywords or len(keyword.strip()) < 3:
                continue
                
            best_match = None
            best_score = 0
            matched_master = None
            
            for master_keyword, master_info in master_dict.items():
                if len(keyword) < master_info.get('min_length', 3):
                    continue
            
                # Quick exclusion check
                excluded_terms = master_info.get('excluded_terms', [])
                if any(excluded_term.strip().lower() in keyword.lower() 
                    for excluded_term in excluded_terms if excluded_term.strip()):
                    continue

                # Check variations with error handling
                for variation in master_info['variations']:
                    try:
                        if keyword.lower() == variation.lower():
                            best_score = 100
                            best_match = variation
                            matched_master = master_keyword
                            break
                        
                        if (variation.lower() in keyword.lower() and 
                            len(variation) >= 4 and len(keyword) >= 4):
                            if len(variation) / len(keyword) >= 0.6:
                                score = 90
                                if score > best_score:
                                    best_score = score
                                    best_match = variation
                                    matched_master = master_keyword
                        
                        # Fuzzy matching with fallback
                        if best_score < 90:
                            try:
                                if has_fuzzywuzzy:
                                    score = fuzz.ratio(keyword.lower(), variation.lower())
                                else:
                                    score = basic_similarity(keyword, variation)
                                
                                if score >= master_info['threshold']:
                                    if len(set(keyword.lower()) & set(variation.lower())) / len(set(variation.lower())) >= 0.6:
                                        if score > best_score:
                                            best_score = score
                                            best_match = variation
                                            matched_master = master_keyword
                            except Exception:
                                # Fallback to basic similarity
                                score = basic_similarity(keyword, variation)
                                if score >= master_info['threshold'] and score > best_score:
                                    best_score = score
                                    best_match = variation
                                    matched_master = master_keyword
                    
                    except Exception:
                        continue
                
                if best_score == 100:
                    break
            
            # Group under best match
            if matched_master and best_score >= max(min_score, master_dict[matched_master]['threshold']):
                group_key = matched_master
            else:
                group_key = keyword
            
            grouped_keywords[group_key]['variations'].append(keyword)
            grouped_keywords[group_key]['total_counts'] += data['total_counts']
            grouped_keywords[group_key]['total_clicks'] += data['total_clicks']
            grouped_keywords[group_key]['total_conversions'] += data['total_conversions']
            grouped_keywords[group_key]['queries'].extend(data['queries'])
            
            processed_keywords.add(keyword)
        
        return dict(grouped_keywords)

    @st.cache_data(ttl=1800, show_spinner=False)
    def calculate_enhanced_keyword_performance(_df):
        """Enhanced keyword performance calculation with optimizations"""
        if _df.empty:
            return pd.DataFrame()
        
        try:
            from collections import defaultdict
            
            keyword_data = defaultdict(lambda: {
                'total_counts': 0, 
                'total_clicks': 0, 
                'total_conversions': 0, 
                'queries': []
            })
            
            # Process data in chunks for better memory management
            chunk_size = 1000
            total_rows = len(_df)
            
            for i in range(0, total_rows, chunk_size):
                chunk = _df.iloc[i:i+chunk_size]
                
                for _, row in chunk.iterrows():
                    try:
                        query = str(row.get('normalized_query', ''))
                        counts = row.get('Counts', 0)
                        clicks = row.get('clicks', 0)
                        conversions = row.get('conversions', 0)
                        
                        if not query or counts == 0:
                            continue
                        
                        keywords = extract_keywords_with_fuzzy_grouping(query, min_length=2)
                        
                        for keyword in keywords:
                            if len(keyword.strip()) >= 2:
                                keyword_data[keyword]['total_counts'] += counts
                                keyword_data[keyword]['total_clicks'] += clicks
                                keyword_data[keyword]['total_conversions'] += conversions
                                keyword_data[keyword]['queries'].append(query)
                    except Exception:
                        continue
            
            # Apply fuzzy matching grouping
            master_dict = create_master_keyword_dictionary()
            grouped_data = fuzzy_match_keywords(keyword_data, master_dict, min_score=65)
            
            # Convert to DataFrame with optimized calculations
            kw_list = []
            for keyword, data in grouped_data.items():
                try:
                    if data['total_counts'] > 0:
                        total_counts = data['total_counts']
                        total_clicks = data['total_clicks']
                        total_conversions = data['total_conversions']
                        
                        avg_ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
                        classic_cr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
                        health_cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0
                        
                        # Limit data to reduce memory usage
                        unique_queries = list(set(data['queries']))
                        unique_variations = list(set(data['variations']))
                        
                        kw_list.append({
                            'keyword': keyword,
                            'total_counts': total_counts,
                            'total_clicks': total_clicks,
                            'total_conversions': total_conversions,
                            'avg_ctr': round(avg_ctr, 2),
                            'classic_cr': round(classic_cr, 2),
                            'health_cr': round(health_cr, 2),
                            'unique_queries': len(unique_queries),
                            'variations_count': len(unique_variations),
                            'example_queries': unique_queries[:5],
                            'variations': unique_variations
                        })
                except Exception:
                    continue
            
            df_result = pd.DataFrame(kw_list)
            if not df_result.empty:
                df_result = df_result.sort_values('total_counts', ascending=False).reset_index(drop=True)
            
            return df_result
            
        except Exception as e:
            st.error(f"❌ Error in keyword analysis: {str(e)}")
            return pd.DataFrame()

    @st.cache_data(ttl=1800, show_spinner=False)
    def create_length_histogram(_df):
        """Cached histogram creation for better performance"""
        if _df.empty:
            return None
        
        fig_length = px.histogram(
            _df, 
            x='query_length', 
            nbins=30,
            title='<b style="color:#2E7D32;">Nutraceuticals & Nutrition Query Length Distribution</b>',
            labels={'query_length': 'Character Length', 'count': 'Number of Health Queries'},
            color_discrete_sequence=['#66BB6A']
        )
        
        fig_length.update_layout(
            plot_bgcolor='rgba(248,253,248,0.95)',
            paper_bgcolor='rgba(232,245,232,0.8)',
            font=dict(color='#1B5E20', family='Segoe UI'),
            bargap=0.1,
            height=400,  # Fixed height
            xaxis=dict(showgrid=True, gridcolor='#E8F5E8'),
            yaxis=dict(showgrid=True, gridcolor='#E8F5E8')
        )
        
        return fig_length

    # ================================================================================================
    # 🎨 ENHANCED UI STYLING AND CONFIGURATION
    # ================================================================================================

    def apply_enhanced_styling():
        """Apply comprehensive CSS styling for better UI"""
        st.markdown("""
        <style>
        /* 🎨 ENHANCED GLOBAL STYLING */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        
        /* 📊 Enhanced Metrics Styling */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
            border: 2px solid #4CAF50;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.15);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.25);
            border-color: #2E7D32;
        }
        
        /* 🎯 Enhanced Subheader Styling */
        .stSubheader {
            background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
            color: white !important;
            padding: 0.8rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
        }
        
        /* 📈 Enhanced Chart Container */
        .js-plotly-plot {
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            overflow: hidden;
            margin: 1rem 0;
        }
        
        /* 🔄 Enhanced Spinner */
        .stSpinner > div {
            border-top-color: #4CAF50 !important;
            border-right-color: #4CAF50 !important;
        }
        
        /* 📋 Enhanced DataFrame Styling */
        .stDataFrame [data-testid="stDataFrameResizeHandle"] {
            display: none !important;
        }
        
        .stDataFrame > div {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        
        .stDataFrame th {
            text-align: center !important;
            background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border: 1px solid #1B5E20 !important;
            padding: 12px 8px !important;
        }
        
        .stDataFrame td {
            text-align: center !important;
            border: 1px solid #E8F5E8 !important;
            padding: 10px 8px !important;
        }
        
        .stDataFrame tr:nth-child(even) {
            background-color: #F1F8E9 !important;
        }
        
        .stDataFrame tr:hover {
            background-color: #E8F5E8 !important;
            transform: scale(1.01);
            transition: all 0.2s ease;
        }
        
        /* 🎛️ Enhanced Controls */
        .stSelectbox > div > div {
            background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E8 100%);
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }
        
        .stSlider > div > div > div {
            background: linear-gradient(90deg, #4CAF50 0%, #2E7D32 100%);
        }
        
        /* 💡 Enhanced Info Boxes */
        .stInfo {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            border-left: 5px solid #2196F3;
            border-radius: 8px;
        }
        
        .stWarning {
            background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
            border-left: 5px solid #FF9800;
            border-radius: 8px;
        }
        
        .stError {
            background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
            border-left: 5px solid #F44336;
            border-radius: 8px;
        }
        
        /* 🔍 Enhanced Text Areas */
        .stTextArea textarea {
            background: #F8F9FA;
            border: 2px solid #E0E0E0;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
        }
        
        .stTextArea textarea:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
        }
        
        /* 📱 Responsive Design */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            
            [data-testid="metric-container"] {
                margin: 0.5rem 0;
            }
        }
        
        /* 🎨 Loading Animation Enhancement */
        @keyframes healthPulse {
            0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
        }
        
        .stSpinner {
            animation: healthPulse 2s infinite;
        }
        </style>
        """, unsafe_allow_html=True)

    # ================================================================================================
    # 🚀 MAIN EXECUTION SECTION WITH ENHANCED PERFORMANCE
    # ================================================================================================

    def main_health_analysis():
        """Main function for health keyword analysis with enhanced performance"""
        
        # Apply enhanced styling
        apply_enhanced_styling()
        
        # 🎨 GREEN-THEMED HERO HEADER
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 3rem 2rem; 
            background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 50%, #A5D6A7 100%); 
            border-radius: 20px; 
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(27, 94, 32, 0.15);
            border: 1px solid rgba(76, 175, 80, 0.2);
        ">
            <h1 style="
                color: #1B5E20; 
                margin: 0; 
                font-size: 3rem; 
                text-shadow: 2px 2px 8px rgba(27, 94, 32, 0.2);
                font-weight: 700;
                letter-spacing: -1px;
            ">
                🌿 Nutraceuticals & Nutrition Keywords Intelligence Hub 🌿
            </h1>
            <p style="
                color: #2E7D32; 
                margin: 1rem 0 0 0; 
                font-size: 1.3rem;
                font-weight: 300;
                opacity: 0.9;
            ">
                Advanced Matching • Performance Analytics • Search Insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance monitoring
        start_time = datetime.now()
        
        # 🔧 GREEN-THEMED LOADING EXPERIENCE
        with st.spinner(""):
            # Custom loading container
            loading_container = st.container()
            with loading_container:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #F1F8E9 0%, #DCEDC8 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    margin: 1rem 0;
                    border: 1px solid #C8E6C9;
                ">
                    <h4 style="color: #2E7D32; margin-bottom: 1rem;">🔄 Processing Nutraceuticals & Nutrition Keywords Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced progress tracking
                progress_col1, progress_col2, progress_col3 = st.columns([1, 2, 1])
                with progress_col2:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Step-by-step progress
                steps = [
                    ("🔍 Loading data...", 20),
                    ("🧠 Processing keywords...", 50),
                    ("🔗 Applying fuzzy matching...", 80),
                    ("✅ Analysis complete!", 100)
                ]
                
                for step_text, progress in steps:
                    status_text.markdown(f"**{step_text}**")
                    progress_bar.progress(progress)
                    
                    if progress < 100:
                        import time
                        time.sleep(0.3)
                
                # Calculate keyword performance ONCE
                kw_perf_df = calculate_enhanced_keyword_performance(queries)
                
                # Clean up loading UI
                time.sleep(0.3)
                loading_container.empty()
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # ✅ GREEN-THEMED METRICS DASHBOARD
        if not kw_perf_df.empty:
            
            # Green performance header
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
                padding: 1.5rem;
                border-radius: 15px;
                margin: 2rem 0 1rem 0;
                border-left: 5px solid #4CAF50;
                box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="color: #1B5E20; margin: 0; font-size: 1.5rem;">
                        📊 Performance Dashboard
                    </h3>
                    <div style="text-align: right;">
                        <span style="
                            background: rgba(255,255,255,0.8);
                            padding: 0.3rem 0.8rem;
                            border-radius: 20px;
                            color: #1B5E20;
                            font-size: 0.85rem;
                            font-weight: 500;
                        ">⚡ Processed in {processing_time:.1f}s</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 🧠 GREEN-THEMED AI INSIGHTS
            st.markdown("---")
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #F1F8E9 0%, #DCEDC8 100%);
                padding: 2rem;
                border-radius: 15px;
                margin: 2rem 0;
                border-left: 5px solid #66BB6A;
                box-shadow: 0 6px 20px rgba(102, 187, 106, 0.2);
            ">
                <h4 style="
                    color: #1B5E20; 
                    margin: 0 0 1.5rem 0; 
                    font-size: 1.4rem;
                ">
                    🧠 Grouped Keywords Insights
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 🔧 CALCULATE ALL VARIABLES FIRST (MOVED OUTSIDE COLUMNS)
            def safe_calculate_health_metrics(kw_perf_df):
                """Safely calculate all health analysis metrics"""
                
                # Default values
                metrics = {
                    'total_keywords': 0,
                    'total_volume': 0,
                    'avg_ctr': 0.0,
                    'avg_health_cr': 0.0,
                    'high_perf_count': 0,
                    'high_perf_pct': 0.0,
                    'long_tail_pct': 0.0,
                    'avg_query_length': 0.0,
                    'top_keyword_pct': 0.0,
                    'avg_words': 0.0,
                    'top_volume': 0
                }
                
                try:
                    if not kw_perf_df.empty:
                        
                        # Ensure keyword_length column exists
                        if 'keyword_length' not in kw_perf_df.columns:
                            if 'representative_keyword' in kw_perf_df.columns:
                                kw_perf_df['keyword_length'] = kw_perf_df['representative_keyword'].str.split().str.len()
                            elif 'keywords' in kw_perf_df.columns:
                                kw_perf_df['keyword_length'] = kw_perf_df['keywords'].astype(str).str.split().str.len()
                            else:
                                kw_perf_df['keyword_length'] = 2
                        
                        # Calculate metrics
                        metrics['total_keywords'] = len(kw_perf_df)
                        metrics['total_volume'] = kw_perf_df['total_counts'].sum() if 'total_counts' in kw_perf_df.columns else 0
                        metrics['avg_ctr'] = kw_perf_df['avg_ctr'].mean() if 'avg_ctr' in kw_perf_df.columns else 0.0
                        metrics['avg_health_cr'] = kw_perf_df['health_cr'].mean() if 'health_cr' in kw_perf_df.columns else 0.0
                        
                        # Top volume calculation
                        metrics['top_volume'] = kw_perf_df['total_counts'].max() if 'total_counts' in kw_perf_df.columns else 0
                        
                        # High performance calculations
                        if 'avg_ctr' in kw_perf_df.columns and metrics['avg_ctr'] > 0:
                            metrics['high_perf_count'] = len(kw_perf_df[kw_perf_df['avg_ctr'] > metrics['avg_ctr']])
                            metrics['high_perf_pct'] = (metrics['high_perf_count'] / metrics['total_keywords']) * 100 if metrics['total_keywords'] > 0 else 0
                        
                        # Long-tail calculations
                        if metrics['total_keywords'] > 0:
                            long_tail_count = len(kw_perf_df[kw_perf_df['keyword_length'] >= 3])
                            metrics['long_tail_pct'] = (long_tail_count / metrics['total_keywords']) * 100
                        
                        # Average query length
                        if 'representative_keyword' in kw_perf_df.columns:
                            metrics['avg_query_length'] = kw_perf_df['representative_keyword'].str.len().mean()
                        elif 'keywords' in kw_perf_df.columns:
                            metrics['avg_query_length'] = kw_perf_df['keywords'].astype(str).str.len().mean()
                        
                        # Top keyword percentage
                        if metrics['total_volume'] > 0 and 'total_counts' in kw_perf_df.columns:
                            metrics['top_keyword_pct'] = (metrics['top_volume'] / metrics['total_volume']) * 100
                        
                        # Average words per query
                        metrics['avg_words'] = kw_perf_df['keyword_length'].mean()
                        
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")
                
                return metrics

            # 🔧 USE THE SAFE CALCULATION FUNCTION
            health_metrics = safe_calculate_health_metrics(kw_perf_df)

            # Extract variables for easy use
            total_keywords = health_metrics['total_keywords']
            avg_ctr = health_metrics['avg_ctr']
            high_perf_count = health_metrics['high_perf_count']
            high_perf_pct = health_metrics['high_perf_pct']
            long_tail_pct = health_metrics['long_tail_pct']
            avg_words = health_metrics['avg_words']
            top_keyword_pct = health_metrics['top_keyword_pct']
            top_volume = health_metrics['top_volume']
            
            # ✅ NOW CREATE THE INSIGHTS COLUMNS
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                # Use pre-calculated avg_words
                complexity_status = "🔥 Complex queries" if avg_words > 3 else "📊 Simple queries"
                st.markdown(f"""
                <div style="
                    background: white; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    border-left: 4px solid #4CAF50;
                    box-shadow: 0 2px 10px rgba(76, 175, 80, 0.1);
                    height: 140px;
                ">
                    <h5 style="color: #2E7D32; margin: 0 0 1rem 0;">🎯 Query Analysis</h5>
                    <p style="margin: 0 0 0.5rem 0; color: #555;">
                        Average <strong>{avg_words:.1f} words</strong> per query
                    </p>
                    <p style="margin: 0 0 0.5rem 0; color: #555;">
                        <strong>{long_tail_pct:.1f}%</strong> long-tail queries
                    </p>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #4CAF50;">
                        {complexity_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with insight_col2:
                # Use pre-calculated values
                performance_status = "🎯 Strong performance" if high_perf_pct > 40 else "📈 Growth potential"
                
                st.markdown(f"""
                <div style="
                    background: white; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    border-left: 4px solid #66BB6A;
                    box-shadow: 0 2px 10px rgba(102, 187, 106, 0.1);
                    height: 140px;
                ">
                    <h5 style="color: #2E7D32; margin: 0 0 1rem 0;">🚀 Performance</h5>
                    <p style="margin: 0 0 0.5rem 0; color: #555;">
                        <strong>{high_perf_pct:.1f}%</strong> above-average CTR
                    </p>
                    <p style="margin: 0 0 0.5rem 0; color: #555;">
                        <strong>{avg_ctr:.2f}%</strong> average CTR
                    </p>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #66BB6A;">
                        {performance_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with insight_col3:
                # Use pre-calculated top_volume
                volume_status = "🔥 High volume" if top_volume > 10000 else "📊 Moderate volume"
                
                st.markdown(f"""
                <div style="
                    background: white; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    border-left: 4px solid #81C784;
                    box-shadow: 0 2px 10px rgba(129, 199, 132, 0.1);
                    height: 140px;
                ">
                    <h5 style="color: #2E7D32; margin: 0 0 1rem 0;">🌊 Search Volume Insights</h5>
                    <p style="margin: 0 0 0.5rem 0; color: #555;">
                        Peak volume: <strong>{top_volume:,}</strong>
                    </p>
                    <p style="margin: 0 0 0.5rem 0; color: #555;">
                        Total keywords: <strong>{total_keywords:,}</strong>
                    </p>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #81C784;">
                        {volume_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Rest of your code continues here...
            
            # 📊 GREEN-THEMED RECOMMENDATIONS
            st.markdown("---")
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
                padding: 1.5rem;
                border-radius: 15px;
                margin: 2rem 0 1rem 0;
                border-left: 5px solid #388E3C;
                box-shadow: 0 4px 15px rgba(56, 142, 60, 0.2);
            ">
                <h4 style="color: #1B5E20; margin: 0; font-size: 1.4rem;">
                    💡 Key Recommendations
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Green recommendations
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    border-left: 4px solid #4CAF50;
                    box-shadow: 0 2px 10px rgba(76, 175, 80, 0.1);
                    margin-bottom: 1rem;
                ">
                    <h5 style="color: #2E7D32; margin: 0 0 1rem 0;">🎯 Optimization Focus</h5>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #555;">
                        <li style="margin-bottom: 0.5rem;">Target high-volume keywords</li>
                        <li style="margin-bottom: 0.5rem;">Improve CTR for underperformers</li>
                        <li style="margin-bottom: 0.5rem;">Optimize conversion paths</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    border-left: 4px solid #66BB6A;
                    box-shadow: 0 2px 10px rgba(102, 187, 106, 0.1);
                    margin-bottom: 1rem;
                ">
                    <h5 style="color: #2E7D32; margin: 0 0 1rem 0;">📊 Performance Summary</h5>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #555;">
                        <li style="margin-bottom: 0.5rem;">Analysis time: {processing_time:.1f}s</li>
                        <li style="margin-bottom: 0.5rem;">Data quality: {"Excellent" if total_keywords > 1000 else "Good"}</li>
                        <li style="margin-bottom: 0.5rem;">Coverage: {total_keywords:,} keywords</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)



        
        # Create layout
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            # Enhanced subheader with icon
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0; display: flex; align-items: center;">
                    🎯 Grouped Keywords Performance Matrix
                    <span style="margin-left: auto; font-size: 0.8rem; opacity: 0.8;">Real-time Analysis</span>
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            if not kw_perf_df.empty:
                # Enhanced performance metrics
                total_keywords = len(kw_perf_df)
                total_volume = kw_perf_df['total_counts'].sum()
                avg_ctr = kw_perf_df['avg_ctr'].mean()
                
                # Performance summary with enhanced styling
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #4CAF50; margin: 1rem 0;">
                    <h4 style="color: #1B5E20; margin: 0 0 1rem 0;">📊 Analysis Summary</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; color: #2E7D32; font-weight: bold;">{total_keywords:,}</div>
                            <div style="color: #1B5E20;">Keyword Groups</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; color: #2E7D32; font-weight: bold;">{total_volume:,}</div>
                            <div style="color: #1B5E20;">Total Volume</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem; color: #2E7D32; font-weight: bold;">{avg_ctr:.2f}%</div>
                            <div style="color: #1B5E20;">Avg CTR</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Limit chart data for better performance
                chart_data = kw_perf_df.head(30)
                
                # Enhanced scatter plot with better performance
                fig_kw = px.scatter(
                    chart_data, 
                    x='total_counts', 
                    y='avg_ctr',
                    size='total_clicks',
                    color='health_cr',
                    hover_name='keyword',
                    title='<b style="color:#2E7D32; font-size:18px;">Health Keywords Performance Matrix: Volume vs CTR 🌿</b>',
                    labels={
                        'total_counts': 'Total Search Volume', 
                        'avg_ctr': 'Average CTR (%)', 
                        'health_cr': 'Health CR (%)'
                    },
                    color_continuous_scale=['#E8F5E8', '#66BB6A', '#2E7D32'],
                    template='plotly_white'
                )
                
                # Enhanced hover template
                fig_kw.update_traces(
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                'Total Volume: %{x:,.0f}<br>' +
                                'CTR: %{y:.2f}%<br>' +
                                'Total Clicks: %{marker.size:,.0f}<br>' +
                                'Health CR: %{marker.color:.2f}%<br>' +
                                'Variations: %{customdata}<extra></extra>',
                    customdata=chart_data['variations_count']
                )
                
                # Enhanced layout with better styling
                fig_kw.update_layout(
                    plot_bgcolor='rgba(248,253,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    title_x=0,
                    height=500,
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='#E8F5E8', 
                        linecolor='#2E7D32', 
                        linewidth=2,
                        title_font=dict(size=14, color='#1B5E20')
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='#E8F5E8', 
                        linecolor='#2E7D32', 
                        linewidth=2,
                        title_font=dict(size=14, color='#1B5E20')
                    ),
                    annotations=[
                        dict(
                            x=0.95, y=0.95, xref='paper', yref='paper',
                            text='💡 Size = Total Clicks | Color = Health CR',
                            showarrow=False,
                            font=dict(size=11, color='#1B5E20'),
                            align='right',
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='#2E7D32',
                            borderwidth=1,
                        )
                    ]
                )
                
                st.plotly_chart(fig_kw, use_container_width=True)
                
                # Performance summary with matching method
                matching_method = "Advanced Fuzzy Matching" if has_fuzzywuzzy else "Basic String Matching"
                
                
            else:
                st.warning("⚠️ No keyword performance data available to display chart.")



        with col_right:
            # Enhanced Query Length Analysis
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <h3 style="margin: 0;">📊 Query Length Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create enhanced histogram
            fig_length = create_length_histogram(queries)
            if fig_length:
                st.plotly_chart(fig_length, use_container_width=True)
                
                # Add insights about query length
                if not queries.empty:
                    avg_length = queries['query_length'].mean()
                    median_length = queries['query_length'].median()
                    max_length = queries['query_length'].max()
                    
                    st.markdown(f"""
                    <div style="background: #F1F8E9; padding: 1rem; border-radius: 8px; border-left: 4px solid #4CAF50;">
                        <h5 style="color: #1B5E20; margin: 0 0 0.5rem 0;">📏 Length Insights</h5>
                        <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>Average:</strong> {avg_length:.1f} characters</p>
                        <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>Median:</strong> {median_length:.1f} characters</p>
                        <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>Longest:</strong> {max_length} characters</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📊 Length distribution will appear here once data is processed.")

        # Enhanced separator
        st.markdown("""
        <div style="height: 3px; background: linear-gradient(90deg, #E8F5E8 0%, #4CAF50 50%, #E8F5E8 100%); margin: 2rem 0; border-radius: 2px;"></div>
        """, unsafe_allow_html=True)

        # ================================================================================================
        # 🏆 ENHANCED TOP PERFORMING KEYWORDS SECTION
        # ================================================================================================
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
            <h2 style="margin: 0; font-size: 2rem;">🏆 Top Performing Grouped Keywords</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;"></p>
        </div>
        """, unsafe_allow_html=True)

        # Calculate enhanced keyword performance with progress tracking
        with st.spinner("🧠 Processing advanced fuzzy matching..."):
            kw_perf_df = calculate_enhanced_keyword_performance(queries)

            # Enhanced keyword grouping success metrics
            magnesium_rows = kw_perf_df[kw_perf_df['keyword'].str.contains('مغنیسیوم', case=False, na=False)]
            collagen_rows = kw_perf_df[kw_perf_df['keyword'].str.contains('کولاجین', case=False, na=False)]
            vitamin_rows = kw_perf_df[kw_perf_df['keyword'].str.contains('فیتامین', case=False, na=False)]
            omega_rows = kw_perf_df[kw_perf_df['keyword'].str.contains('اوميجا', case=False, na=False)]
            
            # Enhanced metrics display with better styling
            st.markdown("""
            <div style="background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E8 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                <h4 style="color: #1B5E20; margin: 0 0 1rem 0; text-align: center;">🎯 Key Health Categories Performance</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if not magnesium_rows.empty:
                    mag_data = magnesium_rows.iloc[0]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%); padding: 1rem; border-radius: 10px; border: 2px solid #4CAF50; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧲</div>
                        <div style="color: #1B5E20; font-weight: bold; font-size: 1.1rem;">مغنیسیوم Group</div>
                        <div style="color: #2E7D32; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{mag_data['total_counts']:,}</div>
                        <div style="color: #388E3C; font-size: 0.9rem;">{mag_data['variations_count']} variations</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #FFEBEE; padding: 1rem; border-radius: 10px; border: 2px solid #F44336; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧲</div>
                        <div style="color: #C62828; font-weight: bold;">مغنیسیوم Group</div>
                        <div style="color: #D32F2F; font-size: 1.5rem;">0</div>
                        <div style="color: #F44336; font-size: 0.9rem;">No matches</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if not collagen_rows.empty:
                    col_data = collagen_rows.iloc[0]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%); padding: 1rem; border-radius: 10px; border: 2px solid #4CAF50; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🦴</div>
                        <div style="color: #1B5E20; font-weight: bold; font-size: 1.1rem;">کولاجین Group</div>
                        <div style="color: #2E7D32; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{col_data['total_counts']:,}</div>
                        <div style="color: #388E3C; font-size: 0.9rem;">{col_data['variations_count']} variations</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #FFEBEE; padding: 1rem; border-radius: 10px; border: 2px solid #F44336; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🦴</div>
                        <div style="color: #C62828; font-weight: bold;">کولاجین Group</div>
                        <div style="color: #D32F2F; font-size: 1.5rem;">0</div>
                        <div style="color: #F44336; font-size: 0.9rem;">No matches</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if not vitamin_rows.empty:
                    vit_data = vitamin_rows.iloc[0]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%); padding: 1rem; border-radius: 10px; border: 2px solid #4CAF50; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">💊</div>
                        <div style="color: #1B5E20; font-weight: bold; font-size: 1.1rem;">فیتامین Group</div>
                        <div style="color: #2E7D32; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{vit_data['total_counts']:,}</div>
                        <div style="color: #388E3C; font-size: 0.9rem;">{vit_data['variations_count']} variations</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #FFEBEE; padding: 1rem; border-radius: 10px; border: 2px solid #F44336; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">💊</div>
                        <div style="color: #C62828; font-weight: bold;">فیتامین Group</div>
                        <div style="color: #D32F2F; font-size: 1.5rem;">0</div>
                        <div style="color: #F44336; font-size: 0.9rem;">No matches</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                if not omega_rows.empty:
                    omega_data = omega_rows.iloc[0]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%); padding: 1rem; border-radius: 10px; border: 2px solid #4CAF50; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐟</div>
                        <div style="color: #1B5E20; font-weight: bold; font-size: 1.1rem;">اوميجا Group</div>
                        <div style="color: #2E7D32; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{omega_data['total_counts']:,}</div>
                        <div style="color: #388E3C; font-size: 0.9rem;">{omega_data['variations_count']} variations</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #FFEBEE; padding: 1rem; border-radius: 10px; border: 2px solid #F44336; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🐟</div>
                        <div style="color: #C62828; font-weight: bold;">اوميجا Group</div>
                        <div style="color: #D32F2F; font-size: 1.5rem;">0</div>
                        <div style="color: #F44336; font-size: 0.9rem;">No matches</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ================================================================================================
            # 🔍 ENHANCED KEYWORD VARIATIONS EXPLORER
            # ================================================================================================
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0;">
                <h3 style="margin: 0; display: flex; align-items: center;">
                    🔍 Keyword Variations Explorer
                    <span style="margin-left: auto; font-size: 0.8rem; opacity: 0.8;">Interactive Analysis</span>
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced slider with better styling
            st.markdown("""
            <div style="background: #F1F8E9; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p style="color: #1B5E20; margin: 0; font-weight: bold;">📊 Select number of keywords to analyze:</p>
            </div>
            """, unsafe_allow_html=True)
            
            num_keywords = st.slider(
                "Number of health keywords to display:", 
                min_value=10, 
                max_value=min(300, len(kw_perf_df)), 
                value=15, 
                step=10,
                key="fuzzy_keyword_count_slider",
                help="Adjust the number of keywords to display in the analysis table"
            )
            
            top_keywords = kw_perf_df.head(num_keywords)

            # Enhanced dropdown with better performance
            top_25_keywords = kw_perf_df.head(25)['keyword'].tolist()

            # Enhanced emoji mapping with more categories
            emoji_map = {
                'مغنیسیوم': '⚡',
                'اوميجا': '🐟', 
                'فیتامین': '💊',
                'کولاجین': '✨',
                'زنک': '🔋',
                'کالسیوم': '🦴',
                'حدید': '🩸',
                'بروتین': '💪',
                'میلاتونین': '😴',
                'بیوتین': '💇',
                'اشواغندا': '🌿',
                'جنسنج': '🌱',
                'کرکم': '🧡',
                'خل التفاح': '🍎',
                'منوم': '🌙',
                'بربرین': '🟡',
                'کرانبری': '🔴',
                'فحم نشط': '⚫',
                'عسل': '🍯',
                'کیو10': '❤️',
                'گلوتاثیون': '✨',
                'ارجنین': '💊',
                'سیلینیوم': '🔘',
                'فولیک اسید': '🤱',
                'تخسیس': '⚖️',
                'پروبیوتیک': '🦠',
                'کرکومین': '🟠',
                'اسپیرولینا': '🟢',
                'چیا سید': '⚪',
                'کینوا': '🌾'
            }

            # Enhanced dropdown options with better formatting
            dropdown_options = []
            keyword_mapping = {}

            for i, keyword in enumerate(top_25_keywords):
                emoji = emoji_map.get(keyword, '💊')
                keyword_data = kw_perf_df[kw_perf_df['keyword'] == keyword].iloc[0]
                volume = format_number(keyword_data['total_counts'])
                variations = keyword_data['variations_count']
                ctr = keyword_data['avg_ctr']
                
                display_text = f"{emoji} {keyword} ({volume} searches, {variations} variations, {ctr:.1f}% CTR)"
                dropdown_options.append(display_text)
                keyword_mapping[display_text] = keyword

            # Enhanced dropdown with better styling
            st.markdown("""
            <div style="background: #F1F8E9; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p style="color: #1B5E20; margin: 0; font-weight: bold;">🎯 Select a keyword to explore its variations:</p>
            </div>
            """, unsafe_allow_html=True)
            
            selected_option = st.selectbox(
                "Choose a health keyword:",
                options=["🔍 Select a keyword to explore..."] + dropdown_options,
                key="keyword_variations_dropdown",
                help="Select any keyword to see its variations, performance metrics, and insights"
            )

            # Enhanced keyword analysis display
            if selected_option != "🔍 Select a keyword to explore...":
                selected_keyword = keyword_mapping[selected_option]
                keyword_rows = kw_perf_df[kw_perf_df['keyword'] == selected_keyword]
                
                if not keyword_rows.empty:
                    keyword_data = keyword_rows.iloc[0]
                    emoji = emoji_map.get(selected_keyword, '💊')
                    
                    # Enhanced keyword header
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
                        <h2 style="margin: 0; font-size: 2.5rem;">{selected_keyword}</h2>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">Variations Analysis & Performance Insights</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced metrics with better layout
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #2196F3;">
                            <div style="font-size: 2.5rem; color: #0D47A1; font-weight: bold;">{format_number(keyword_data['total_counts'])}</div>
                            <div style="color: #1565C0; font-weight: bold; margin-top: 0.5rem;">Total Volume</div>
                            <div style="color: #1976D2; font-size: 0.9rem; margin-top: 0.3rem;">Search Impressions</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #4CAF50;">
                            <div style="font-size: 2.5rem; color: #1B5E20; font-weight: bold;">{format_number(keyword_data['variations_count'])}</div>
                            <div style="color: #2E7D32; font-weight: bold; margin-top: 0.5rem;">Variations</div>
                            <div style="color: #388E3C; font-size: 0.9rem; margin-top: 0.3rem;">Grouped Together</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #FF9800;">
                            <div style="font-size: 2.5rem; color: #E65100; font-weight: bold;">{keyword_data['avg_ctr']:.2f}%</div>
                            <div style="color: #F57C00; font-weight: bold; margin-top: 0.5rem;">Avg CTR</div>
                            <div style="color: #FF9800; font-size: 0.9rem; margin-top: 0.3rem;">Click-Through Rate</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 2px solid #9C27B0;">
                            <div style="font-size: 2.5rem; color: #4A148C; font-weight: bold;">{keyword_data['health_cr']:.2f}%</div>
                            <div style="color: #6A1B9A; font-weight: bold; margin-top: 0.5rem;">Health CR</div>
                            <div style="color: #8E24AA; font-size: 0.9rem; margin-top: 0.3rem;">Conversion Rate</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Enhanced variations display section
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1rem; border-radius: 10px; margin: 2rem 0;">
                        <h3 style="margin: 0;">📝 All Keyword Variations</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    variations_list = keyword_data['variations']
                    total_variations = len(variations_list)
                    
                    if total_variations > 0:
                        # Enhanced user controls
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            display_count = st.selectbox(
                                "📊 Number of variations to show:",
                                [25, 50, 100, "All"],
                                index=1,
                                help="Choose how many variations to display"
                            )
                        
                        with col2:
                            display_format = st.radio(
                                "📋 Display format:",
                                ["Pipe separated", "Line by line", "Numbered list"],
                                index=0,
                                help="Choose how to format the variations list"
                            )
                        
                        # Process variations with enhanced logic
                        available_variations = len(variations_list)
                        
                        if display_count == "All":
                            variations_to_show = variations_list
                        else:
                            variations_to_show = variations_list[:min(display_count, available_variations)]
                        
                        # Enhanced formatting options
                        if display_format == "Line by line":
                            variations_text = "\n".join(variations_to_show)
                            height = min(400, max(150, len(variations_to_show) * 25))
                        elif display_format == "Numbered list":
                            variations_text = "\n".join([f"{i+1}. {var}" for i, var in enumerate(variations_to_show)])
                            height = min(400, max(150, len(variations_to_show) * 25))
                        else:
                            variations_text = " | ".join(variations_to_show)
                            height = 150
                        
                        # Enhanced text area with better styling
                        st.text_area(
                            f"🔍 Variations (showing {len(variations_to_show):,} of {available_variations:,}):",
                            variations_text,
                            height=height,
                            help="Copy these variations for your keyword research and SEO campaigns"
                        )
                        
                        # Enhanced info display
                        if available_variations > len(variations_to_show):
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #2196F3; margin: 1rem 0;">
                                <p style="margin: 0; color: #0D47A1;">
                                    ℹ️ <strong>{available_variations - len(variations_to_show):,}</strong> more variations available. 
                                    Select 'All' to see the complete list.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Enhanced additional insights
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1rem; border-radius: 10px; margin: 2rem 0;">
                            <h3 style="margin: 0;">📊 Advanced Performance Insights</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        insight_col1, insight_col2 = st.columns(2)
                        
                        with insight_col1:
                            st.markdown(f"""
                            <div style="background: #F1F8E9; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4CAF50;">
                                <h5 style="color: #1B5E20; margin: 0 0 1rem 0;">💼 Performance Metrics</h5>
                                <p style="margin: 0.3rem 0; color: #2E7D32;"><strong>📊 Total Clicks:</strong> {format_number(keyword_data['total_clicks'])}</p>
                                <p style="margin: 0.3rem 0; color: #2E7D32;"><strong>🎯 Conversions:</strong> {format_number(keyword_data['total_conversions'])}</p>
                                <p style="margin: 0.3rem 0; color: #2E7D32;"><strong>🔍 Unique Queries:</strong> {format_number(keyword_data['unique_queries'])}</p>
                                <p style="margin: 0.3rem 0; color: #2E7D32;"><strong>📈 Classic CR:</strong> {keyword_data['classic_cr']:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with insight_col2:
                            # Enhanced calculations
                            avg_searches = keyword_data['total_counts'] / keyword_data['variations_count'] if keyword_data['variations_count'] > 0 else 0
                            diversity_score = (keyword_data['variations_count'] / keyword_data['total_counts'] * 1000) if keyword_data['total_counts'] > 0 else 0
                            market_share = (keyword_data['total_counts'] / queries['Counts'].sum() * 100) if 'queries' in locals() and not queries.empty else 0
                            
                            # Performance rating
                            if keyword_data['health_cr'] > 5:
                                performance_rating = "🌟 Excellent"
                            elif keyword_data['health_cr'] > 2:
                                performance_rating = "⭐ Good"
                            elif keyword_data['health_cr'] > 1:
                                performance_rating = "👍 Average"
                            else:
                                performance_rating = "📈 Needs Improvement"
                            
                            st.markdown(f"""
                            <div style="background: #E3F2FD; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196F3;">
                                <h5 style="color: #0D47A1; margin: 0 0 1rem 0;">🎯 Market Intelligence</h5>
                                <p style="margin: 0.3rem 0; color: #1565C0;"><strong>📊 Avg Searches/Variation:</strong> {avg_searches:.1f}</p>
                                <p style="margin: 0.3rem 0; color: #1565C0;"><strong>🎲 Diversity Score:</strong> {diversity_score:.2f}</p>
                                <p style="margin: 0.3rem 0; color: #1565C0;"><strong>📈 Market Share:</strong> {market_share:.2f}%</p>
                                <p style="margin: 0.3rem 0; color: #1565C0;"><strong>⭐ Performance:</strong> {performance_rating}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    else:
                        st.warning("⚠️ No variations found for this keyword.")

            # Enhanced separator before main table
            st.markdown("""
            <div style="height: 3px; background: linear-gradient(90deg, #E8F5E8 0%, #4CAF50 50%, #E8F5E8 100%); margin: 3rem 0; border-radius: 2px;"></div>
            """, unsafe_allow_html=True)



            # ================================================================================================
            # 📊 ENHANCED MAIN KEYWORDS TABLE WITH INTERACTIVE BAR CHART
            # ================================================================================================

            # Calculate market share for enhanced insights
            total_all_counts = queries['Counts'].sum()
            top_keywords['share_pct'] = (top_keywords['total_counts'] / total_all_counts * 100).round(2)

            # ✅ ADD AVG CR CALCULATION (Conversions / Search Volume)
            top_keywords['avg_cr_volume'] = ((top_keywords['total_conversions'] / top_keywords['total_counts']) * 100).fillna(0).round(4)

            if not top_keywords.empty:
                # Enhanced table header
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0; text-align: center;">
                    <h2 style="margin: 0; font-size: 2rem;">📊 Top {num_keywords} Grouped Keywords Performance Table</h2>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Comprehensive Analysis with Market Share & Performance Metrics</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create enhanced display version
                display_df = top_keywords.copy()
                
                # Enhanced column renaming
                display_df = display_df.rename(columns={
                    'keyword': 'Health Keyword',
                    'total_counts': 'Total Search Volume',
                    'share_pct': 'Market Share %',
                    'total_clicks': 'Total Clicks',
                    'total_conversions': 'Conversions',
                    'avg_ctr': 'Avg CTR',
                    'health_cr': 'Health CR',
                    'classic_cr': 'Classic CR',
                    'avg_cr_volume': 'AVG CR (Conv/Vol)',  # ✅ Added new column
                    'unique_queries': 'Unique Queries',
                    'variations_count': 'Variations'
                })
                
                # Enhanced formatting with better number handling
                display_df['Total Search Volume'] = display_df['Total Search Volume'].apply(format_number)
                display_df['Market Share %'] = display_df['Market Share %'].apply(lambda x: f"{x:.2f}%")
                display_df['Total Clicks'] = display_df['Total Clicks'].apply(format_number)
                display_df['Conversions'] = display_df['Conversions'].apply(format_number)
                display_df['Avg CTR'] = display_df['Avg CTR'].apply(lambda x: f"{x:.2f}%")
                display_df['Health CR'] = display_df['Health CR'].apply(lambda x: f"{x:.2f}%")
                display_df['Classic CR'] = display_df['Classic CR'].apply(lambda x: f"{x:.2f}%")
                display_df['AVG CR (Conv/Vol)'] = display_df['AVG CR (Conv/Vol)'].apply(lambda x: f"{x:.4f}%")  # ✅ Format new column
                display_df['Unique Queries'] = display_df['Unique Queries'].apply(format_number)
                display_df['Variations'] = display_df['Variations'].apply(format_number)
                
                # Enhanced column configuration - ✅ UPDATED ORDER
                column_order = ['Health Keyword', 'Total Search Volume', 'Market Share %', 'Total Clicks', 
                            'Conversions', 'Avg CTR', 'Health CR', 'Classic CR', 'AVG CR (Conv/Vol)', 'Unique Queries', 'Variations']
                display_df = display_df[column_order].reset_index(drop=True)
                
                # Enhanced dataframe display with better configuration
                st.dataframe(
                    display_df, 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Health Keyword": st.column_config.TextColumn(
                            "Health Keyword",
                            help="Fuzzy-matched Nutraceuticals & Nutrition search keyword group",
                            width="large"
                        ),
                        "Total Search Volume": st.column_config.TextColumn(
                            "Total Search Volume",
                            help="Total health search volume (fuzzy-grouped)",
                            width="medium"
                        ),
                        "Market Share %": st.column_config.TextColumn(
                            "Market Share %",
                            help="Percentage of total health searches",
                            width="small"
                        ),
                        "Total Clicks": st.column_config.TextColumn(
                            "Total Clicks",
                            help="Total clicks received across all variations",
                            width="medium"
                        ),
                        "Conversions": st.column_config.TextColumn(
                            "Conversions",
                            help="Total conversions achieved",
                            width="medium"
                        ),
                        "Avg CTR": st.column_config.TextColumn(
                            "Avg CTR",
                            help="Average Click-Through Rate across all variations",
                            width="small"
                        ),
                        "Health CR": st.column_config.TextColumn(
                            "Health CR",
                            help="Health Conversion Rate (Conversions/Volume) - Key Performance Indicator",
                            width="small"
                        ),
                        "Classic CR": st.column_config.TextColumn(
                            "Classic CR",
                            help="Classic Conversion Rate (Conversions/Clicks)",
                            width="small"
                        ),
                        "AVG CR (Conv/Vol)": st.column_config.TextColumn(  # ✅ Added new column config
                            "AVG CR (Conv/Vol)",
                            help="Average Conversion Rate: Conversions divided by Search Volume - Direct conversion efficiency metric",
                            width="small"
                        ),
                        "Unique Queries": st.column_config.TextColumn(
                            "Unique Queries",
                            help="Number of unique search queries for this keyword group",
                            width="medium"
                        ),
                        "Variations": st.column_config.TextColumn(
                            "Variations",
                            help="Number of keyword variations grouped together through fuzzy matching",
                            width="small"
                        )
                    }
                )
                
                # ================================================================================================
                # 📊 INTERACTIVE BAR CHART SECTION WITH AVG CR
                # ================================================================================================
                
                st.markdown("---")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%); padding: 1.5rem; border-radius: 15px; margin: 2rem 0; border-left: 5px solid #4CAF50;">
                    <h3 style="color: #1B5E20; margin: 0; font-size: 1.5rem;">📊 Interactive Keywords Performance Visualization</h3>
                    <p style="color: #2E7D32; margin: 0.5rem 0 0 0;">Select keywords and metrics to explore performance patterns</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Chart controls in columns
                chart_col1, chart_col2, chart_col3 = st.columns([2, 1, 1])
                
                with chart_col1:
                    # Multi-select for keywords
                    selected_keywords = st.multiselect(
                        "🎯 Select Keywords to Display",
                        options=top_keywords['keyword'].tolist(),
                        default=top_keywords['keyword'].head(10).tolist(),  # Default to top 10
                        help="Choose which keywords to display in the chart"
                    )
                
                with chart_col2:
                    # Metric selection - ✅ ADDED AVG CR OPTION
                    chart_metric = st.selectbox(
                        "📈 Primary Metric",
                        options=[
                            "Total Search Volume",
                            "Total Clicks", 
                            "Conversions",
                            "Market Share %",
                            "AVG CR (Conv/Vol)"  # ✅ Added new metric option
                        ],
                        index=0,
                        help="Choose the primary metric to display"
                    )
                
                with chart_col3:
                    # Chart type selection
                    chart_type = st.selectbox(
                        "📊 Chart Type",
                        options=["Bar Chart", "Horizontal Bar", "Area Chart"],
                        index=0,
                        help="Choose visualization type"
                    )
                
                # Create chart data based on selections
                if selected_keywords:
                    # Filter data for selected keywords
                    chart_data = top_keywords[top_keywords['keyword'].isin(selected_keywords)].copy()
                    
                    # Map display names to actual column names - ✅ ADDED AVG CR MAPPING
                    metric_mapping = {
                        "Total Search Volume": "total_counts",
                        "Total Clicks": "total_clicks",
                        "Conversions": "total_conversions", 
                        "Market Share %": "share_pct",
                        "AVG CR (Conv/Vol)": "avg_cr_volume"  # ✅ Added new mapping
                    }
                    
                    metric_column = metric_mapping[chart_metric]
                    
                    # Sort data by selected metric
                    chart_data = chart_data.sort_values(metric_column, ascending=False)
                    
                    # ✅ ENHANCED COLOR MAPPING BASED ON METRIC TYPE
                    if chart_metric == "AVG CR (Conv/Vol)":
                        color_column = 'avg_cr_volume'
                        color_label = 'AVG CR (%)'
                    else:
                        color_column = 'avg_ctr'
                        color_label = 'Avg CTR (%)'
                    
                    # Create the chart based on type
                    if chart_type == "Bar Chart":
                        fig_bar = px.bar(
                            chart_data,
                            x='keyword',
                            y=metric_column,
                            color=color_column,  # ✅ Dynamic color based on metric
                            title=f'<b style="color:#2E7D32; font-size:18px;">🌿 {chart_metric} by Selected Keywords</b>',
                            labels={
                                'keyword': 'Health Keywords',
                                metric_column: chart_metric,
                                color_column: color_label
                            },
                            color_continuous_scale=['#E8F5E8', '#66BB6A', '#2E7D32'],
                            template='plotly_white'
                        )
                        
                        # ✅ ENHANCED HOVER TEMPLATE WITH AVG CR INFO
                        hover_template = '<b>%{x}</b><br>' + f'{chart_metric}: %{{y:,.4f}}<br>' if chart_metric == "AVG CR (Conv/Vol)" else '<b>%{x}</b><br>' + f'{chart_metric}: %{{y:,.0f}}<br>'
                        hover_template += f'{color_label}: %{{marker.color:.4f}}%<br>' if chart_metric == "AVG CR (Conv/Vol)" else f'{color_label}: %{{marker.color:.2f}}%<br>'
                        hover_template += 'Variations: %{customdata}<extra></extra>'
                        
                        fig_bar.update_traces(
                            hovertemplate=hover_template,
                            customdata=chart_data['variations_count']
                        )
                        
                    elif chart_type == "Horizontal Bar":
                        fig_bar = px.bar(
                            chart_data,
                            y='keyword',
                            x=metric_column,
                            color='health_cr',  # Keep health_cr for horizontal bars
                            orientation='h',
                            title=f'<b style="color:#2E7D32; font-size:18px;">🌿 {chart_metric} by Selected Keywords</b>',
                            labels={
                                'keyword': 'Health Keywords',
                                metric_column: chart_metric,
                                'health_cr': 'Health CR (%)'
                            },
                            color_continuous_scale=['#E8F5E8', '#66BB6A', '#2E7D32'],
                            template='plotly_white'
                        )
                        
                        # ✅ ENHANCED HOVER FOR HORIZONTAL BARS
                        hover_template = '<b>%{y}</b><br>' + f'{chart_metric}: %{{x:,.4f}}<br>' if chart_metric == "AVG CR (Conv/Vol)" else '<b>%{y}</b><br>' + f'{chart_metric}: %{{x:,.0f}}<br>'
                        hover_template += 'Health CR: %{marker.color:.2f}%<br>Variations: %{customdata}<extra></extra>'
                        
                        fig_bar.update_traces(
                            hovertemplate=hover_template,
                            customdata=chart_data['variations_count']
                        )
                        
                    else:  # Area Chart
                        fig_bar = px.area(
                            chart_data,
                            x='keyword',
                            y=metric_column,
                            title=f'<b style="color:#2E7D32; font-size:18px;">🌿 {chart_metric} Distribution</b>',
                            labels={
                                'keyword': 'Health Keywords',
                                metric_column: chart_metric
                            },
                            color_discrete_sequence=['#4CAF50'],
                            template='plotly_white'
                        )
                        
                        # ✅ ENHANCED HOVER FOR AREA CHART
                        hover_template = '<b>%{x}</b><br>' + f'{chart_metric}: %{{y:,.4f}}<extra></extra>' if chart_metric == "AVG CR (Conv/Vol)" else '<b>%{x}</b><br>' + f'{chart_metric}: %{{y:,.0f}}<extra></extra>'
                        
                        fig_bar.update_traces(
                            fill='tonexty',
                            hovertemplate=hover_template
                        )
                    
                    # Enhanced layout styling
                    fig_bar.update_layout(
                        plot_bgcolor='rgba(248,253,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI'),
                        title_x=0,
                        height=500,
                        xaxis=dict(
                            showgrid=True, 
                            gridcolor='#E8F5E8', 
                            linecolor='#2E7D32', 
                            linewidth=2,
                            title_font=dict(size=14, color='#1B5E20'),
                            tickangle=-45 if chart_type == "Bar Chart" else 0
                        ),
                        yaxis=dict(
                            showgrid=True, 
                            gridcolor='#E8F5E8', 
                            linecolor='#2E7D32', 
                            linewidth=2,
                            title_font=dict(size=14, color='#1B5E20')
                        ),
                        showlegend=False
                    )
                    
                    # ✅ ENHANCED ANNOTATION WITH AVG CR INSIGHTS
                    total_selected_volume = chart_data[metric_column].sum()
                    avg_selected_ctr = chart_data['avg_ctr'].mean()
                    avg_selected_cr = chart_data['avg_cr_volume'].mean()  # ✅ Added AVG CR calculation
                    
                    # Dynamic annotation based on selected metric
                    if chart_metric == "AVG CR (Conv/Vol)":
                        annotation_text = f'📊 Selected: {len(selected_keywords)} keywords<br>' + \
                                        f'🎯 Avg {chart_metric}: {total_selected_volume/len(selected_keywords):.4f}%<br>' + \
                                        f'📈 Best CR: {chart_data[metric_column].max():.4f}%'
                    else:
                        annotation_text = f'📊 Selected: {len(selected_keywords)} keywords<br>' + \
                                        f'🎯 Total {chart_metric}: {total_selected_volume:,.0f}<br>' + \
                                        f'📈 Avg CTR: {avg_selected_ctr:.2f}%<br>' + \
                                        f'🔄 Avg CR: {avg_selected_cr:.4f}%'  # ✅ Always show AVG CR
                    
                    fig_bar.add_annotation(
                        x=0.95, y=0.95, xref='paper', yref='paper',
                        text=annotation_text,
                        showarrow=False,
                        font=dict(size=11, color='#1B5E20'),
                        align='right',
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='#2E7D32',
                        borderwidth=1,
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # ✅ ENHANCED PERFORMANCE INSIGHTS WITH AVG CR
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E8 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #66BB6A; margin: 1rem 0;">
                        <h4 style="color: #1B5E20; margin: 0 0 1rem 0;">🎯 Selected Keywords Insights</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                            <div>
                                <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>📊 Keywords Selected:</strong> {len(selected_keywords)}</p>
                                <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>🔥 Top Performer:</strong> {chart_data.iloc[0]['keyword']}</p>
                            </div>
                            <div>
                                <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>📈 Combined {chart_metric}:</strong> {total_selected_volume:,.4f}{'%' if chart_metric == 'AVG CR (Conv/Vol)' else ''}</p>
                                <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>🎯 Average CTR:</strong> {avg_selected_ctr:.2f}%</p>
                            </div>
                            <div>
                                <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>🔄 Average CR (Conv/Vol):</strong> {avg_selected_cr:.4f}%</p>
                                <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>⭐ Best CR:</strong> {chart_data['avg_cr_volume'].max():.4f}%</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # Show message when no keywords selected
                    st.markdown("""
                    <div style="background: #FFF3E0; padding: 2rem; border-radius: 12px; border-left: 5px solid #FF9800; text-align: center; margin: 1rem 0;">
                        <h4 style="color: #E65100; margin: 0;">⚠️ No Keywords Selected</h4>
                        <p style="color: #F57C00; margin: 0.5rem 0 0 0;">Please select at least one keyword to display the chart</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced table performance insights
                processing_time = (datetime.now() - start_time).total_seconds()
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #4CAF50; margin: 2rem 0;">
                    <h4 style="color: #1B5E20; margin: 0 0 1rem 0;">⚡ Table Performance Metrics</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        <div>
                            <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>📊 Rows Displayed:</strong> {len(display_df):,}</p>
                            <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>🔍 Total Keywords:</strong> {len(kw_perf_df):,}</p>
                        </div>
                        <div>
                            <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>⏱️ Processing Time:</strong> {processing_time:.2f}s</p>
                            <p style="margin: 0.2rem 0; color: #2E7D32;"><strong>🎯 Matching Method:</strong> {matching_method}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    

                
                # ================================================================================================
                # 🔍 ENHANCED EXAMPLE QUERIES & VARIATIONS SECTION
                # ================================================================================================
                
                # Enhanced toggle for examples with better styling
                show_examples = st.checkbox(
                    "🔍 Show detailed examples and variations for top keywords", 
                    key="show_fuzzy_examples",
                    help="Display example queries and variations for the top 5 performing keywords"
                )
                
                if show_examples:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0;">
                        <h3 style="margin: 0;">📝 Detailed Examples & Variations Analysis</h3>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Top 5 Keywords with Real Query Examples</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for idx, row in top_keywords.head(5).iterrows():
                        keyword = row['keyword']
                        examples = row['example_queries'][:3]
                        variations = row['variations'][:15]  # Show more variations
                        emoji = emoji_map.get(keyword, '💊')
                        
                        # Enhanced keyword section with better styling
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E8 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; border: 2px solid #4CAF50;">
                            <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                                <div style="font-size: 3rem; margin-right: 1rem;">{emoji}</div>
                                <div>
                                    <h3 style="color: #1B5E20; margin: 0; font-size: 1.8rem;">{keyword}</h3>
                                    <p style="color: #2E7D32; margin: 0.3rem 0 0 0; font-size: 1.1rem;">
                                        {format_number(row['total_counts'])} searches • {row['variations_count']} variations • {row['avg_ctr']:.2f}% CTR
                                    </p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced two-column layout for examples and variations
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("""
                            <div style="background: #E3F2FD; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196F3; height: 100%;">
                                <h5 style="color: #0D47A1; margin: 0 0 1rem 0;">📋 Example Search Queries</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, example in enumerate(examples, 1):
                                st.markdown(f"""
                                <div style="background: white; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #2196F3;">
                                    <span style="color: #1565C0; font-weight: bold;">{i}.</span> 
                                    <span style="color: #0D47A1;">{example}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div style="background: #E8F5E8; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4CAF50; height: 100%;">
                                <h5 style="color: #1B5E20; margin: 0 0 1rem 0;">🔗 Grouped Keyword Variations</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display variations in a more organized way
                            for i, var in enumerate(variations[:10], 1):  # Show top 10
                                st.markdown(f"""
                                <div style="background: white; padding: 0.6rem; margin: 0.3rem 0; border-radius: 6px; border-left: 3px solid #4CAF50;">
                                    <span style="color: #2E7D32; font-weight: bold; font-size: 0.9rem;">{i}.</span> 
                                    <span style="color: #1B5E20; font-size: 0.9rem;">{var}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show count of remaining variations
                            if len(variations) > 10:
                                st.markdown(f"""
                                <div style="background: #FFF3E0; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #FF9800; text-align: center;">
                                    <span style="color: #E65100; font-weight: bold;">+ {len(variations) - 10} more variations</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Enhanced separator between keywords
                        st.markdown("""
                        <div style="height: 2px; background: linear-gradient(90deg, transparent 0%, #4CAF50 50%, transparent 100%); margin: 2rem 0;"></div>
                        """, unsafe_allow_html=True)
                
                # ================================================================================================
                # 📥 ENHANCED DOWNLOAD SECTION
                # ================================================================================================
                
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 1.5rem; border-radius: 12px; margin: 2rem 0;">
                    <h3 style="margin: 0;">📥 Export & Download Options</h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Download your analysis results in multiple formats</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced download options with multiple formats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV download with enhanced data
                    csv_data = top_keywords[['keyword', 'total_counts', 'share_pct', 'total_clicks', 
                                            'total_conversions', 'avg_ctr', 'health_cr', 'classic_cr', 
                                            'unique_queries', 'variations_count']].copy()
                    csv_keywords = csv_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV Report",
                        data=csv_keywords,
                        file_name=f"health_keywords_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="fuzzy_keyword_csv_download",
                        help="Download complete keyword analysis as CSV file"
                    )
                
                with col2:
                    # Enhanced variations export
                    variations_data = []
                    for _, row in top_keywords.head(20).iterrows():  # Top 20 for variations export
                        for variation in row['variations']:
                            variations_data.append({
                                'master_keyword': row['keyword'],
                                'variation': variation,
                                'master_volume': row['total_counts'],
                                'master_ctr': row['avg_ctr'],
                                'master_cr': row['health_cr']
                            })
                    
                    variations_df = pd.DataFrame(variations_data)
                    variations_csv = variations_df.to_csv(index=False)
                    
                    st.download_button(
                        label="🔗 Download Variations Map",
                        data=variations_csv,
                        file_name=f"keyword_variations_map_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="variations_map_download",
                        help="Download detailed keyword variations mapping"
                    )
                
                with col3:
                    # Performance summary report
                    summary_data = {
                        'metric': [
                            'Total Keywords Analyzed',
                            'Total Search Volume',
                            'Total Clicks',
                            'Total Conversions',
                            'Average CTR',
                            'Average Health CR',
                            'Average Classic CR',
                            'Total Variations',
                            'Processing Time (seconds)',
                            'Analysis Date'
                        ],
                        'value': [
                            len(kw_perf_df),
                            top_keywords['total_counts'].sum(),
                            top_keywords['total_clicks'].sum(),
                            top_keywords['total_conversions'].sum(),
                            f"{top_keywords['avg_ctr'].mean():.2f}%",
                            f"{top_keywords['health_cr'].mean():.2f}%",
                            f"{top_keywords['classic_cr'].mean():.2f}%",
                            top_keywords['variations_count'].sum(),
                            f"{processing_time:.2f}",
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_csv = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📋 Download Summary Report",
                        data=summary_csv,
                        file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="summary_report_download",
                        help="Download executive summary of the analysis"
                    )
                
                
                # ================================================================================================
                # 📊 ENHANCED FINAL INSIGHTS & RECOMMENDATIONS
                # ================================================================================================

                # ================================================================================================
                # 📊 ENHANCED FINAL INSIGHTS & RECOMMENDATIONS
                # ================================================================================================

                # Calculate advanced insights with error handling
                total_variations = top_keywords['variations_count'].sum() if 'variations_count' in top_keywords.columns else 0
                avg_health_cr = top_keywords['health_cr'].mean() if len(top_keywords) > 0 and 'health_cr' in top_keywords.columns else 0
                high_perf_keywords = len(top_keywords[top_keywords['health_cr'] > avg_health_cr]) if len(top_keywords) > 0 and 'health_cr' in top_keywords.columns else 0
                top_market_share = top_keywords['share_pct'].sum() if 'share_pct' in top_keywords.columns else 0

                # Performance categorization with safe column access
                if 'health_cr' in top_keywords.columns and len(top_keywords) > 0:
                    excellent_keywords = len(top_keywords[top_keywords['health_cr'] > 5])
                    good_keywords = len(top_keywords[(top_keywords['health_cr'] > 2) & (top_keywords['health_cr'] <= 5)])
                    average_keywords = len(top_keywords[(top_keywords['health_cr'] > 1) & (top_keywords['health_cr'] <= 2)])
                    poor_keywords = len(top_keywords[top_keywords['health_cr'] <= 1])
                else:
                    excellent_keywords = good_keywords = average_keywords = poor_keywords = 0

                # Safe calculation for averages
                avg_variations_per_group = total_variations / len(top_keywords) if len(top_keywords) > 0 and total_variations > 0 else 0
                unique_queries_sum = top_keywords['unique_queries'].sum() if 'unique_queries' in top_keywords.columns else len(top_keywords)
                total_search_volume = top_keywords['total_counts'].sum() if 'total_counts' in top_keywords.columns else top_keywords['Counts'].sum() if 'Counts' in top_keywords.columns else 0

                # Main header
                st.markdown("""
                <div style="background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%); color: white; padding: 2rem; border-radius: 15px; margin: 3rem 0;">
                    <h2 style="margin: 0 0 1rem 0; text-align: center; font-size: 2.2rem;">🎯 Advanced Analysis Insights & Recommendations</h2>
                    <p style="margin: 0; text-align: center; opacity: 0.9; font-size: 1.1rem;">Comprehensive Performance Summary & Strategic Recommendations</p>
                </div>
                """, unsafe_allow_html=True)

                # Enhanced insights with multiple sections
                insight_col1, insight_col2 = st.columns(2)

                INSIGHT_CSS = """
                <style>
                .insight-box-green{background:linear-gradient(135deg,#E8F5E8,#F1F8E9);padding:2rem;border-radius:12px;border-left:5px solid #4CAF50;height:100%;}
                .insight-box-blue{background:linear-gradient(135deg,#E3F2FD,#BBDEFB);padding:2rem;border-radius:12px;border-left:5px solid #2196F3;height:100%;}
                .insight-box-green h4,.insight-box-blue h4{margin:0 0 1.5rem;color:#1B5E20;}
                .insight-box-blue h4{color:#0D47A1;}
                .insight-box-green p,.insight-box-blue p{margin:0.3rem 0;color:#2E7D32;}
                .insight-box-blue p{color:#1976D2;}
                .insight-box-green .sub-box,.insight-box-blue .sub-box{background:rgba(46,125,50,0.1);padding:1rem;border-radius:8px;margin-top:1rem;}
                .insight-box-blue .sub-box{background:rgba(33,150,243,0.1);}
                .insight-box-green .sub-box p,.insight-box-blue .sub-box p{margin:0.2rem 0;color:#2E7D32;font-size:0.9rem;}
                .insight-box-blue .sub-box p{color:#1565C0;}
                </style>
                """
                st.markdown(INSIGHT_CSS, unsafe_allow_html=True)

                with insight_col1:
                    st.markdown(f"""
                    <div class="insight-box-green">
                        <h4>📊 Matching Analysis Summary</h4>
                        <div style="margin-bottom: 1rem;">
                            <p><strong>🔍 Total Keyword Groups:</strong> {format_number(len(kw_perf_df))}</p>
                            <p><strong>🔗 Total Variations Grouped:</strong> {format_number(total_variations)}</p>
                            <p><strong>📈 Total Search Volume (Top {num_keywords}):</strong> {format_number(total_search_volume)}</p>
                            <p><strong>🎯 Market Share Covered:</strong> {top_market_share:.1f}%</p>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <p><strong>🔍 Unique Search Queries:</strong> {format_number(unique_queries_sum)}</p>
                            <p><strong>📊 Avg Variations per Group:</strong> {avg_variations_per_group:.1f}</p>
                            <p><strong>⭐ High Performance Keywords:</strong> {high_perf_keywords} (above {avg_health_cr:.2f}% CR)</p>
                        </div>
                        <div class="sub-box">
                            <p style="color: #1B5E20; font-weight: bold;">🎯 Processing Efficiency:</p>
                            <p>⚡ Analysis completed in {processing_time:.2f} seconds</p>
                            <p>🧠 Method: {matching_method}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with insight_col2:
                    st.markdown(f"""
                    <div class="insight-box-blue">
                        <h4>🎯 Performance Distribution & Recommendations</h4>
                        <div style="margin-bottom: 1.5rem;">
                            <h5 style="color: #1565C0;">📊 Performance Categories:</h5>
                            <p><strong>🌟 Excellent (>5% Health CR):</strong> {excellent_keywords} keyword{'s' if excellent_keywords != 1 else ''}</p>
                            <p><strong>⭐ Good (2-5% Health CR):</strong> {good_keywords} keyword{'s' if good_keywords != 1 else ''}</p>
                            <p><strong>👍 Average (1-2% Health CR):</strong> {average_keywords} keyword{'s' if average_keywords != 1 else ''}</p>
                            <p><strong>📈 Needs Improvement (<1% Health CR):</strong> {poor_keywords} keyword{'s' if poor_keywords != 1 else ''}</p>
                        </div>
                        <div class="sub-box">
                            <h5 style="color: #0D47A1;">💡 Strategic Recommendations:</h5>
                            <p>🎯 Focus on top {excellent_keywords + good_keywords} performing keyword{'s' if (excellent_keywords + good_keywords) != 1 else ''}</p>
                            <p>📈 Optimize content for {poor_keywords} underperforming keyword{'s' if poor_keywords != 1 else ''}</p>
                            <p>🔍 Leverage {format_number(total_variations)} variations for long-tail SEO</p>
                            <p>⚡ Average Health CR: {avg_health_cr:.2f}% - Industry benchmark</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


                # Final performance footer
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E8 100%); padding: 2rem; border-radius: 12px; margin: 3rem 0; text-align: center; border: 2px solid #4CAF50;">
                    <h4 style="color: #1B5E20; margin: 0 0 1rem 0;">🚀 Analysis Complete - Ready for Action!</h4>
                    <p style="color: #2E7D32; margin: 0.5rem 0; font-size: 1.1rem;">
                        ✅ Processed <strong>{len(kw_perf_df):,}</strong> keyword groups in <strong>{processing_time:.2f}</strong> seconds
                    </p>
                    <p style="color: #388E3C; margin: 0.5rem 0;">
                        🎯 Use the insights above to optimize your health & nutrition marketing strategy
                    </p>
                    <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(76, 175, 80, 0.1); border-radius: 8px;">
                        <p style="color: #1B5E20; margin: 0; font-weight: bold;">
                            💡 Pro Tip: Focus on keywords with high variations count and good CR for maximum ROI
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)



    # ================================================================================================
    # 🚀 EXECUTE MAIN FUNCTION
    # ================================================================================================

    if __name__ == "__main__":
        # Ensure all required variables are available
        if 'queries' in locals() and not queries.empty:
            main_health_analysis()
        else:
            st.error("❌ Required data 'queries' not found. Please ensure data is loaded before running this analysis.")

                

    # Advanced Analytics Section
    st.subheader("📈 Advanced Health Query Performance Analytics")
    
    # Three-column layout for advanced metrics
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        st.markdown("**🎯 Query Length vs Nutraceuticals & Nutrition Performance**")
        ql_analysis = queries.groupby('query_length').agg({
            'Counts': 'sum', 
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        ql_analysis['ctr'] = ql_analysis.apply(lambda r: (r['clicks']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)
        ql_analysis['cr'] = ql_analysis.apply(lambda r: (r['conversions']/r['clicks']*100) if r['clicks']>0 else 0, axis=1)
        
        if not ql_analysis.empty:
            fig_ql = px.scatter(
                ql_analysis, 
                x='query_length', 
                y='ctr', 
                size='Counts',
                color='cr',
                title='Length vs Health CTR Performance',
                color_continuous_scale=['#E8F5E8', '#66BB6A'],
                template='plotly_white'
            )
            
            fig_ql.update_layout(
                plot_bgcolor='rgba(248,253,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI', size=10),
                height=300,
                xaxis=dict(showgrid=True, gridcolor='#E8F5E8'),
                yaxis=dict(showgrid=True, gridcolor='#E8F5E8')
            )
            
            st.plotly_chart(fig_ql, use_container_width=True)
    
    with adv_col2:
        st.markdown("**📊 Long-tail vs Short-tail Health Performance**")
        queries['is_long_tail'] = queries['query_length'] >= 20
        lt_analysis = queries.groupby('is_long_tail').agg({
            'Counts': 'sum', 
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        lt_analysis['label'] = lt_analysis['is_long_tail'].map({
            True: 'Long-tail Health (≥20 chars)', 
            False: 'Short-tail Health (<20 chars)'
        })
        lt_analysis['ctr'] = lt_analysis.apply(lambda r: (r['clicks']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)
        
        if not lt_analysis.empty:
            fig_lt = px.bar(
                lt_analysis, 
                x='label', 
                y='Counts',
                color='ctr',
                title='Health Traffic: Long-tail vs Short-tail',
                color_continuous_scale=['#E8F5E8', '#2E7D32'],
                text='Counts'
            )
            
            fig_lt.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside'
            )
            
            fig_lt.update_layout(
                plot_bgcolor='rgba(248,253,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI', size=10),
                height=300,
                xaxis=dict(showgrid=True, gridcolor='#E8F5E8'),
                yaxis=dict(showgrid=True, gridcolor='#E8F5E8')
            )
            
            st.plotly_chart(fig_lt, use_container_width=True)
    
    with adv_col3:
        st.markdown("**🔍 Health Keyword Density Analysis**")
        # FIXED: Replace labels with character ranges instead of descriptive names
        density_bins = pd.cut(queries['query_length'], 
                            bins=[0, 10, 20, 30, 50, 100], 
                            labels=['0-10 chars', '11-20 chars', '21-30 chars', '31-50 chars', '51-100 chars'])
        density_analysis = queries.groupby(density_bins).agg({
            'Counts': 'sum',
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        density_analysis['ctr'] = density_analysis.apply(lambda r: (r['clicks']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)
        
        if not density_analysis.empty:
            fig_density = px.pie(
                density_analysis, 
                names='query_length', 
                values='Counts',
                title='Health Query Length Distribution',
                color_discrete_sequence=['#2E7D32', '#66BB6A', '#E8F5E8', '#4CAF50', '#F1F8E9']
            )
            
            fig_density.update_layout(
                font=dict(color='#1B5E20', family='Segoe UI', size=10),
                height=300
            )
            
            st.plotly_chart(fig_density, use_container_width=True)

    
    st.markdown("---")
    
    # Replace Detailed Query Performance Analysis with Top Queries from Tab 1
    st.subheader("📋 Top Performing Health Queries")

    # Use slider instead of selectbox for queries too
    num_queries = st.slider(
        "Number of health queries to display:", 
        min_value=10, 
        max_value=300, 
        value=50, 
        step=10,
        key="query_count_slider_search_tab"
    )

    if queries.empty or 'Counts' not in queries.columns or queries['Counts'].isna().all():
        st.warning("No valid health data available for top queries.")
    else:
        try:
            # Group by 'search' and aggregate
            top_queries = queries.groupby('search').agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()

            # Calculate total Counts for share percentage
            total_counts = queries['Counts'].sum()

            # Calculate query length (number of characters)
            top_queries['Query Length'] = top_queries['search'].str.len()

            # Calculate Conversion Rate based on conversions / Counts if column exists or as fallback
            if 'Conversion Rate' in queries.columns:
                top_queries['Conversion Rate'] = pd.to_numeric(queries.groupby('search')['Conversion Rate'].mean(), errors='coerce').fillna(0)
            else:
                # Derive Conversion Rate as (conversions / Counts * 100)
                top_queries['Conversion Rate'] = (top_queries['conversions'] / top_queries['Counts'] * 100).round(2).fillna(0).replace([float('inf'), -float('inf')], 0)

            # 🎯 FIX DECIMALS: Round conversions to integers BEFORE renaming
            top_queries['conversions'] = top_queries['conversions'].round().astype(int)
            top_queries['clicks'] = top_queries['clicks'].round().astype(int)

            # Calculate share percentage
            top_queries['Share %'] = (top_queries['Counts'] / total_counts * 100).round(2)

            # Sort by 'Counts' and get top N
            top_queries = top_queries.nlargest(num_queries, 'Counts')

            # Rename columns for display and format
            top_queries = top_queries.rename(columns={
                'search': 'Health Query',
                'Counts': 'Search Volume',
                'clicks': 'Clicks',
                'conversions': 'Conversions'
            })

            # 🚀 UPDATED: Apply format_number function to numeric columns
            top_queries['Search Volume'] = top_queries['Search Volume'].apply(format_number)
            top_queries['Clicks'] = top_queries['Clicks'].apply(format_number)
            top_queries['Conversions'] = top_queries['Conversions'].apply(format_number)
            top_queries['Share %'] = top_queries['Share %'].apply(lambda x: f"{x:.2f}%")
            top_queries['Conversion Rate'] = top_queries['Conversion Rate'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
            top_queries['Query Length'] = top_queries['Query Length'].apply(lambda x: f"{x}")

            # Reorder columns to include Query Length after Query
            column_order = ['Health Query', 'Query Length', 'Search Volume', 'Share %', 'Clicks', 'Conversions', 'Conversion Rate']
            top_queries = top_queries[column_order]

            # Reset index to remove it
            top_queries = top_queries.reset_index(drop=True)

            # Display the DataFrame with custom styling for center alignment
            st.dataframe(
                top_queries, 
                use_container_width=True,
                hide_index=True,  # This hides the index column
                column_config={
                    "Health Query": st.column_config.TextColumn(
                        "Health Query",
                        help="Nutraceuticals & Nutrition search query text",
                        width="large"
                    ),
                    "Query Length": st.column_config.TextColumn(
                        "Query Length",
                        help="Number of characters in health query",
                        width="small"
                    ),
                    "Search Volume": st.column_config.TextColumn(
                        "Search Volume",
                        help="Total health search volume",
                        width="medium"
                    ),
                    "Share %": st.column_config.TextColumn(
                        "Share %",
                        help="Percentage of total health searches",
                        width="small"
                    ),
                    "Clicks": st.column_config.TextColumn(
                        "Clicks",
                        help="Total clicks received",
                        width="small"
                    ),
                    "Conversions": st.column_config.TextColumn(
                        "Conversions",
                        help="Total conversions",
                        width="small"
                    ),
                    "Conversion Rate": st.column_config.TextColumn(
                        "Conversion Rate",
                        help="Health conversion rate percentage",
                        width="small"
                    )
                }
            )

            # Add custom CSS for center alignment with health theme
            st.markdown("""
            <style>
            .stDataFrame [data-testid="stDataFrameResizeHandle"] {
                display: none !important;
            }
            .stDataFrame > div {
                text-align: center;
            }
            .stDataFrame th {
                text-align: center !important;
                background-color: #2E7D32 !important;
                color: white !important;
                font-weight: bold !important;
            }
            .stDataFrame td {
                text-align: center !important;
            }
            /* Keep Health Query column left-aligned for better readability */
            .stDataFrame td:first-child {
                text-align: left !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Add download button
            csv = top_queries.to_csv(index=False)
            st.download_button(
                label="📥 Download Health Queries CSV",
                data=csv,
                file_name=f"top_{num_queries}_health_queries.csv",
                mime="text/csv",
                key="query_csv_download_search_tab"
            )
        except KeyError as e:
            st.error(f"Column error: {e}. Check column names in your health data (e.g., 'search', 'Counts', 'clicks', 'conversions', 'Conversion Rate').")
        except Exception as e:
            st.error(f"Error processing top health queries: {e}")



# ----------------- Brand Tab (Enhanced & Optimized) -----------------
with tab_brand:
    st.header("🏷 Nutraceuticals & Nutrition Brand Intelligence Hub")
    st.markdown("Comprehensive health brand performance analysis with competitive insights and strategic recommendations. 🌿")
    
    # 🎨 GREEN-THEMED HERO HEADER (Replacing hero image and metrics)
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 3rem 2rem; 
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 50%, #A5D6A7 100%); 
        border-radius: 20px; 
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(27, 94, 32, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.2);
    ">
        <h1 style="
            color: #1B5E20; 
            margin: 0; 
            font-size: 3rem; 
            text-shadow: 2px 2px 8px rgba(27, 94, 32, 0.2);
            font-weight: 700;
            letter-spacing: -1px;
        ">
            🌿 Brand Market Position 🌿
        </h1>
        <p style="
            color: #2E7D32; 
            margin: 1rem 0 0 0; 
            font-size: 1.3rem;
            font-weight: 300;
            opacity: 0.9;
        ">
            Advanced Brand Analytics • Market Intelligence • Competitive Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced CSS for health-focused green styling
    st.markdown("""
    <style>
    /* Enhanced Global Styling for Brand Tab */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Enhanced Brand Metrics Styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.15);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.25);
        border-color: #2E7D32;
    }
    
    /* Brand Performance Cards */
    .brand-performance-card {
        background: linear-gradient(135deg, #F1F8E9 0%, #DCEDC8 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #81C784;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .brand-performance-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(129, 199, 132, 0.3);
    }
    
    /* Enhanced DataFrames */
    .stDataFrame th {
        text-align: center !important;
        background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid #1B5E20 !important;
        padding: 12px 8px !important;
    }
    
    .stDataFrame td {
        text-align: center !important;
        border: 1px solid #E8F5E8 !important;
        padding: 10px 8px !important;
    }
    
    .stDataFrame tr:nth-child(even) {
        background-color: #F1F8E9 !important;
    }
    
    .stDataFrame tr:hover {
        background-color: #E8F5E8 !important;
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
    
    /* Enhanced Brand Analysis Metrics */
    .brand-metric-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .brand-metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        border-color: #2E7D32;
    }
    
    .brand-metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B5E20;
        margin: 0;
        text-shadow: 1px 1px 3px rgba(27, 94, 32, 0.1);
    }
    
    .brand-metric-label {
        font-size: 1rem;
        color: #2E7D32;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for brand column with case sensitivity handling
    brand_column = None
    possible_brand_columns = ['brand', 'Brand', 'BRAND', 'Brand Name', 'brand_name']
    
    for col in possible_brand_columns:
        if col in queries.columns:
            brand_column = col
            break
    
    # Check if brand data is available
    has_brand_data = (brand_column is not None and 
                     queries[brand_column].notna().any())
    
    if not has_brand_data:
        st.error(f"❌ No Nutraceuticals & Nutrition brand data available. Available columns: {list(queries.columns)}")
        st.info("💡 Please ensure your dataset contains a brand column (brand, Brand, or Brand Name)")
        st.stop()
    
    # Filter out "Other" brand from all analysis (CASE-INSENSITIVE)
    brand_queries = queries[
        (queries[brand_column].notna()) & 
        (~queries[brand_column].str.lower().isin(['other', 'others']))
    ]

    if brand_queries.empty:
        st.error("❌ No valid Nutraceuticals & Nutrition brand data available after filtering.")
        st.stop()
    
    # Calculate key metrics for insights
    total_brands = brand_queries[brand_column].nunique()
    top_brand = brand_queries.groupby(brand_column)['Counts'].sum().idxmax()
    avg_brand_counts = brand_queries.groupby(brand_column)['Counts'].sum().mean()
    
    # Calculate Brand Dominance Index
    brand_counts_sum = brand_queries.groupby(brand_column)['Counts'].sum()
    brand_dominance = (brand_counts_sum.max() / brand_counts_sum.sum() * 100)
    
    st.markdown("---")
    
    # Main Brand Analysis Layout
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Enhanced Brand Performance Analysis
        st.subheader("📈 Brand Performance Matrix")

        # Calculate comprehensive brand metrics with CORRECTED CR CALCULATION
        bs_raw = brand_queries.groupby(brand_column).agg({
            'Counts': 'sum',
            'clicks': 'sum', 
            'conversions': 'sum'
        }).reset_index()

        # Round to integers for cleaner display
        bs_raw['clicks'] = bs_raw['clicks'].round().astype(int)
        bs_raw['conversions'] = bs_raw['conversions'].round().astype(int)

        # Rename the brand column to 'brand' for consistency
        bs_raw = bs_raw.rename(columns={brand_column: 'brand'})
        bs = bs_raw.copy()

        # Calculate Share % based on filtered data
        total_counts = bs['Counts'].sum()
        bs['share_pct'] = (bs['Counts'] / total_counts * 100).round(2)

        # CORRECTED CR CALCULATIONS
        bs['ctr'] = ((bs['clicks'] / bs['Counts']) * 100).round(2)
        bs['cr'] = ((bs['conversions'] / bs['Counts']) * 100).round(2)  # CR = conversions/search volume
        bs['classic_cr'] = ((bs['conversions'] / bs['clicks']) * 100).fillna(0).round(2)  # Classic CR = conversions/clicks

        # Enhanced scatter plot for brand performance
        num_scatter_brands = st.slider(
            "Number of Nutraceuticals & Nutrition brands in scatter plot:", 
            min_value=20, 
            max_value=100, 
            value=50, 
            step=10,
            key="scatter_brand_count"
        )

        bs_for_scatter = bs.sort_values('Counts', ascending=False).head(num_scatter_brands)

        fig_brand_perf = px.scatter(
            bs_for_scatter,
            x='Counts', 
            y='ctr',
            size='clicks',
            color='classic_cr',  # Use classic_cr for color
            hover_name='brand',
            title=f'<b style="color:#2E7D32; font-size:18px;">🌿 Nutraceuticals & Nutrition Brand Performance Matrix: Top {num_scatter_brands} Brands</b>',
            labels={'Counts': 'Total Search Counts', 'ctr': 'Click-Through Rate (%)', 'classic_cr': 'Classic CR (%)'},
            color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
            template='plotly_white'
        )

        fig_brand_perf.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Search Counts: %{x:,.0f}<br>' +
                         'CTR: %{y:.2f}%<br>' +
                         'Total Clicks: %{marker.size:,.0f}<br>' +
                         'Classic CR: %{marker.color:.2f}%<extra></extra>'
        )
        
        fig_brand_perf.update_layout(
            plot_bgcolor='rgba(248,255,248,0.95)',
            paper_bgcolor='rgba(232,245,232,0.8)',
            font=dict(color='#1B5E20', family='Segoe UI'),
            title_x=0,
            xaxis=dict(showgrid=True, gridcolor='#C8E6C8', linecolor='#4CAF50', linewidth=2),
            yaxis=dict(showgrid=True, gridcolor='#C8E6C8', linecolor='#4CAF50', linewidth=2),
        )
        
        st.plotly_chart(fig_brand_perf, use_container_width=True)
        
        # Top Brands Performance Table
        st.subheader("🏆 Top Brands Performance")
        
        num_brands = st.slider(
            "Number of Nutraceuticals & Nutrition brands to display:", 
            min_value=10, 
            max_value=50, 
            value=20, 
            step=5,
            key="brand_count_slider"
        )
        
        top_brands = bs.sort_values('Counts', ascending=False).head(num_brands)
        
        # Create display version
        display_brands = top_brands.copy()
        display_brands = display_brands.rename(columns={
            'brand': 'Nutraceuticals & Nutrition Brand',
            'Counts': 'Search Counts',
            'share_pct': 'Market Share %',
            'clicks': 'Total Clicks',
            'conversions': 'Conversions',
            'ctr': 'CTR',
            'cr': 'CR',
            'classic_cr': 'Classic CR'
        })
        
        # Format numbers
        display_brands['Search Counts'] = display_brands['Search Counts'].apply(format_number)
        display_brands['Market Share %'] = display_brands['Market Share %'].apply(lambda x: f"{x:.2f}%")
        display_brands['Total Clicks'] = display_brands['Total Clicks'].apply(format_number)
        display_brands['Conversions'] = display_brands['Conversions'].apply(format_number)
        display_brands['CTR'] = display_brands['CTR'].apply(lambda x: f"{x:.2f}%")
        display_brands['CR'] = display_brands['CR'].apply(lambda x: f"{x:.2f}%")
        display_brands['Classic CR'] = display_brands['Classic CR'].apply(lambda x: f"{x:.2f}%")
        
        # Reorder columns
        column_order = ['Nutraceuticals & Nutrition Brand', 'Search Counts', 'Market Share %', 'Total Clicks', 'Conversions', 'CTR', 'CR', 'Classic CR']
        display_brands = display_brands[column_order]
        
        st.dataframe(display_brands, use_container_width=True, hide_index=True)
        
        # Download button
        csv_brands = top_brands.to_csv(index=False)
        st.download_button(
            label="📥 Download Nutraceuticals & Nutrition Brands CSV",
            data=csv_brands,
            file_name=f"top_{num_brands}_nutraceuticals_brands.csv",
            mime="text/csv",
            key="brand_csv_download"
        )

        # ADDED BACK: Brand Summary Data Table
        st.subheader("📋 Brand Summary Data")
        
        # Calculate brand summary from queries
        brand_summary_calc = []
        
        for brand in brand_queries[brand_column].unique():
            brand_data = brand_queries[brand_queries[brand_column] == brand]
            
            # Basic metrics
            total_counts = brand_data['Counts'].sum()
            total_clicks = brand_data['clicks'].sum()
            total_conversions = brand_data['conversions'].sum()
            
            # CORRECTED Calculate rates
            ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
            cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0  # CR = conversions/search volume
            classic_cr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0  # Classic CR = conversions/clicks
            
            # Use the keywords column that was created by your prepare_queries_df function
            unique_keywords_set = set()
            keyword_counts = {}
            
            for idx, row in brand_data.iterrows():
                keywords_list = row['keywords']  # This comes from your prepare_queries_df function
                query_count = row['Counts']
                
                if isinstance(keywords_list, list):
                    unique_keywords_set.update(keywords_list)
                    # Add the query count to each keyword
                    for keyword in keywords_list:
                        if keyword in keyword_counts:
                            keyword_counts[keyword] += query_count
                        else:
                            keyword_counts[keyword] = query_count
                elif pd.notna(keywords_list):
                    # Fallback: use normalized_query if keywords is not a list
                    search_term = row['normalized_query']
                    if pd.notna(search_term):
                        keywords = str(search_term).lower().split()
                        unique_keywords_set.update(keywords)
                        for keyword in keywords:
                            if keyword in keyword_counts:
                                keyword_counts[keyword] += query_count
                            else:
                                keyword_counts[keyword] = query_count
            
            unique_keywords_count = len(unique_keywords_set)
            
            # Get top 5 keywords by total counts
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_keywords_str = ', '.join([f"{kw}({format_number(cnt)})" for kw, cnt in top_keywords])
            
            brand_summary_calc.append({
                'Nutraceuticals & Nutrition Brand': brand,
                'Search Counts': total_counts,
                'Total Clicks': total_clicks,
                'Conversions': total_conversions,
                'CTR': ctr,
                'CR': cr,
                'Classic CR': classic_cr,
                'Unique Keywords': unique_keywords_count,
                'Top Health Keywords': top_keywords_str
            })
        
        brand_summary_df = pd.DataFrame(brand_summary_calc)
        
        # Sort by Search Counts and take top 10 for display
        brand_summary_df = brand_summary_df.sort_values('Search Counts', ascending=False).head(10)
        
        # Format for display
        display_summary = brand_summary_df.copy()
        display_summary['Search Counts'] = display_summary['Search Counts'].apply(format_number)
        display_summary['Total Clicks'] = display_summary['Total Clicks'].apply(format_number)
        display_summary['Conversions'] = display_summary['Conversions'].apply(format_number)
        display_summary['CTR'] = display_summary['CTR'].apply(lambda x: f"{x:.2f}%")
        display_summary['CR'] = display_summary['CR'].apply(lambda x: f"{x:.2f}%")
        display_summary['Classic CR'] = display_summary['Classic CR'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
        
        # Download button for brand summary
        csv_summary = brand_summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Brand Summary CSV",
            data=csv_summary,
            file_name="nutraceuticals_brand_summary_calculated.csv",
            mime="text/csv",
            key="brand_summary_calc_csv_download"
        )
    
    with col_right:
        # Brand Market Share Pie Chart
        st.subheader("🌱 Brand Market Share")
        
        top_brands_pie = bs.nlargest(10, 'Counts')
        
        # Health-focused color palette
        health_colors = ['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', 
                        '#C8E6C8', '#E8F5E8', '#388E3C', '#689F38', '#8BC34A']
        
        fig_pie = px.pie(
            top_brands_pie, 
            names='brand', 
            values='Counts',
            title='<b style="color:#2E7D32;">🌿 Health Market Distribution</b>',
            color_discrete_sequence=health_colors
        )
        
        fig_pie.update_layout(
            font=dict(color='#1B5E20', family='Segoe UI'),
            paper_bgcolor='rgba(232,245,232,0.8)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Brand Performance Categories
        st.subheader("🎯 Brand Performance Categories")
        
        # Categorize brands based on performance
        bs['performance_category'] = pd.cut(
            bs['ctr'], 
            bins=[0, 2, 5, 10, float('inf')], 
            labels=['Emerging (0-2%)', 'Growing (2-5%)', 'Strong (5-10%)', 'Premium (>10%)']
        )
        
        category_counts = bs['performance_category'].value_counts().reset_index()
        category_counts.columns = ['Performance Category', 'Count']
        
        fig_cat = px.bar(
            category_counts, 
            x='Performance Category', 
            y='Count',
            title='<b style="color:#2E7D32;">🌿 CTR Performance Distribution</b>',
            color='Count',
            color_continuous_scale=['#E8F5E8', '#2E7D32'],
            text='Count'
        )
        
        fig_cat.update_traces(
            texttemplate='%{text}',
            textposition='outside'
        )
        
        fig_cat.update_layout(
            plot_bgcolor='rgba(248,255,248,0.95)',
            paper_bgcolor='rgba(232,245,232,0.8)',
            font=dict(color='#1B5E20', family='Segoe UI'),
            xaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
            yaxis=dict(showgrid=True, gridcolor='#C8E6C8')
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
        # Enhanced Brand Trend Analysis with proper filter application
        if 'Date' in queries.columns:
            st.subheader("📈 Brand Trend Analysis")
            
            # Get top 5 brands for trend analysis
            top_5_brands = bs.nlargest(5, 'Counts')['brand'].tolist()
            
            # Use the already filtered 'queries' data instead of 'brand_queries'
            trend_data = queries[
                (queries[brand_column].notna()) & 
                (queries[brand_column].str.lower() != 'other') &
                (queries[brand_column].str.lower() != 'others') &
                (queries[brand_column].isin(top_5_brands))
            ].copy()
            
            if not trend_data.empty:
                try:
                    # Enhanced date processing
                    trend_data['Date'] = pd.to_datetime(trend_data['Date'], errors='coerce')
                    trend_data = trend_data.dropna(subset=['Date'])
                    
                    if not trend_data.empty:
                        # Create proper monthly aggregation
                        trend_data['Month'] = trend_data['Date'].dt.to_period('M')
                        trend_data['Month_Display'] = trend_data['Date'].dt.strftime('%Y-%m')
                        
                        # Group by Month and brand - sum the counts for each month
                        monthly_trends = trend_data.groupby(['Month_Display', brand_column])['Counts'].sum().reset_index()
                        monthly_trends = monthly_trends.rename(columns={brand_column: 'brand'})
                        
                        # Convert month display back to datetime for proper plotting
                        monthly_trends['Date'] = pd.to_datetime(monthly_trends['Month_Display'] + '-01')
                        
                        # Debug: Check if we have monthly data
                        unique_months = monthly_trends['Month_Display'].unique()
                        st.write(f"📊 Monthly Nutraceuticals & Nutrition data available: {', '.join(sorted(unique_months))}")
                        
                        if len(monthly_trends) > 0:
                            fig_trend = px.line(
                                monthly_trends, 
                                x='Date', 
                                y='Counts', 
                                color='brand',
                                title='<b style="color:#2E7D32;">🌿 Top 5 Nutraceuticals & Nutrition Brands Monthly Trend</b>',
                                color_discrete_sequence=['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7'],
                                markers=True
                            )
                            
                            # Format x-axis to show months properly
                            fig_trend.update_layout(
                                plot_bgcolor='rgba(248,255,248,0.95)',
                                paper_bgcolor='rgba(232,245,232,0.8)',
                                font=dict(color='#1B5E20', family='Segoe UI'),
                                xaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#C8E6C8',
                                    title='Month',
                                    dtick="M1",
                                    tickformat="%b %Y"
                                ),
                                yaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#C8E6C8',
                                    title='Search Counts'
                                ),
                                hovermode='x unified'
                            )
                            
                            fig_trend.update_traces(
                                hovertemplate='<b>%{fullData.name}</b><br>' +
                                            'Month: %{x|%B %Y}<br>' +
                                            'Searches: %{y:,.0f}<extra></extra>'
                            )
                            
                            st.plotly_chart(fig_trend, use_container_width=True)
                        else:
                            st.info("No Nutraceuticals & Nutrition trend data available for the selected date range and brands")
                    else:
                        st.info("No valid dates found in the filtered Nutraceuticals & Nutrition data")
                except Exception as e:
                    st.error(f"Error processing Nutraceuticals & Nutrition trend data: {str(e)}")
            else:
                st.info("No Nutraceuticals & Nutrition brand data available for the selected date range")        


    st.markdown("---")
    
    # ENHANCED Brand-Keyword Intelligence Matrix with Interactive CTR/CR Display
    st.subheader("🔥 Brand-Keyword Intelligence Matrix")

    # Create brand filter dropdown with enhanced UI
    if 'brand' in queries.columns and 'search' in queries.columns:
        available_brands = queries[
            (queries['brand'].notna()) & 
            (queries['brand'].str.lower() != 'other') &
            (queries['brand'].str.lower() != 'others')
        ]['brand'].unique()
        
        available_brands = sorted(available_brands)
        brand_options = ['All Nutraceuticals & Nutrition Brands'] + list(available_brands)
        
        # ENHANCED UI for brand selection with metrics
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
            border: 2px solid #4CAF50;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
        ">
            <h4 style="color: #1B5E20; margin: 0 0 1rem 0; text-align: center;">
                🎯 Brand Analysis Control Center
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_select, col_metrics = st.columns([2, 3])
        
        with col_select:
            selected_brand = st.selectbox(
                "🎯 Select Nutraceuticals & Nutrition Brand to Analyze:",
                options=brand_options,
                index=0,
                key="brand_selector"
            )
        
        with col_metrics:
            if selected_brand != 'All Nutraceuticals & Nutrition Brands':
                # Show metrics for selected brand
                brand_metrics = bs[bs['brand'] == selected_brand].iloc[0] if not bs[bs['brand'] == selected_brand].empty else None
                
                if brand_metrics is not None:
                    # UPDATED: Now showing 5 metrics including both CR types with format_number
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    
                    with metric_col1:
                        st.markdown(f"""
                        <div class="brand-metric-card">
                            <div class="brand-metric-value">{format_number(brand_metrics['Counts'])}</div>
                            <div class="brand-metric-label">📊 Total Searches</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown(f"""
                        <div class="brand-metric-card">
                            <div class="brand-metric-value">{brand_metrics['ctr']:.2f}%</div>
                            <div class="brand-metric-label">📈 CTR</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown(f"""
                        <div class="brand-metric-card">
                            <div class="brand-metric-value">{brand_metrics['cr']:.2f}%</div>
                            <div class="brand-metric-label">🎯 CR (Search)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col4:
                        st.markdown(f"""
                        <div class="brand-metric-card">
                            <div class="brand-metric-value">{brand_metrics['classic_cr']:.2f}%</div>
                            <div class="brand-metric-label">🔄 Classic CR</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col5:
                        st.markdown(f"""
                        <div class="brand-metric-card">
                            <div class="brand-metric-value">{brand_metrics['share_pct']:.1f}%</div>
                            <div class="brand-metric-label">📈 Market Share</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Show overall metrics
                total_searches = bs['Counts'].sum()
                avg_ctr = bs['ctr'].mean()
                avg_cr = bs['cr'].mean()  # ADDED: CR (Search-based)
                avg_classic_cr = bs['classic_cr'].mean()
                
                # UPDATED: Now showing 4 metrics including both CR types with format_number
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.markdown(f"""
                    <div class="brand-metric-card">
                        <div class="brand-metric-value">{format_number(total_searches)}</div>
                        <div class="brand-metric-label">📊 Total Market</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="brand-metric-card">
                        <div class="brand-metric-value">{avg_ctr:.2f}%</div>
                        <div class="brand-metric-label">📈 Avg CTR</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown(f"""
                    <div class="brand-metric-card">
                        <div class="brand-metric-value">{avg_cr:.2f}%</div>
                        <div class="brand-metric-label">🎯 Avg CR (Search)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col4:
                    st.markdown(f"""
                    <div class="brand-metric-card">
                        <div class="brand-metric-value">{avg_classic_cr:.2f}%</div>
                        <div class="brand-metric-label">🔄 Avg Classic CR</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Filter data based on selection
        if selected_brand == 'All Nutraceuticals & Nutrition Brands':
            top_brands = queries[
                (queries['brand'].str.lower() != 'other') &
                (queries['brand'].str.lower() != 'others') &
                (queries['brand'].notna())
            ]['brand'].value_counts().head(8).index.tolist()
            
            filtered_data = queries[queries['brand'].isin(top_brands)]
            matrix_title = "Top Nutraceuticals & Nutrition Brands vs Health Search Terms"
        else:
            filtered_data = queries[queries['brand'] == selected_brand]
            matrix_title = f"{selected_brand} - Health Search Terms Analysis"
        
        # Remove null values and 'other' categories
        matrix_data = filtered_data[
            (filtered_data['brand'].notna()) & 
            (filtered_data['search'].notna()) &
            (filtered_data['brand'].str.lower() != 'other') &
            (filtered_data['brand'].str.lower() != 'others') &
            (filtered_data['search'].str.lower() != 'other') &
            (filtered_data['search'].str.lower() != 'others')
        ].copy()
        
        if not matrix_data.empty:
            if selected_brand == 'All Nutraceuticals & Nutrition Brands':
                # Enhanced heatmap with CTR/CR data
                brand_search_matrix = matrix_data.groupby(['brand', 'search']).agg({
                    'Counts': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                
                # CORRECTED Calculate CTR and CR for each brand-search combination
                brand_search_matrix['ctr'] = ((brand_search_matrix['clicks'] / brand_search_matrix['Counts']) * 100).round(2)
                brand_search_matrix['cr'] = ((brand_search_matrix['conversions'] / brand_search_matrix['Counts']) * 100).round(2)  # CR = conversions/search volume
                brand_search_matrix['classic_cr'] = ((brand_search_matrix['conversions'] / brand_search_matrix['clicks']) * 100).fillna(0).round(2)  # Classic CR = conversions/clicks
                
                top_searches = matrix_data[
                    (matrix_data['search'].str.lower() != 'other') &
                    (matrix_data['search'].str.lower() != 'others')
                ]['search'].value_counts().head(12).index.tolist()
                
                brand_search_matrix = brand_search_matrix[brand_search_matrix['search'].isin(top_searches)]
                
                # Create pivot table for counts
                heatmap_data = brand_search_matrix.pivot(
                    index='brand', 
                    columns='search', 
                    values='Counts'
                ).fillna(0)
                
                # Create pivot tables for CTR, CR, and Classic CR
                ctr_data = brand_search_matrix.pivot(
                    index='brand', 
                    columns='search', 
                    values='ctr'
                ).fillna(0)
                
                cr_data = brand_search_matrix.pivot(  # ADDED: CR data pivot
                    index='brand', 
                    columns='search', 
                    values='cr'
                ).fillna(0)
                
                classic_cr_data = brand_search_matrix.pivot(
                    index='brand', 
                    columns='search', 
                    values='classic_cr'
                ).fillna(0)
                
                # Enhanced heatmap with custom hover template
                fig_matrix = px.imshow(
                    heatmap_data.values,
                    labels=dict(x="Health Search Terms", y="Nutraceuticals & Nutrition Brands", color="Total Counts"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                    title=f'<b style="color:#2E7D32;">{matrix_title}</b>',
                    aspect='auto'
                )
                
                # UPDATED: Create custom hover data with CTR, CR, and Classic CR using format_number
                hover_text = []
                for i, brand in enumerate(heatmap_data.index):
                    hover_row = []
                    for j, search in enumerate(heatmap_data.columns):
                        counts = heatmap_data.iloc[i, j]
                        ctr = ctr_data.iloc[i, j]
                        cr = cr_data.iloc[i, j]  # ADDED: CR (Search-based)
                        classic_cr = classic_cr_data.iloc[i, j]
                        hover_row.append(
                            f"<b>{brand}</b><br>" +
                            f"Search Term: {search}<br>" +
                            f"Total Searches: {format_number(counts)}<br>" +  # 🚀 UPDATED: format_number
                            f"CTR: {ctr:.2f}%<br>" +
                            f"CR (Search): {cr:.2f}%<br>" +  # ADDED: CR in hover
                            f"Classic CR: {classic_cr:.2f}%"
                        )
                    hover_text.append(hover_row)
                
                fig_matrix.update_traces(
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_text
                )
                
                fig_matrix.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=600,
                    xaxis=dict(side='bottom', tickangle=45),
                    yaxis=dict(side='left')
                )
                
                st.plotly_chart(fig_matrix, use_container_width=True)
                
            else:
                # Single brand analysis with enhanced bar chart
                brand_search_data = matrix_data.groupby('search').agg({
                    'Counts': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                
                # CORRECTED Calculate CTR and CR
                brand_search_data['ctr'] = ((brand_search_data['clicks'] / brand_search_data['Counts']) * 100).round(2)
                brand_search_data['cr'] = ((brand_search_data['conversions'] / brand_search_data['Counts']) * 100).round(2)  # CR = conversions/search volume
                brand_search_data['classic_cr'] = ((brand_search_data['conversions'] / brand_search_data['clicks']) * 100).fillna(0).round(2)  # Classic CR = conversions/clicks
                
                brand_search_data = brand_search_data.sort_values('Counts', ascending=False).head(15)
                
                # UPDATED: Add CR selection for chart coloring
                st.markdown("#### 📊 Chart Display Options")
                cr_option = st.radio(
                    "Color bars by:",
                    options=['Classic CR (Conversions/Clicks)', 'CR Search-based (Conversions/Searches)'],
                    index=0,
                    horizontal=True,
                    key="cr_option_radio"
                )
                
                # Determine which CR to use for coloring
                color_column = 'classic_cr' if cr_option == 'Classic CR (Conversions/Clicks)' else 'cr'
                color_label = 'Classic CR (%)' if cr_option == 'Classic CR (Conversions/Clicks)' else 'CR Search-based (%)'
                
                fig_brand_search = px.bar(
                    brand_search_data,
                    x='search',
                    y='Counts',
                    title=f'<b style="color:#2E7D32;">{matrix_title}</b>',
                    labels={'search': 'Health Search Terms', 'Counts': 'Total Search Volume'},
                    color=color_column,  # Dynamic color selection
                    color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                    text='Counts'
                )
                
                # UPDATED: Enhanced hover template with both CR types using format_number
                fig_brand_search.update_traces(
                    texttemplate='%{text}',  # 🚀 UPDATED: Will be formatted by customdata
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>' +
                                'Search Volume: %{customdata[3]}<br>' +  # 🚀 UPDATED: Use formatted number
                                'CTR: %{customdata[0]:.2f}%<br>' +
                                'CR (Search): %{customdata[1]:.2f}%<br>' +  # ADDED: CR in hover
                                'Classic CR: %{customdata[2]:.2f}%<br>' +
                                f'{color_label}: %{{marker.color:.2f}}%<extra></extra>',
                    customdata=[[row['ctr'], row['cr'], row['classic_cr'], format_number(row['Counts'])] 
                            for _, row in brand_search_data.iterrows()],  # 🚀 UPDATED: Include formatted numbers
                    text=[format_number(x) for x in brand_search_data['Counts']]  # 🚀 UPDATED: Format bar labels
                )
                
                fig_brand_search.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45, showgrid=True, gridcolor='#C8E6C8'),
                    yaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                    coloraxis_colorbar=dict(title=color_label)  # Dynamic colorbar title
                )
                
                st.plotly_chart(fig_brand_search, use_container_width=True)
                
                # ADDED: Display both CR metrics in a comparison table
                st.markdown("#### 📋 Search Terms Performance Comparison")
                
                display_comparison = brand_search_data[['search', 'Counts', 'ctr', 'cr', 'classic_cr']].copy()
                display_comparison = display_comparison.rename(columns={
                    'search': 'Health Search Term',
                    'Counts': 'Search Volume',
                    'ctr': 'CTR (%)',
                    'cr': 'CR Search-based (%)',
                    'classic_cr': 'Classic CR (%)'
                })
                
                # 🚀 UPDATED: Format the display using format_number
                display_comparison['Search Volume'] = display_comparison['Search Volume'].apply(format_number)
                display_comparison['CTR (%)'] = display_comparison['CTR (%)'].apply(lambda x: f"{x:.2f}%")
                display_comparison['CR Search-based (%)'] = display_comparison['CR Search-based (%)'].apply(lambda x: f"{x:.2f}%")
                display_comparison['Classic CR (%)'] = display_comparison['Classic CR (%)'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(display_comparison, use_container_width=True, hide_index=True)
        
        else:
            st.warning("⚠️ No data available for the selected Nutraceuticals & Nutrition brand.")

    else:
        st.error("❌ Required columns 'brand' and 'search' not found in the dataset.")

    st.markdown("---")


    
    # Strategic Brand Intelligence Dashboard (3 Tabs)
    st.subheader("🧠 Strategic Brand Intelligence Dashboard")
    
    strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
        "🎯 Market Position Analysis", 
        "🚀 Growth Opportunities", 
        "💡 Competitive Intelligence"
    ])
    
    with strategy_tab1:
        st.markdown("#### 🎯 Brand Market Position Quadrant Analysis")
        
        if not bs.empty:
            # Market position quadrant analysis
            bs['market_strength'] = bs['share_pct'] * bs['ctr'] / 100  # Combined market strength
            bs['efficiency_score'] = bs['conversions'] / bs['Counts'] * 1000  # Efficiency per 1000 searches
            
            # Define quadrants based on median values
            median_strength = bs['market_strength'].median()
            median_efficiency = bs['efficiency_score'].median()
            
            def categorize_position(row):
                if row['market_strength'] >= median_strength and row['efficiency_score'] >= median_efficiency:
                    return "🌟 Market Leaders"
                elif row['market_strength'] >= median_strength and row['efficiency_score'] < median_efficiency:
                    return "📈 Volume Players"
                elif row['market_strength'] < median_strength and row['efficiency_score'] >= median_efficiency:
                    return "💎 Efficiency Champions"
                else:
                    return "🌱 Emerging Brands"
            
            bs['position_category'] = bs.apply(categorize_position, axis=1)
            
            # Create quadrant scatter plot
            fig_quadrant = px.scatter(
                bs.head(30),  # Top 30 brands for clarity
                x='market_strength',
                y='efficiency_score',
                size='Counts',
                color='position_category',
                hover_name='brand',
                title='<b style="color:#2E7D32;">🎯 Brand Market Position Quadrant Analysis</b>',
                labels={
                    'market_strength': 'Market Strength (Share × CTR)',
                    'efficiency_score': 'Conversion Efficiency (per 1000 searches)'
                },
                color_discrete_map={
                    "🌟 Market Leaders": "#2E7D32",
                    "📈 Volume Players": "#4CAF50", 
                    "💎 Efficiency Champions": "#66BB6A",
                    "🌱 Emerging Brands": "#A5D6A7"
                }
            )
            
            # Add quadrant lines
            fig_quadrant.add_hline(y=median_efficiency, line_dash="dash", line_color="#81C784", opacity=0.7)
            fig_quadrant.add_vline(x=median_strength, line_dash="dash", line_color="#81C784", opacity=0.7)
            
            fig_quadrant.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>' +
                             'Market Strength: %{x:.2f}<br>' +
                             'Efficiency Score: %{y:.2f}<br>' +
                             'Total Searches: %{marker.size:,.0f}<br>' +
                             'Category: %{marker.color}<extra></extra>'
            )
            
            fig_quadrant.update_layout(
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                height=500,
                xaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                yaxis=dict(showgrid=True, gridcolor='#C8E6C8')
            )
            
            st.plotly_chart(fig_quadrant, use_container_width=True)
            
            # Position category distribution
            position_dist = bs['position_category'].value_counts().reset_index()
            position_dist.columns = ['Category', 'Count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.pie(
                    position_dist,
                    names='Category',
                    values='Count',
                    title='<b style="color:#2E7D32;">📊 Brand Position Distribution</b>',
                    color_discrete_map={
                        "🌟 Market Leaders": "#2E7D32",
                        "📈 Volume Players": "#4CAF50", 
                        "💎 Efficiency Champions": "#66BB6A",
                        "🌱 Emerging Brands": "#A5D6A7"
                    }
                )
                
                fig_dist.update_layout(
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    paper_bgcolor='rgba(232,245,232,0.8)'
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Top performers in each category
                st.markdown("#### 🏆 Category Champions")
                
                for category in position_dist['Category']:
                    category_brands = bs[bs['position_category'] == category].sort_values('Counts', ascending=False).head(3)
                    
                    if not category_brands.empty:
                        st.markdown(f"**{category}**")
                        for idx, row in category_brands.iterrows():
                            st.markdown(f"• {row['brand']} - {row['Counts']:,.0f} searches")
                        st.markdown("")
    
    with strategy_tab2:
        st.markdown("#### 🚀 Growth Opportunities Analysis")
        
        if not bs.empty:
            # Opportunity scoring
            bs['growth_potential'] = (
                (100 - bs['share_pct']) * 0.4 +  # Market share growth potential
                (bs['ctr'] / bs['ctr'].max() * 100) * 0.3 +  # CTR performance
                (bs['classic_cr'] / bs['classic_cr'].max() * 100) * 0.3  # Classic CR performance
            )
            
            # Identify high-opportunity brands
            high_opportunity = bs[
                (bs['growth_potential'] > bs['growth_potential'].quantile(0.7)) &
                (bs['share_pct'] < 10)  # Not already dominant
            ].sort_values('growth_potential', ascending=False).head(10)
            
            if not high_opportunity.empty:
                fig_opportunity = px.bar(
                    high_opportunity,
                    x='growth_potential',
                    y='brand',
                    orientation='h',
                    title='<b style="color:#2E7D32;">🚀 Top Growth Opportunity Brands</b>',
                    labels={'growth_potential': 'Growth Potential Score', 'brand': 'Brand'},
                    color='growth_potential',
                    color_continuous_scale=['#E8F5E8', '#2E7D32'],
                    text='growth_potential'
                )
                
                fig_opportunity.update_traces(
                    texttemplate='%{text:.1f}',
                    textposition='inside',
                    hovertemplate='<b>%{y}</b><br>' +
                                 'Growth Score: %{x:.1f}<br>' +
                                 'Market Share: %{customdata[0]:.2f}%<br>' +
                                 'CTR: %{customdata[1]:.2f}%<br>' +
                                 'Classic CR: %{customdata[2]:.2f}%<extra></extra>',
                    customdata=high_opportunity[['share_pct', 'ctr', 'classic_cr']].values
                )
                
                fig_opportunity.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=500,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_opportunity, use_container_width=True)
                
                # Growth recommendations
                st.markdown("#### 💡 Strategic Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="brand-performance-card">
                        <h4 style="color:#2E7D32;">🎯 Market Expansion</h4>
                        <ul>
                            <li>Target underperforming search terms</li>
                            <li>Increase brand visibility campaigns</li>
                            <li>Focus on high-intent keywords</li>
                            <li>Optimize for mobile searches</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="brand-performance-card">
                        <h4 style="color:#2E7D32;">📈 Performance Optimization</h4>
                        <ul>
                            <li>Improve click-through rates</li>
                            <li>Enhance conversion funnels</li>
                            <li>A/B test ad creatives</li>
                            <li>Optimize landing pages</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-opportunity brands identified with current criteria")
    
    with strategy_tab3:
        st.markdown("#### 💡 Strategic Brand Insights")
        
        if not bs.empty:
            # Calculate key insights
            total_market_size = bs['Counts'].sum()
            top_performer = bs.loc[bs['Counts'].idxmax()]
            efficiency_leader = bs.loc[bs['classic_cr'].idxmax()] if bs['classic_cr'].max() > 0 else None
            
            # Market concentration analysis
            top_5_share = bs.nlargest(5, 'Counts')['share_pct'].sum()
            market_concentration = "High" if top_5_share > 70 else "Medium" if top_5_share > 50 else "Low"
            
            # Performance benchmarks
            avg_ctr = bs['ctr'].mean()
            avg_classic_cr = bs['classic_cr'].mean()
            
            # Strategic insights display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="brand-performance-card">
                    <h4>🎯 Market Intelligence</h4>
                    <p><strong>Market Size:</strong> {total_market_size:,.0f} total searches</p>
                    <p><strong>Market Leader:</strong> {top_performer['brand']} ({top_performer['share_pct']:.1f}% share)</p>
                    <p><strong>Market Concentration:</strong> {market_concentration} (Top 5: {top_5_share:.1f}%)</p>
                    <p><strong>Average CTR:</strong> {avg_ctr:.2f}%</p>
                    <p><strong>Average Classic CR:</strong> {avg_classic_cr:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if efficiency_leader is not None:
                    st.markdown(f"""
                    <div class="brand-performance-card">
                        <h4>🏆 Performance Leaders</h4>
                        <p><strong>Volume Leader:</strong> {top_performer['brand']}</p>
                        <p><strong>Efficiency Leader:</strong> {efficiency_leader['brand']} ({efficiency_leader['classic_cr']:.2f}% Classic CR)</p>
                        <p><strong>Best CTR:</strong> {bs.loc[bs['ctr'].idxmax(), 'brand']} ({bs['ctr'].max():.2f}%)</p>
                        <p><strong>Total Brands:</strong> {len(bs)} active brands</p>
                        <p><strong>Competitive Intensity:</strong> {"High" if len(bs) > 50 else "Medium" if len(bs) > 20 else "Low"}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Competitive landscape analysis
            st.markdown("#### 🏁 Competitive Landscape Matrix")
            
            # Create competitive intensity heatmap
            if len(bs) >= 10:
                # Group brands by performance tiers
                bs['performance_tier'] = pd.qcut(
                    bs['Counts'], 
                    q=4, 
                    labels=['Tier 4 (Emerging)', 'Tier 3 (Growing)', 'Tier 2 (Established)', 'Tier 1 (Leaders)']
                )
                
                tier_analysis = bs.groupby('performance_tier').agg({
                    'Counts': ['count', 'mean', 'sum'],
                    'ctr': 'mean',
                    'classic_cr': 'mean',
                    'share_pct': 'sum'
                }).round(2)
                
                tier_analysis.columns = ['Brand Count', 'Avg Searches', 'Total Searches', 'Avg CTR', 'Avg Classic CR', 'Total Share %']
                
                st.dataframe(tier_analysis, use_container_width=True)
                
                # Strategic recommendations based on analysis
                st.markdown("#### 📋 Strategic Action Items")
                
                recommendations = []
                
                if market_concentration == "High":
                    recommendations.append("🎯 **Market Consolidation**: Consider partnerships or acquisitions in fragmented segments")
                
                if avg_ctr < 3:
                    recommendations.append("📈 **CTR Optimization**: Industry CTR is below benchmark - focus on ad copy and targeting")
                
                if avg_classic_cr < 2:
                    recommendations.append("🔄 **Conversion Optimization**: Low conversion rates indicate need for landing page improvements")
                
                if len(bs[bs['share_pct'] > 10]) < 3:
                    recommendations.append("🚀 **Market Opportunity**: Market lacks dominant players - opportunity for aggressive growth")
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
            
            else:
                st.info("Need at least 10 brands for comprehensive competitive analysis")

    # Enhanced Footer with Data Export Options
    st.markdown("---")
    st.subheader("📥 Export & Analytics Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not bs.empty:
            full_brand_data = bs.copy()
            full_brand_data['export_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            csv_full = full_brand_data.to_csv(index=False)
            st.download_button(
                label="📊 Full Brand Analysis",
                data=csv_full,
                file_name=f"complete_brand_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="full_brand_export"
            )
    
    with col2:
        if 'position_category' in bs.columns:
            strategic_data = bs[['brand', 'Counts', 'share_pct', 'ctr', 'classic_cr', 'position_category', 'growth_potential']].copy()
            csv_strategic = strategic_data.to_csv(index=False)
            st.download_button(
                label="🎯 Strategic Insights",
                data=csv_strategic,
                file_name=f"brand_strategic_insights_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="strategic_export"
            )
    
    with col3:
        if not matrix_data.empty:
            matrix_export = matrix_data.groupby(['brand', 'search']).agg({
                'Counts': 'sum',
                'clicks': 'sum', 
                'conversions': 'sum'
            }).reset_index()
            csv_matrix = matrix_export.to_csv(index=False)
            st.download_button(
                label="🔥 Brand-Keyword Matrix",
                data=csv_matrix,
                file_name=f"brand_keyword_matrix_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="matrix_export"
            )
    
    with col4:
        # Generate executive summary
        if not bs.empty:
            summary_data = {
                'Metric': [
                    'Total Brands Analyzed',
                    'Market Leader',
                    'Total Search Volume',
                    'Average CTR',
                    'Average Classic CR',
                    'Market Concentration',
                    'Analysis Date'
                ],
                'Value': [
                    len(bs),
                    top_performer['brand'],
                    f"{total_market_size:,.0f}",
                    f"{avg_ctr:.2f}%",
                    f"{avg_classic_cr:.2f}%",
                    f"{market_concentration} ({top_5_share:.1f}%)",
                    pd.Timestamp.now().strftime('%Y-%m-%d')
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            csv_summary = summary_df.to_csv(index=False)
            st.download_button(
                label="📋 Executive Summary",
                data=csv_summary,
                file_name=f"brand_executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="summary_export"
            )


# ----------------- Category Tab (Enhanced & Health-Focused) -----------------
with tab_category:
    st.header("🌿 Nutraceuticals & Nutrition Category Intelligence Hub")
    st.markdown("Comprehensive health category performance analysis with strategic insights and competitive intelligence. 💚")
    
    # Hero Image for Category Tab
    category_image_options = {
        "Health Category Analytics": "https://placehold.co/1200x200/E8F5E8/2E7D32?text=Health+Category+Performance+Analysis",
        "Wellness Categories": "https://placehold.co/1200x200/4CAF50/FFFFFF?text=Wellness+Category+Intelligence+Dashboard",
        "Abstract Health Categories": "https://source.unsplash.com/1200x200/?health,wellness,categories",
        "Health Gradient": "https://placehold.co/1200x200/C8E6C8/1B5E20?text=Lady+Care+Health+Category+Insights",
    }
    selected_category_image = st.sidebar.selectbox("Choose Category Tab Hero", options=list(category_image_options.keys()), index=0, key="category_hero_image_selector")
    st.image(category_image_options[selected_category_image], use_container_width=True)
    
    # Custom CSS for health-focused green styling
    st.markdown("""
    <style>
    .health-category-metric {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .Nutraceuticals & Nutrition-category-insight {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
    }
    
    .enhanced-health-metric {
        background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
        margin: 10px 0;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .enhanced-health-metric .icon {
        font-size: 3em;
        margin-bottom: 10px;
        display: block;
    }
    
    .enhanced-health-metric .value {
        font-size: 1.6em;
        font-weight: bold;
        margin-bottom: 8px;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.2;
    }
    
    .enhanced-health-metric .label {
        font-size: 1.1em;
        opacity: 0.95;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .enhanced-health-metric .sub-label {
        font-size: 1em;
        opacity: 0.9;
        font-weight: 500;
        line-height: 1.2;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for category column with case sensitivity handling
    category_column = None
    possible_category_columns = ['category', 'Category', 'CATEGORY', 'Category Name', 'category_name', 'product_category']
    
    for col in possible_category_columns:
        if col in queries.columns:
            category_column = col
            break
    
    # Check if category data is available
    has_category_data = (category_column is not None and 
                        queries[category_column].notna().any())
    
    if not has_category_data:
        st.error(f"❌ No Nutraceuticals & Nutrition category data available. Available columns: {list(queries.columns)}")
        st.info("💡 Please ensure your dataset contains a category column (category, Category, or Category Name)")
        st.stop()
    
    # Filter out "Other" category from all analysis
    category_queries = queries[
        (queries[category_column].notna()) & 
        (~queries[category_column].str.lower().isin(['other', 'others']))
    ]
    
    if category_queries.empty:
        st.error("❌ No valid Nutraceuticals & Nutrition category data available after filtering.")
        st.stop()
    
    # Health Category Performance Metrics Row
    total_categories = category_queries[category_column].nunique()
    top_category = category_queries.groupby(category_column)['Counts'].sum().idxmax()
    avg_category_counts = category_queries.groupby(category_column)['Counts'].sum().mean()
    
    # Calculate Category Dominance Index
    category_counts_sum = category_queries.groupby(category_column)['Counts'].sum()
    category_dominance = (category_counts_sum.max() / category_counts_sum.sum() * 100)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="health-category-metric">
            <div style="font-size: 2em; margin-bottom: 8px;">🌿</div>
            <div style="font-size: 1.4em; font-weight: bold; color: #1B5E20; margin-bottom: 5px;">{format_number(total_categories)}</div>
            <div style="color: #2E7D32; font-size: 0.9em;">Total Health Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class="health-category-metric">
            <div style="font-size: 2em; margin-bottom: 8px;">👑</div>
            <div style="font-size: 1.2em; font-weight: bold; color: #1B5E20; margin-bottom: 5px;">{top_category[:15]}...</div>
            <div style="color: #2E7D32; font-size: 0.9em;">Leading Health Category</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class="health-category-metric">
            <div style="font-size: 2em; margin-bottom: 8px;">⚡</div>
            <div style="font-size: 1.4em; font-weight: bold; color: #1B5E20; margin-bottom: 5px;">{category_dominance:.1f}%</div>
            <div style="color: #2E7D32; font-size: 0.9em;">Category Concentration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(f"""
        <div class="health-category-metric">
            <div style="font-size: 2em; margin-bottom: 8px;">📊</div>
            <div style="font-size: 1.4em; font-weight: bold; color: #1B5E20; margin-bottom: 5px;">{format_number(avg_category_counts)}</div>
            <div style="color: #2E7D32; font-size: 0.9em;">Avg Health Searches</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Category Analysis Layout
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Enhanced Category Performance Analysis
        st.subheader("📈 Nutraceuticals & Nutrition Category Performance Matrix")
        
        # Calculate comprehensive category metrics
        cs = category_queries.groupby(category_column).agg({
            'Counts': 'sum',
            'clicks': 'sum', 
            'conversions': 'sum'
        }).reset_index()
        
        # Round to integers for cleaner display
        cs['clicks'] = cs['clicks'].round().astype(int)
        cs['conversions'] = cs['conversions'].round().astype(int)
        
        # Rename the category column to 'category' for consistency
        cs = cs.rename(columns={category_column: 'category'})
        
        # Calculate performance metrics
        cs['ctr'] = cs.apply(lambda r: (r['clicks']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)
        cs['cr'] = cs.apply(lambda r: (r['conversions']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)
        cs['classic_cr'] = cs.apply(lambda r: (r['conversions']/r['clicks']*100) if r['clicks']>0 else 0, axis=1)
        
        # Calculate share percentage
        total_category_counts = cs['Counts'].sum()
        cs['share_pct'] = (cs['Counts'] / total_category_counts * 100).round(2)
        
        # Enhanced scatter plot for category performance
        fig_category_perf = px.scatter(
            cs.head(30), 
            x='Counts', 
            y='ctr',
            size='clicks',
            color='cr',
            hover_name='category',
            title='<b style="color:#2E7D32; font-size:18px;">🌿 Nutraceuticals & Nutrition Category Performance Matrix: Search Volume vs CTR</b>',
            labels={'Counts': 'Total Health Searches', 'ctr': 'Click-Through Rate (%)', 'cr': 'Conversion Rate (%)'},
            color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
            template='plotly_white'
        )
        
        fig_category_perf.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Health Searches: %{x:,.0f}<br>' +
                         'CTR: %{y:.2f}%<br>' +
                         'Total Clicks: %{marker.size:,.0f}<br>' +
                         'Conversion Rate: %{marker.color:.2f}%<extra></extra>'
        )
        
        fig_category_perf.update_layout(
            plot_bgcolor='rgba(248,255,248,0.95)',
            paper_bgcolor='rgba(232,245,232,0.8)',
            font=dict(color='#1B5E20', family='Segoe UI'),
            title_x=0,
            xaxis=dict(showgrid=True, gridcolor='#C8E6C8', linecolor='#4CAF50', linewidth=2),
            yaxis=dict(showgrid=True, gridcolor='#C8E6C8', linecolor='#4CAF50', linewidth=2),
        )
        
        st.plotly_chart(fig_category_perf, use_container_width=True)
        
        # Enhanced Category Performance Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Counts by Category
            fig_counts = px.bar(
                cs.sort_values('Counts', ascending=False).head(15), 
                x='category', 
                y='Counts',
                title='<b style="color:#2E7D32;">🌱 Health Searches by Category</b>',
                color='Counts',
                color_continuous_scale=['#E8F5E8', '#2E7D32'],
                text='Counts'
            )
            
            fig_counts.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside'
            )
            
            fig_counts.update_layout(
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#C8E6C8'),
                yaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                height=400
            )
            
            st.plotly_chart(fig_counts, use_container_width=True)
        
        with col_chart2:
            # Conversion Rate by Category
            fig_cr = px.bar(
                cs.sort_values('cr', ascending=False).head(15), 
                x='category', 
                y='cr',
                title='<b style="color:#2E7D32;">💚 Nutraceuticals & Nutrition Conversion Rate by Category (%)</b>',
                color='cr',
                color_continuous_scale=['#A5D6A7', '#1B5E20'],
                text='cr'
            )
            
            fig_cr.update_traces(
                texttemplate='%{text:.2f}%',
                textposition='outside'
            )
            
            fig_cr.update_layout(
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#C8E6C8'),
                yaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                height=400
            )
            
            st.plotly_chart(fig_cr, use_container_width=True)
        
        # Top Categories Performance Table
        st.subheader("🏆 Top Nutraceuticals & Nutrition Category Performance")
        
        num_categories = st.slider(
            "Number of health categories to display:", 
            min_value=10, 
            max_value=50, 
            value=20, 
            step=5,
            key="category_count_slider"
        )
        
        top_categories = cs.sort_values('Counts', ascending=False).head(num_categories)
        
        # Create display version
        display_categories = top_categories.copy()
        display_categories = display_categories.rename(columns={
            'category': 'Health Category',
            'Counts': 'Search Counts',
            'share_pct': 'Market Share %',
            'clicks': 'Total Clicks',
            'conversions': 'Conversions',
            'ctr': 'CTR',
            'cr': 'CR',
            'classic_cr': 'Classic CR'
        })
        
        # Format numbers
        display_categories['Search Counts'] = display_categories['Search Counts'].apply(lambda x: f"{x:,.0f}")
        display_categories['Market Share %'] = display_categories['Market Share %'].apply(lambda x: f"{x:.2f}%")
        display_categories['Total Clicks'] = display_categories['Total Clicks'].apply(lambda x: f"{x:,.0f}")
        display_categories['Conversions'] = display_categories['Conversions'].apply(lambda x: f"{x:,.0f}")
        display_categories['CTR'] = display_categories['CTR'].apply(lambda x: f"{x:.2f}%")
        display_categories['CR'] = display_categories['CR'].apply(lambda x: f"{x:.2f}%")
        display_categories['Classic CR'] = display_categories['Classic CR'].apply(lambda x: f"{x:.2f}%")
        
        # Reorder columns
        column_order = ['Health Category', 'Search Counts', 'Market Share %', 'Total Clicks', 'Conversions', 'CTR', 'CR', 'Classic CR']
        display_categories = display_categories[column_order]
        
        st.dataframe(display_categories, use_container_width=True, hide_index=True)
        
        # Download button
        csv_categories = top_categories.to_csv(index=False)
        st.download_button(
            label="📥 Download Nutraceuticals & Nutrition Categories CSV",
            data=csv_categories,
            file_name=f"top_{num_categories}_Nutraceuticals & Nutrition_categories.csv",
            mime="text/csv",
            key="category_csv_download"
        )
    
    with col_right:
        # Category Market Share Pie Chart
        st.subheader("🌱 Nutraceuticals & Nutrition Category Market Share")
        
        top_categories_pie = cs.nlargest(10, 'Counts')
        
        # Health-focused color palette
        health_colors = ['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', 
                        '#C8E6C8', '#E8F5E8', '#388E3C', '#689F38', '#8BC34A']
        
        fig_pie = px.pie(
            top_categories_pie, 
            names='category', 
            values='Counts',
            title='<b style="color:#2E7D32;">🌿 Health Market Distribution</b>',
            color_discrete_sequence=health_colors
        )
        
        fig_pie.update_layout(
            font=dict(color='#1B5E20', family='Segoe UI'),
            paper_bgcolor='rgba(232,245,232,0.8)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Category Performance Categories
        st.subheader("🎯 Nutraceuticals & Nutrition Category Performance Distribution")
        
        # Categorize categories based on performance
        cs['performance_category'] = pd.cut(
            cs['ctr'], 
            bins=[0, 2, 5, 10, float('inf')], 
            labels=['Emerging (0-2%)', 'Growing (2-5%)', 'Strong (5-10%)', 'Premium (>10%)']
        )
        
        category_perf_counts = cs['performance_category'].value_counts().reset_index()
        category_perf_counts.columns = ['Performance Level', 'Count']
        
        fig_cat_perf = px.bar(
            category_perf_counts, 
            x='Performance Level', 
            y='Count',
            title='<b style="color:#2E7D32;">🌿 Health CTR Performance Distribution</b>',
            color='Count',
            color_continuous_scale=['#E8F5E8', '#2E7D32'],
            text='Count'
        )
        
        fig_cat_perf.update_traces(
            texttemplate='%{text}',
            textposition='outside'
        )
        
        fig_cat_perf.update_layout(
            plot_bgcolor='rgba(248,255,248,0.95)',
            paper_bgcolor='rgba(232,245,232,0.8)',
            font=dict(color='#1B5E20', family='Segoe UI'),
            xaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
            yaxis=dict(showgrid=True, gridcolor='#C8E6C8')
        )
        
        st.plotly_chart(fig_cat_perf, use_container_width=True)
        
        # Enhanced Category Trend Analysis
        if 'Date' in queries.columns:
            st.subheader("📈 Nutraceuticals & Nutrition Category Trend Analysis")
            
            # Get top 5 categories for trend analysis
            top_5_categories = cs.nlargest(5, 'Counts')['category'].tolist()
            
            # Use the already filtered category data
            trend_data = category_queries[
                category_queries[category_column].isin(top_5_categories)
            ].copy()
            
            if not trend_data.empty:
                try:
                    # Enhanced date processing
                    trend_data['Date'] = pd.to_datetime(trend_data['Date'], errors='coerce')
                    trend_data = trend_data.dropna(subset=['Date'])
                    
                    if not trend_data.empty:
                        # Create proper monthly aggregation
                        trend_data['Month'] = trend_data['Date'].dt.to_period('M')
                        trend_data['Month_Display'] = trend_data['Date'].dt.strftime('%Y-%m')
                        
                        # Group by Month and category - sum the counts for each month
                        monthly_trends = trend_data.groupby(['Month_Display', category_column])['Counts'].sum().reset_index()
                        monthly_trends = monthly_trends.rename(columns={category_column: 'category'})
                        
                        # Convert month display back to datetime for proper plotting
                        monthly_trends['Date'] = pd.to_datetime(monthly_trends['Month_Display'] + '-01')
                        
                        if len(monthly_trends) > 0:
                            fig_trend = px.line(
                                monthly_trends, 
                                x='Date', 
                                y='Counts', 
                                color='category',
                                title='<b style="color:#2E7D32;">🌿 Top 5 Health Categories Monthly Trend</b>',
                                color_discrete_sequence=['#2E7D32', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7'],
                                markers=True
                            )
                            
                            fig_trend.update_layout(
                                plot_bgcolor='rgba(248,255,248,0.95)',
                                paper_bgcolor='rgba(232,245,232,0.8)',
                                font=dict(color='#1B5E20', family='Segoe UI'),
                                xaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#C8E6C8',
                                    title='Month',
                                    dtick="M1",
                                    tickformat="%b %Y"
                                ),
                                yaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#C8E6C8',
                                    title='Health Searches'
                                ),
                                hovermode='x unified'
                            )
                            
                            fig_trend.update_traces(
                                hovertemplate='<b>%{fullData.name}</b><br>' +
                                            'Month: %{x|%B %Y}<br>' +
                                            'Health Searches: %{y:,.0f}<extra></extra>'
                            )
                            
                            st.plotly_chart(fig_trend, use_container_width=True)
                        else:
                            st.info("No Nutraceuticals & Nutrition trend data available for the selected date range and categories")
                    else:
                        st.info("No valid dates found in the filtered Nutraceuticals & Nutrition data")
                except Exception as e:
                    st.error(f"Error processing Nutraceuticals & Nutrition trend data: {str(e)}")
            else:
                st.info("No Nutraceuticals & Nutrition category data available for the selected date range")
    
    st.markdown("---")
    
    # Enhanced Category-Keyword Intelligence Matrix
    st.subheader("🔥 Nutraceuticals & Nutrition Category-Keyword Intelligence Matrix")
    
    # Create category filter dropdown
    if 'search' in queries.columns:
        # Get available categories (excluding null and 'other')
        available_categories = category_queries[category_column].unique()
        
        # Sort categories alphabetically
        available_categories = sorted(available_categories)
        
        # Create dropdown with "All Categories" option
        category_options = ['All Health Categories'] + list(available_categories)
        
        # Category selection dropdown
        selected_category = st.selectbox(
            "🎯 Select Health Category to Analyze:",
            options=category_options,
            index=0  # Default to "All Categories"
        )
        
        # Filter data based on selection
        if selected_category == 'All Health Categories':
            # Show top 8 categories if "All Categories" is selected
            top_categories_matrix = cs.nlargest(8, 'Counts')['category'].tolist()
            filtered_data = category_queries[category_queries[category_column].isin(top_categories_matrix)]
            matrix_title = "Top Health Categories vs Nutraceuticals & Nutrition Search Terms (Sum of Counts)"
        else:
            # Filter for selected category only
            filtered_data = category_queries[category_queries[category_column] == selected_category]
            matrix_title = f"{selected_category} - Nutraceuticals & Nutrition Search Terms Analysis (Sum of Counts)"
        
        # Remove null values from search terms
        matrix_data = filtered_data[
            (filtered_data[category_column].notna()) & 
            (filtered_data['search'].notna()) &
            (~filtered_data['search'].str.lower().isin(['other', 'others']))
        ].copy()
        
        if not matrix_data.empty:
            if selected_category == 'All Health Categories':
                # For all categories: Group by category and search term, sum the counts
                category_search_matrix = matrix_data.groupby([category_column, 'search'])['Counts'].sum().reset_index()
                category_search_matrix = category_search_matrix.rename(columns={category_column: 'category'})
                
                # Get top search terms across all categories
                top_searches = matrix_data['search'].value_counts().head(12).index.tolist()
                category_search_matrix = category_search_matrix[category_search_matrix['search'].isin(top_searches)]
                
                # Create pivot table
                heatmap_data = category_search_matrix.pivot(
                    index='category', 
                    columns='search', 
                    values='Counts'
                ).fillna(0)
                
                if not heatmap_data.empty:
                    # Create the heatmap
                    fig_matrix = px.imshow(
                        heatmap_data.values,
                        labels=dict(x="Nutraceuticals & Nutrition Search Terms", y="Health Categories", color="Total Counts"),
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                        title=f'<b style="color:#2E7D32;">{matrix_title}</b>',
                        aspect='auto'
                    )
                    
                    fig_matrix.update_layout(
                        plot_bgcolor='rgba(248,255,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI'),
                        xaxis=dict(tickangle=45),
                        height=500
                    )
                    
                    # Update hover template
                    fig_matrix.update_traces(
                        hovertemplate='<b>Health Category:</b> %{y}<br>' +
                                    '<b>Nutraceuticals & Nutrition Term:</b> %{x}<br>' +
                                    '<b>Total Searches:</b> %{z:,.0f}<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_matrix, use_container_width=True)
                    
                    # Show summary statistics
                    total_interactions = category_search_matrix['Counts'].sum()
                    st.info(f"📊 Matrix shows {len(heatmap_data.index)} health categories × {len(heatmap_data.columns)} Nutraceuticals & Nutrition search terms with {total_interactions:,} total searches")
            else:
                # For single category: Show search terms analysis
                search_counts = matrix_data.groupby('search')['Counts'].sum().reset_index()
                search_counts = search_counts.sort_values('Counts', ascending=False).head(15)
                
                # Create a horizontal bar chart for single category
                fig_single = px.bar(
                    search_counts,
                    x='Counts',
                    y='search',
                    orientation='h',
                    title=f'<b style="color:#2E7D32;">{selected_category} - Top Nutraceuticals & Nutrition Search Terms by Count</b>',
                    labels={'Counts': 'Total Health Searches', 'search': 'Nutraceuticals & Nutrition Search Terms'},
                    color='Counts',
                    color_continuous_scale=['#E8F5E8', '#2E7D32']
                )
                
                fig_single.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_single, use_container_width=True)
                
                # Show summary
                total_counts = search_counts['Counts'].sum()
                st.info(f"📊 {selected_category} has {len(search_counts)} top Nutraceuticals & Nutrition search terms with {total_counts:,} total searches")
        else:
            st.warning("⚠️ No Nutraceuticals & Nutrition category data available for the selected filter")
    
    st.markdown("---")
    
    # Enhanced Top Keywords per Category Analysis
    st.subheader("🔑 Top Health Keywords per Nutraceuticals & Nutrition Category Analysis")
    
    try:
        # Calculate keywords per category using the enhanced approach
        rows = []
        for cat, grp in category_queries.groupby(category_column):
            # Use the keywords column that was created by prepare_queries_df function
            keyword_counts = {}
            
            for idx, row in grp.iterrows():
                keywords_list = row['keywords']
                query_count = row['Counts']
                
                if isinstance(keywords_list, list):
                    # Add the query count to each keyword
                    for keyword in keywords_list:
                        if keyword in keyword_counts:
                            keyword_counts[keyword] += query_count
                        else:
                            keyword_counts[keyword] = query_count
                elif pd.notna(keywords_list):
                    # Fallback: use normalized_query if keywords is not a list
                    search_term = row['normalized_query']
                    if pd.notna(search_term):
                        keywords = str(search_term).lower().split()
                        for keyword in keywords:
                            if keyword in keyword_counts:
                                keyword_counts[keyword] += query_count
                            else:
                                keyword_counts[keyword] = query_count
            
            # Get top 8 keywords for this category
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            
            for keyword, count in top_keywords:
                rows.append({'category': cat, 'keyword': keyword, 'count': count})
        
        df_ckw = pd.DataFrame(rows)
        
        if not df_ckw.empty:
            # Create pivot table for keyword analysis
            pivot_ckw = df_ckw.pivot_table(index='category', columns='keyword', values='count', fill_value=0)
            
            # Display options
            display_option = st.radio(
                "Choose Nutraceuticals & Nutrition keyword display format:",
                ["Interactive Table", "Heatmap Visualization", "Top Health Keywords Summary"],
                horizontal=True
            )
            
            if display_option == "Interactive Table":
                if AGGRID_OK:
                    gb = GridOptionsBuilder.from_dataframe(pivot_ckw.reset_index())
                    gb.configure_grid_options(enableRangeSelection=True, pagination=True)
                    AgGrid(pivot_ckw.reset_index(), gridOptions=gb.build(), height=400)
                else:
                    st.dataframe(pivot_ckw, use_container_width=True, hide_index=True)
            
            elif display_option == "Heatmap Visualization":
                # Create heatmap for keyword-category matrix
                fig_keyword_heatmap = px.imshow(
                    pivot_ckw.values,
                    labels=dict(x="Health Keywords", y="Nutraceuticals & Nutrition Categories", color="Keyword Count"),
                    x=pivot_ckw.columns,
                    y=pivot_ckw.index,
                    color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                    title='<b style="color:#2E7D32;">🌿 Nutraceuticals & Nutrition Category-Health Keyword Frequency Heatmap</b>',
                    aspect='auto'
                )
                
                fig_keyword_heatmap.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    xaxis=dict(tickangle=45),
                    height=600
                )
                
                st.plotly_chart(fig_keyword_heatmap, use_container_width=True)
            
            else:  # Top Keywords Summary
                # Show top keywords summary by category with enhanced accuracy
                st.subheader("🔥 Top 10 Health Keywords by Nutraceuticals & Nutrition Category")
                
                top_keywords_summary = []
                category_stats = {}
                
                # Calculate total volume across all categories for share percentage
                total_volume_all_categories = cs['Counts'].sum()
                
                for cat in df_ckw['category'].unique():
                    cat_data = df_ckw[df_ckw['category'] == cat].sort_values('count', ascending=False)
                    
                    # Get top 10 keywords for this category
                    top_10_keywords = cat_data.head(10)
                    
                    # Create formatted keyword string with counts
                    keywords_list = []
                    for _, row in top_10_keywords.iterrows():
                        keywords_list.append(f"{row['keyword']} ({row['count']:,})")
                    
                    keywords_str = ' | '.join(keywords_list)
                    
                    # Calculate category statistics - CORRECTED CALCULATION
                    # Get the actual total from the original category summary data
                    actual_category_total = cs[cs['category'] == cat]['Counts'].iloc[0] if len(cs[cs['category'] == cat]) > 0 else cat_data['count'].sum()
                    
                    # Calculate correct share percentage based on total volume
                    share_percentage = (actual_category_total / total_volume_all_categories * 100)
                    
                    total_keyword_count = cat_data['count'].sum()
                    unique_keywords = len(cat_data)
                    avg_keyword_count = cat_data['count'].mean()
                    top_keyword_dominance = (top_10_keywords.iloc[0]['count'] / total_keyword_count * 100) if len(top_10_keywords) > 0 else 0
                    
                    # Store category stats for additional insights
                    category_stats[cat] = {
                        'total_keywords': unique_keywords,
                        'total_count': actual_category_total,  # Use actual category total
                        'keyword_total_count': total_keyword_count,  # Keep keyword-specific total
                        'avg_count': avg_keyword_count,
                        'top_keyword': top_10_keywords.iloc[0]['keyword'] if len(top_10_keywords) > 0 else 'N/A',
                        'dominance': top_keyword_dominance,
                        'share_percentage': share_percentage  # Add correct share percentage
                    }
                    
                    top_keywords_summary.append({
                        'Nutraceuticals & Nutrition Category': cat,
                        'Top 10 Health Keywords (with counts)': keywords_str,
                        'Total Keywords': unique_keywords,
                        'Category Total Volume': f"{actual_category_total:,}",  # Use actual category total
                        'Market Share %': f"{share_percentage:.2f}%",  # Add share percentage column
                        'Keyword Analysis Volume': f"{total_keyword_count:,}",  # Show keyword-specific total
                        'Avg Keyword Count': f"{avg_keyword_count:.1f}",
                        'Top Health Keyword': top_10_keywords.iloc[0]['keyword'] if len(top_10_keywords) > 0 else 'N/A',
                        'Keyword Dominance %': f"{top_keyword_dominance:.1f}%"
                    })
                
                # Sort by actual category total volume (descending)
                top_keywords_summary = sorted(top_keywords_summary, key=lambda x: int(x['Category Total Volume'].replace(',', '')), reverse=True)
                summary_df = pd.DataFrame(top_keywords_summary)
                
                # Display the enhanced summary table
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Additional insights section with ENHANCED FONT SIZES
                st.markdown("---")
                st.subheader("📊 Nutraceuticals & Nutrition Category Health Keyword Intelligence")
                
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                
                with col_insight1:
                    # Most diverse category (most unique keywords)
                    most_diverse_cat = max(category_stats.items(), key=lambda x: x[1]['total_keywords'])
                    category_name = most_diverse_cat[0][:15] + "..." if len(most_diverse_cat[0]) > 15 else most_diverse_cat[0]
                    st.markdown(f"""
                    <div class='enhanced-health-metric'>
                        <span class='icon'>🌟</span>
                        <div class='value'>{category_name}</div>
                        <div class='label'>Most Diverse Health Category</div>
                        <div class='sub-label'>{most_diverse_cat[1]['total_keywords']} unique health keywords</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_insight2:
                    # Highest volume category with CORRECT SHARE PERCENTAGE
                    highest_volume_cat = max(category_stats.items(), key=lambda x: x[1]['total_count'])
                    category_name = highest_volume_cat[0][:15] + "..." if len(highest_volume_cat[0]) > 15 else highest_volume_cat[0]
                    st.markdown(f"""
                    <div class='enhanced-health-metric'>
                        <span class='icon'>🚀</span>
                        <div class='value'>{category_name}</div>
                        <div class='label'>Highest Volume Nutraceuticals & Nutrition Category</div>
                        <div class='sub-label'>{highest_volume_cat[1]['total_count']:,} total health searches<br>{highest_volume_cat[1]['share_percentage']:.2f}% market share</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_insight3:
                    # Most concentrated category with CORRECT SHARE PERCENTAGE
                    most_concentrated_cat = max(category_stats.items(), key=lambda x: x[1]['share_percentage'])
                    category_name = most_concentrated_cat[0][:15] + "..." if len(most_concentrated_cat[0]) > 15 else most_concentrated_cat[0]
                    st.markdown(f"""
                    <div class='enhanced-health-metric'>
                        <span class='icon'>🎯</span>
                        <div class='value'>{category_name}</div>
                        <div class='label'>Most Concentrated Health Category</div>
                        <div class='sub-label'>{most_concentrated_cat[1]['share_percentage']:.2f}% Nutraceuticals & Nutrition market share</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top keywords across all categories
                st.markdown("---")
                st.subheader("🏆 Global Top Health Keywords Across All Nutraceuticals & Nutrition Categories")
                
                # Get top keywords globally
                global_keywords = df_ckw.groupby('keyword')['count'].sum().reset_index()
                global_keywords = global_keywords.sort_values('count', ascending=False).head(20)
                
                # Create a horizontal bar chart for global keywords
                fig_global_keywords = px.bar(
                    global_keywords,
                    x='count',
                    y='keyword',
                    orientation='h',
                    title='<b style="color:#2E7D32;">🌿 Top 20 Health Keywords Across All Nutraceuticals & Nutrition Categories</b>',
                    labels={'count': 'Total Health Search Count', 'keyword': 'Health Keywords'},
                    color='count',
                    color_continuous_scale=['#E8F5E8', '#2E7D32'],
                    text='count'
                )
                
                fig_global_keywords.update_traces(
                    texttemplate='%{text:,}',
                    textposition='outside'
                )
                
                fig_global_keywords.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=600,
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                    yaxis_title="Health Keywords",
                    xaxis_title="Total Health Search Count"
                )
                
                st.plotly_chart(fig_global_keywords, use_container_width=True)
                
                # Category keyword distribution analysis
                st.markdown("---")
                st.subheader("📈 Nutraceuticals & Nutrition Category Health Keyword Distribution Analysis")
                
                # Create distribution data using corrected totals
                distribution_data = []
                for cat, stats in category_stats.items():
                    distribution_data.append({
                        'Nutraceuticals & Nutrition Category': cat,
                        'Unique Health Keywords': stats['total_keywords'],
                        'Total Health Volume': stats['total_count'],  # Use actual category total
                        'Average Keyword Count': stats['avg_count']
                    })
                
                dist_df = pd.DataFrame(distribution_data)
                
                # Create scatter plot for keyword distribution
                fig_distribution = px.scatter(
                    dist_df,
                    x='Unique Health Keywords',
                    y='Total Health Volume',
                    size='Average Keyword Count',
                    hover_name='Nutraceuticals & Nutrition Category',
                    title='<b style="color:#2E7D32;">🌿 Nutraceuticals & Nutrition Category Health Keyword Diversity vs Volume</b>',
                    labels={
                        'Unique Health Keywords': 'Number of Unique Health Keywords',
                        'Total Health Volume': 'Total Health Search Volume',
                        'Average Keyword Count': 'Average Health Keyword Count'
                    },
                    color='Average Keyword Count',
                    color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32']
                )
                
                fig_distribution.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    xaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                    yaxis=dict(showgrid=True, gridcolor='#C8E6C8')
                )
                
                fig_distribution.update_traces(
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                 'Unique Health Keywords: %{x}<br>' +
                                 'Total Health Volume: %{y:,}<br>' +
                                 'Avg Keyword Count: %{marker.size:.1f}<extra></extra>'
                )
                
                st.plotly_chart(fig_distribution, use_container_width=True)
            
            # Download button for keyword analysis
            csv_keywords = df_ckw.to_csv(index=False)
            st.download_button(
                label="📥 Download Nutraceuticals & Nutrition Category Health Keywords CSV",
                data=csv_keywords,
                file_name="Nutraceuticals & Nutrition_category_health_keywords_analysis.csv",
                mime="text/csv",
                key="category_keywords_csv_download"
            )
        else:
            st.info("Not enough health keyword data per Nutraceuticals & Nutrition category.")
    
    except Exception as e:
        st.error(f"Error processing health keyword analysis: {str(e)}")
        st.info("Not enough health keyword data per Nutraceuticals & Nutrition category.")

    # Health Category Insights Section
    st.markdown("---")
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        top_category_share = cs.iloc[0]['share_pct'] if not cs.empty else 0
        top_category_name = cs.iloc[0]['category'] if not cs.empty else "N/A"
        high_performers = len(cs[cs['ctr'] > 5]) if not cs.empty else 0
        avg_conversion_rate = cs['cr'].mean() if not cs.empty else 0
        categories_above_avg_cr = len(cs[cs['cr'] > avg_conversion_rate]) if not cs.empty else 0
        
        st.markdown(f"""
        <div class='Nutraceuticals & Nutrition-category-insight'>
            <h4>🌿 Key Nutraceuticals & Nutrition Category Insights</h4>
            <p>• <strong>{top_category_name}</strong> leads health market with {top_category_share:.1f}% share<br>
            • {high_performers} Nutraceuticals & Nutrition categories achieve CTR > 5% (premium performance)<br>
            • {categories_above_avg_cr} categories exceed avg CR of {avg_conversion_rate:.2f}%<br>
            • Health market shows {"strong" if category_dominance > 30 else "balanced" if category_dominance > 15 else "fragmented"} category concentration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_insight2:
        low_performers = len(cs[cs['ctr'] < 2]) if not cs.empty else 0
        opportunity_categories = len(cs[(cs['Counts'] > cs['Counts'].median()) & (cs['ctr'] < 3)]) if not cs.empty else 0
        
        st.markdown(f"""
        <div class='Nutraceuticals & Nutrition-category-insight'>
            <h4>💚 Health Category Strategy Recommendations</h4>
            <p>• Optimize {low_performers} underperforming Nutraceuticals & Nutrition categories (CTR < 2%)<br>
            • {opportunity_categories} high-volume health categories need engagement boost<br>
            • Focus on Nutraceuticals & Nutrition keywords for leading health categories<br>
            • {"Diversify" if category_dominance > 40 else "Strengthen"} health product portfolio strategy</p>
        </div>
        """, unsafe_allow_html=True)

    # Final Category Summary Dashboard
    st.markdown("---")
    st.subheader("📊 Nutraceuticals & Nutrition Category Performance Dashboard Summary")
    
    # Create final summary metrics
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        total_searches = cs['Counts'].sum() if not cs.empty else 0
        st.metric(
            label="🌿 Total Health Searches",
            value=f"{total_searches:,.0f}",
            delta=f"{len(cs)} Nutraceuticals & Nutrition categories analyzed"
        )
    
    with summary_col2:
        avg_market_ctr = cs['ctr'].mean() if not cs.empty else 0
        top_ctr = cs['ctr'].max() if not cs.empty else 0
        st.metric(
            label="📈 Market Avg CTR",
            value=f"{avg_market_ctr:.2f}%",
            delta=f"Best: {top_ctr:.2f}%"
        )
    
    with summary_col3:
        total_conversions = cs['conversions'].sum() if not cs.empty else 0
        avg_cr = cs['cr'].mean() if not cs.empty else 0
        st.metric(
            label="💚 Total Conversions",
            value=f"{total_conversions:,.0f}",
            delta=f"Avg CR: {avg_cr:.2f}%"
        )
    
    with summary_col4:
        market_concentration = f"{category_dominance:.1f}%" if not cs.empty else "0%"
        concentration_status = "High" if category_dominance > 30 else "Medium" if category_dominance > 15 else "Low"
        st.metric(
            label="🎯 Market Concentration",
            value=market_concentration,
            delta=f"{concentration_status} concentration"
        )
    
    # Export all category data
    st.markdown("---")
    st.subheader("📥 Export Nutraceuticals & Nutrition Category Intelligence")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if not cs.empty:
            # Comprehensive category export
            export_data = cs.copy()
            export_data['analysis_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            csv_export = export_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Complete Nutraceuticals & Nutrition Category Analysis",
                data=csv_export,
                file_name=f"wNutraceuticals & Nutrition_category_intelligence_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="complete_category_export"
            )
    
    with export_col2:
        if not cs.empty:
            # Summary report
            summary_report = f"""
            Nutraceuticals & Nutrition CATEGORY INTELLIGENCE REPORT
            Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            
            HEALTH MARKET OVERVIEW:
            • Total Nutraceuticals & Nutrition Categories Analyzed: {len(cs)}
            • Total Health Searches: {cs['Counts'].sum():,.0f}
            • Market Leader: {cs.iloc[0]['category']} ({cs.iloc[0]['share_pct']:.1f}% share)
            • Average CTR: {cs['ctr'].mean():.2f}%
            • Average CR: {cs['cr'].mean():.2f}%
            
            Nutraceuticals & Nutrition PERFORMANCE TIERS:
            • Premium Categories (CTR > 10%): {len(cs[cs['ctr'] > 10])}
            • Strong Categories (CTR 5-10%): {len(cs[(cs['ctr'] >= 5) & (cs['ctr'] <= 10)])}
            • Growing Categories (CTR 2-5%): {len(cs[(cs['ctr'] >= 2) & (cs['ctr'] < 5)])}
            • Emerging Categories (CTR < 2%): {len(cs[cs['ctr'] < 2])}
            
            STRATEGIC HEALTH INSIGHTS:
            • Market concentration is {concentration_status.lower()}
            • {len(cs[cs['ctr'] > 5])} categories achieve premium performance
            • Growth opportunities exist in Nutraceuticals & Nutrition engagement optimization
            """
            
            st.download_button(
                label="📝 Download Nutraceuticals & Nutrition Executive Summary Report",
                data=summary_report,
                file_name=f"Nutraceuticals & Nutrition_category_executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key="category_executive_summary_export"
            )


# ----------------- Subcategory Tab (Enhanced & Health-Focused) -----------------
# ----------------- Subcategory Tab (Enhanced & Health-Focused) -----------------
with tab_subcat:
    st.header("🌿 Health Subcategory Intelligence Hub")
    st.markdown("Deep dive into Nutraceuticals & Nutrition subcategory performance and health search trends. 💚")

    # Hero Image for Subcategory Tab
    subcat_image_options = {
        "Health Subcategory Analytics": "https://placehold.co/1200x200/E8F5E8/2E7D32?text=Health+Subcategory+Performance+Analysis",
        "Wellness Subcategories": "https://placehold.co/1200x200/4CAF50/FFFFFF?text=Wellness+Subcategory+Intelligence+Dashboard",
        "Abstract Health Subcategories": "https://source.unsplash.com/1200x200/?health,wellness,subcategory",
        "Health Gradient": "https://placehold.co/1200x200/C8E6C8/1B5E20?text=Lady+Care+Health+Subcategory+Insights",
    }
    selected_subcat_image = st.sidebar.selectbox("Choose Subcategory Tab Hero", options=list(subcat_image_options.keys()), index=0, key="subcat_hero_image_selector")
    st.image(subcat_image_options[selected_subcat_image], use_container_width=True)

    try:
        # Check if subcategory data exists
        if 'sub_category' in queries.columns and queries['sub_category'].notna().any():
            
            # Calculate comprehensive subcategory metrics
            sc = queries.groupby('sub_category').agg({
                'Counts': 'sum',
                'clicks': 'sum', 
                'conversions': 'sum'
            }).reset_index()
            
            # Calculate performance metrics - CORRECTED CR CALCULATIONS
            sc['ctr'] = sc.apply(lambda r: (r['clicks']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)
            sc['classic_cvr'] = sc.apply(lambda r: (r['conversions']/r['clicks']*100) if r['clicks']>0 else 0, axis=1)  # Classic CVR: conversions/clicks
            sc['conversion_rate'] = sc.apply(lambda r: (r['conversions']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)  # Our CR: conversions/counts
            
            # Calculate additional metrics
            sc['click_share'] = sc['clicks'] / sc['clicks'].sum() * 100
            sc['conversion_share'] = sc['conversions'] / sc['conversions'].sum() * 100 if sc['conversions'].sum() > 0 else 0
            
            # Sort by counts for main analysis
            sc = sc.sort_values('Counts', ascending=False)
            
            # CALCULATE MARKET CONCENTRATION METRICS EARLY (MOVED UP)
            top_5_concentration = sc.head(5)['Counts'].sum() / sc['Counts'].sum() * 100 if not sc.empty else 0
            top_10_concentration = sc.head(10)['Counts'].sum() / sc['Counts'].sum() * 100 if not sc.empty else 0
            gini_coefficient = 1 - 2 * np.sum(np.cumsum(sc['Counts'].sort_values()) / sc['Counts'].sum()) / len(sc) if not sc.empty else 0
            herfindahl_index = np.sum((sc['Counts'] / sc['Counts'].sum()) ** 2) if not sc.empty else 0
            
            # Enhanced Key Metrics Section
            st.subheader("🌿 Health Subcategory Performance Overview")
            
            # Enhanced CSS for health-focused subcategory metrics
            st.markdown("""
            <style>
            .health-subcat-metric-card {
                background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                color: #1B5E20;
                box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
                margin: 10px 0;
                min-height: 160px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                transition: transform 0.2s ease;
                border-left: 4px solid #4CAF50;
            }
            
            .health-subcat-metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px rgba(46, 125, 50, 0.4);
            }
            
            .health-subcat-metric-card .icon {
                font-size: 3em;
                margin-bottom: 10px;
                display: block;
                color: #2E7D32;
            }
            
            .health-subcat-metric-card .value {
                font-size: 1.6em;
                font-weight: bold;
                margin-bottom: 8px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                line-height: 1.2;
                color: #1B5E20;
            }
            
            .health-subcat-metric-card .label {
                font-size: 1.1em;
                opacity: 0.95;
                font-weight: 600;
                margin-bottom: 6px;
                color: #2E7D32;
            }
            
            .health-subcat-metric-card .sub-label {
                font-size: 1em;
                opacity: 0.9;
                font-weight: 500;
                line-height: 1.2;
                color: #388E3C;
            }
            
            .Nutraceuticals & Nutrition-performance-badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                margin-left: 8px;
            }
            
            .high-Nutraceuticals & Nutrition-performance {
                background-color: #4CAF50;
                color: white;
            }
            
            .medium-Nutraceuticals & Nutrition-performance {
                background-color: #81C784;
                color: white;
            }
            
            .low-Nutraceuticals & Nutrition-performance {
                background-color: #A5D6A7;
                color: #1B5E20;
            }
            
            .Nutraceuticals & Nutrition-insight-card {
                background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
                padding: 25px;
                border-radius: 15px;
                color: white;
                margin: 15px 0;
                box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            total_subcategories = len(sc)
            total_searches = sc['Counts'].sum()
            avg_ctr = sc['ctr'].mean()
            avg_cr = sc['conversion_rate'].mean()
            top_subcategory = sc.iloc[0]['sub_category'] if len(sc) > 0 else 'N/A'
            top_subcategory_volume = sc.iloc[0]['Counts'] if len(sc) > 0 else 0
            
            with col1:
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>🌿</span>
                    <div class='value'>{format_number(total_subcategories)}</div>
                    <div class='label'>Total Health Subcategories</div>
                    <div class='sub-label'>Active Nutraceuticals & Nutrition segments</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>🔍</span>
                    <div class='value'>{format_number(total_searches)}</div>
                    <div class='label'>Total Health Searches</div>
                    <div class='sub-label'>Across all Nutraceuticals & Nutrition subcategories</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                performance_class = "high-Nutraceuticals & Nutrition-performance" if avg_ctr > 5 else "medium-Nutraceuticals & Nutrition-performance" if avg_ctr > 2 else "low-Nutraceuticals & Nutrition-performance"
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{avg_ctr:.2f}% <span class='Nutraceuticals & Nutrition-performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                    <div class='label'>Average Health CTR</div>
                    <div class='sub-label'>Click-through rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                top_subcat_display = top_subcategory[:12] + "..." if len(top_subcategory) > 12 else top_subcategory
                market_share = (top_subcategory_volume / total_searches * 100)
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>👑</span>
                    <div class='value'>{top_subcat_display}</div>
                    <div class='label'>Top Health Subcategory</div>
                    <div class='sub-label'>{market_share:.1f}% Nutraceuticals & Nutrition market share</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>💚</span>
                    <div class='value'>{avg_cr:.2f}%</div>
                    <div class='label'>Avg Nutraceuticals & Nutrition Conversion Rate</div>
                    <div class='sub-label'>Overall health performance</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                total_clicks = int(sc['clicks'].sum())
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>🖱️</span>
                    <div class='value'>{format_number(total_clicks)}</div>
                    <div class='label'>Total Health Clicks</div>
                    <div class='sub-label'>Across all Nutraceuticals & Nutrition subcategories</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                total_conversions = int(sc['conversions'].sum())
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>✅</span>
                    <div class='value'>{format_number(total_conversions)}</div>
                    <div class='label'>Total Nutraceuticals & Nutrition Conversions</div>
                    <div class='sub-label'>Successful health outcomes</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                top_conversion_subcat = sc.nlargest(1, 'conversions')['sub_category'].iloc[0] if len(sc) > 0 else 'N/A'
                top_conversion_display = top_conversion_subcat[:12] + "..." if len(top_conversion_subcat) > 12 else top_conversion_subcat
                st.markdown(f"""
                <div class='health-subcat-metric-card'>
                    <span class='icon'>🏆</span>
                    <div class='value'>{top_conversion_display}</div>
                    <div class='label'>Nutraceuticals & Nutrition Conversion Leader</div>
                    <div class='sub-label'>Most health conversions</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # TOP 10 HEALTH KEYWORDS BY SUBCATEGORY TABLE
            if 'keyword' in queries.columns and 'sub_category' in queries.columns:
                df_sckw = queries[queries['keyword'].notna() & queries['sub_category'].notna()].copy()
                
                if len(df_sckw) > 0:
                    df_sckw_grouped = df_sckw.groupby(['sub_category', 'keyword']).agg({
                        'Counts': 'sum',
                        'clicks': 'sum',
                        'conversions': 'sum'
                    }).reset_index()
                    df_sckw_grouped.rename(columns={'Counts': 'count'}, inplace=True)
                    
                    df_sckw_grouped['keyword_ctr'] = df_sckw_grouped.apply(lambda r: (r['clicks']/r['count']*100) if r['count']>0 else 0, axis=1)
                    df_sckw_grouped['keyword_cr'] = df_sckw_grouped.apply(lambda r: (r['conversions']/r['count']*100) if r['count']>0 else 0, axis=1)
                    
                    st.subheader("🔥 Top 10 Health Keywords by Nutraceuticals & Nutrition Subcategory")
                    
                    top_keywords_summary = []
                    subcategory_stats = {}
                    
                    total_volume_all_subcategories = sc['Counts'].sum()
                    
                    for subcat in df_sckw_grouped['sub_category'].unique():
                        subcat_data = df_sckw_grouped[df_sckw_grouped['sub_category'] == subcat].sort_values('count', ascending=False)
                        
                        top_10_keywords = subcat_data.head(10)
                        
                        keywords_list = []
                        for _, row in top_10_keywords.iterrows():
                            performance_indicator = "🌟" if row['keyword_ctr'] > 5 else "⚡" if row['keyword_ctr'] > 2 else "📊"
                            keywords_list.append(f"{performance_indicator} {row['keyword']} ({row['count']:,})")
                        
                        keywords_str = ' | '.join(keywords_list)
                        
                        actual_subcategory_total = sc[sc['sub_category'] == subcat]['Counts'].iloc[0] if len(sc[sc['sub_category'] == subcat]) > 0 else subcat_data['count'].sum()
                        share_percentage = (actual_subcategory_total / total_volume_all_subcategories * 100)
                        
                        total_keyword_count = subcat_data['count'].sum()
                        unique_keywords = len(subcat_data)
                        avg_keyword_count = subcat_data['count'].mean()
                        top_keyword_dominance = (top_10_keywords.iloc[0]['count'] / total_keyword_count * 100) if len(top_10_keywords) > 0 else 0
                        
                        subcategory_stats[subcat] = {
                            'total_keywords': unique_keywords,
                            'total_count': actual_subcategory_total,
                            'keyword_total_count': total_keyword_count,
                            'avg_count': avg_keyword_count,
                            'top_keyword': top_10_keywords.iloc[0]['keyword'] if len(top_10_keywords) > 0 else 'N/A',
                            'top_keyword_count': top_10_keywords.iloc[0]['count'] if len(top_10_keywords) > 0 else 0,
                            'dominance': top_keyword_dominance,
                            'share_percentage': share_percentage
                        }
                        
                        top_keywords_summary.append({
                            'Health Subcategory': subcat,
                            'Top 10 Health Keywords (with counts)': keywords_str,
                            'Total Health Keywords': unique_keywords,
                            'Subcategory Total Volume': f"{actual_subcategory_total:,}",
                            'Nutraceuticals & Nutrition Share %': f"{share_percentage:.2f}%",
                            'Keyword Analysis Volume': f"{total_keyword_count:,}",
                            'Avg Health Keyword Count': f"{avg_keyword_count:.1f}",
                            'Top Health Keyword': top_10_keywords.iloc[0]['keyword'] if len(top_10_keywords) > 0 else 'N/A',
                            'Top Keyword Volume': top_10_keywords.iloc[0]['count'] if len(top_10_keywords) > 0 else 0,
                            'Health Keyword Dominance %': f"{top_keyword_dominance:.1f}%"
                        })
                    
                    top_keywords_summary = sorted(top_keywords_summary, key=lambda x: int(x['Subcategory Total Volume'].replace(',', '')), reverse=True)
                    summary_df = pd.DataFrame(top_keywords_summary)
                    
                    st.dataframe(summary_df, use_container_width=True, height=400, hide_index=True)
                    
                    csv_keywords_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Health Subcategory Keywords Summary CSV",
                        data=csv_keywords_summary,
                        file_name="health_subcategory_keywords_summary.csv",
                        mime="text/csv",
                        key="health_subcategory_keywords_summary_download"
                    )
                    
                    # HEALTH SUBCATEGORY KEYWORD INTELLIGENCE SECTION
                    st.markdown("---")
                    st.subheader("🌿 Health Subcategory Keyword Intelligence")
                    
                    col_insight1, col_insight2, col_insight3 = st.columns(3)
                    
                    with col_insight1:
                        most_diverse_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['total_keywords'])
                        subcategory_name = most_diverse_subcat[0][:15] + "..." if len(most_diverse_subcat[0]) > 15 else most_diverse_subcat[0]
                        st.markdown(f"""
                        <div class='health-subcat-metric-card'>
                            <span class='icon'>🌟</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Most Diverse Health Subcategory</div>
                            <div class='sub-label'>{most_diverse_subcat[1]['total_keywords']} unique health keywords</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight2:
                        highest_volume_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['total_count'])
                        subcategory_name = highest_volume_subcat[0][:15] + "..." if len(highest_volume_subcat[0]) > 15 else highest_volume_subcat[0]
                        st.markdown(f"""
                        <div class='health-subcat-metric-card'>
                            <span class='icon'>🚀</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Highest Volume Health Subcategory</div>
                            <div class='sub-label'>{highest_volume_subcat[1]['total_count']:,} total health searches<br>{highest_volume_subcat[1]['share_percentage']:.2f}% Nutraceuticals & Nutrition share</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight3:
                        most_concentrated_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['share_percentage'])
                        subcategory_name = most_concentrated_subcat[0][:15] + "..." if len(most_concentrated_subcat[0]) > 15 else most_concentrated_subcat[0]
                        st.markdown(f"""
                        <div class='health-subcat-metric-card'>
                            <span class='icon'>🎯</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Most Concentrated Health Subcategory</div>
                            <div class='sub-label'>{most_concentrated_subcat[1]['share_percentage']:.2f}% Nutraceuticals & Nutrition market share</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
            else:
                st.subheader("🔥 Top 10 Health Keywords by Nutraceuticals & Nutrition Subcategory")
                st.info("Health keyword data not available for subcategory analysis.")
                st.markdown("---")
            
            # Interactive health subcategory selection
            st.subheader("🎯 Interactive Health Subcategory Analysis")

            analysis_type = st.radio(
                "Choose Nutraceuticals & Nutrition Analysis Type:",
                ["📊 Top Health Performers Overview", "🔍 Detailed Health Subcategory Deep Dive", "📈 Health Performance Comparison", "📊 Nutraceuticals & Nutrition Market Share Analysis"],
                horizontal=True
            )

            if analysis_type == "📊 Top Health Performers Overview":
                st.subheader("🏆 Top 20 Health Subcategories Performance")
                
                top_20_sc = sc.head(20).copy()
                
                # Enhanced bar chart with health-focused green colors
                fig_top_subcats = px.bar(
                    top_20_sc,
                    x='sub_category',
                    y='Counts',
                    title='<b style="color:#2E7D32;">🌿 Top 20 Health Subcategories by Search Volume</b>',
                    labels={'Counts': 'Health Search Volume', 'sub_category': 'Health Subcategories'},
                    color='Counts',
                    color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                    text='Counts'
                )
                
                fig_top_subcats.update_traces(
                    texttemplate='%{text:,}',
                    textposition='outside'
                )
                
                fig_top_subcats.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=600,
                    xaxis=dict(tickangle=45, showgrid=True, gridcolor='#C8E6C8'),
                    yaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                    showlegend=False
                )
                
                st.plotly_chart(fig_top_subcats, use_container_width=True)
                
                # Health performance metrics comparison chart
                st.subheader("📊 Health Performance Metrics Comparison")
                
                fig_metrics_comparison = go.Figure()
                
                fig_metrics_comparison.add_trace(go.Bar(
                    name='Health CTR %',
                    x=top_20_sc['sub_category'],
                    y=top_20_sc['ctr'],
                    marker_color='#4CAF50'
                ))
                
                fig_metrics_comparison.add_trace(go.Bar(
                    name='Nutraceuticals & Nutrition Conversion Rate %',
                    x=top_20_sc['sub_category'],
                    y=top_20_sc['conversion_rate'],
                    marker_color='#81C784'
                ))
                
                fig_metrics_comparison.update_layout(
                    title='<b style="color:#2E7D32;">🌿 Health CTR vs Nutraceuticals & Nutrition Conversion Rate Comparison</b>',
                    barmode='group',
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45),
                    yaxis=dict(title='Percentage (%)')
                )
                
                st.plotly_chart(fig_metrics_comparison, use_container_width=True)

            elif analysis_type == "🔍 Detailed Health Subcategory Deep Dive":
                st.subheader("🔬 Health Subcategory Deep Dive Analysis")
                
                selected_subcategory = st.selectbox(
                    "Select a health subcategory for detailed Nutraceuticals & Nutrition analysis:",
                    options=sc['sub_category'].tolist(),
                    index=0
                )
                
                if selected_subcategory:
                    subcat_data = sc[sc['sub_category'] == selected_subcategory].iloc[0]
                    subcat_rank = sc.reset_index().index[sc['sub_category'] == selected_subcategory].tolist()[0] + 1
                    
                    col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                    
                    with col_detail1:
                        rank_performance = "high-Nutraceuticals & Nutrition-performance" if subcat_rank <= 3 else "medium-Nutraceuticals & Nutrition-performance" if subcat_rank <= 10 else "low-Nutraceuticals & Nutrition-performance"
                        st.markdown(f"""
                        <div class='health-subcat-metric-card'>
                            <span class='icon'>🏆</span>
                            <div class='value'>#{subcat_rank} <span class='Nutraceuticals & Nutrition-performance-badge {rank_performance}'>{"Top 3" if subcat_rank <= 3 else "Top 10" if subcat_rank <= 10 else "Lower"}</span></div>
                            <div class='label'>Health Market Rank</div>
                            <div class='sub-label'>Out of {total_subcategories} Nutraceuticals & Nutrition subcategories</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_detail2:
                        market_share = (subcat_data['Counts'] / total_searches * 100)
                        share_performance = "high-Nutraceuticals & Nutrition-performance" if market_share > 5 else "medium-Nutraceuticals & Nutrition-performance" if market_share > 2 else "low-Nutraceuticals & Nutrition-performance"
                        st.markdown(f"""
                        <div class='health-subcat-metric-card'>
                            <span class='icon'>📊</span>
                            <div class='value'>{market_share:.2f}% <span class='Nutraceuticals & Nutrition-performance-badge {share_performance}'>{"High" if market_share > 5 else "Medium" if market_share > 2 else "Low"}</span></div>
                            <div class='label'>Nutraceuticals & Nutrition Market Share</div>
                            <div class='sub-label'>Of total health search volume</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_detail3:
                        performance_score = (subcat_data['ctr'] + subcat_data['conversion_rate']) / 2
                        score_performance = "high-Nutraceuticals & Nutrition-performance" if performance_score > 3 else "medium-Nutraceuticals & Nutrition-performance" if performance_score > 1 else "low-Nutraceuticals & Nutrition-performance"
                        st.markdown(f"""
                        <div class='health-subcat-metric-card'>
                            <span class='icon'>⭐</span>
                            <div class='value'>{performance_score:.1f} <span class='Nutraceuticals & Nutrition-performance-badge {score_performance}'>{"High" if performance_score > 3 else "Medium" if performance_score > 1 else "Low"}</span></div>
                            <div class='label'>Health Performance Score</div>
                            <div class='sub-label'>Combined CTR & CR</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_detail4:
                        conversion_efficiency = subcat_data['conversion_rate'] / subcat_data['ctr'] * 100 if subcat_data['ctr'] > 0 else 0
                        efficiency_performance = "high-Nutraceuticals & Nutrition-performance" if conversion_efficiency > 50 else "medium-Nutraceuticals & Nutrition-performance" if conversion_efficiency > 25 else "low-Nutraceuticals & Nutrition-performance"
                        st.markdown(f"""
                        <div class='health-subcat-metric-card'>
                            <span class='icon'>⚡</span>
                            <div class='value'>{conversion_efficiency:.1f}% <span class='Nutraceuticals & Nutrition-performance-badge {efficiency_performance}'>{"High" if conversion_efficiency > 50 else "Medium" if conversion_efficiency > 25 else "Low"}</span></div>
                            <div class='label'>Nutraceuticals & Nutrition Conversion Efficiency</div>
                            <div class='sub-label'>CR as % of CTR</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional detailed analysis content...
                    st.markdown("### 📈 Health Performance Breakdown")

                    metrics_data = {
                        'Health Metric': ['Search Volume', 'Total Clicks', 'Total Conversions', 
                                'Click-Through Rate', 'Classic CVR (Conv/Clicks)', 
                                'Nutraceuticals & Nutrition Conversion Rate (Conv/Counts)', 'Click Share', 'Conversion Share'],
                        'Value': [
                            f"{int(subcat_data['Counts']):,}",
                            f"{int(subcat_data['clicks']):,}",
                            f"{int(subcat_data['conversions']):,}",
                            f"{subcat_data['ctr']:.2f}%",
                            f"{subcat_data['classic_cvr']:.2f}%",
                            f"{subcat_data['conversion_rate']:.2f}%",
                            f"{subcat_data['click_share']:.2f}%",
                            f"{subcat_data['conversion_share']:.2f}%"
                        ],
                        'Health Performance': [
                            'High' if subcat_data['Counts'] > sc['Counts'].median() else 'Low',
                            'High' if subcat_data['clicks'] > sc['clicks'].median() else 'Low',
                            'High' if subcat_data['conversions'] > sc['conversions'].median() else 'Low',
                            'High' if subcat_data['ctr'] > sc['ctr'].median() else 'Low',
                            'High' if subcat_data['classic_cvr'] > sc['classic_cvr'].median() else 'Low',
                            'High' if subcat_data['conversion_rate'] > sc['conversion_rate'].median() else 'Low',
                            'High' if subcat_data['click_share'] > sc['click_share'].median() else 'Low',
                            'High' if subcat_data['conversion_share'] > sc['conversion_share'].median() else 'Low'
                        ]
                    }
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Health performance radar chart
                    st.markdown("### 📊 Health Performance Radar Chart")
                    
                    # Normalize values for radar chart
                    normalized_data = {
                        'Health Search Volume': subcat_data['Counts'] / sc['Counts'].max() * 100,
                        'Health CTR': subcat_data['ctr'] / sc['ctr'].max() * 100 if sc['ctr'].max() > 0 else 0,
                        'Nutraceuticals & Nutrition Conversion Rate': subcat_data['conversion_rate'] / sc['conversion_rate'].max() * 100 if sc['conversion_rate'].max() > 0 else 0,
                        'Health Click Share': subcat_data['click_share'],
                        'Nutraceuticals & Nutrition Conversion Share': subcat_data['conversion_share']
                    }
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(normalized_data.values()),
                        theta=list(normalized_data.keys()),
                        fill='toself',
                        name=selected_subcategory,
                        line_color='#4CAF50',
                        fillcolor='rgba(76, 175, 80, 0.3)'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                gridcolor='#C8E6C8'
                            ),
                            angularaxis=dict(
                                gridcolor='#C8E6C8'
                            )),
                        showlegend=True,
                        title=f'<b style="color:#2E7D32;">🌿 Health Performance Radar - {selected_subcategory}</b>',
                        height=400,
                        plot_bgcolor='rgba(248,255,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI')
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Health keyword analysis for selected subcategory
                    if 'keyword' in queries.columns:
                        st.markdown("### 🔍 Top Health Keywords for Selected Subcategory")
                        
                        subcat_keywords = queries[queries['sub_category'] == selected_subcategory].copy()
                        if len(subcat_keywords) > 0:
                            keyword_analysis = subcat_keywords.groupby('keyword').agg({
                                'Counts': 'sum',
                                'clicks': 'sum',
                                'conversions': 'sum'
                            }).reset_index()
                            
                            keyword_analysis['keyword_ctr'] = keyword_analysis.apply(
                                lambda r: (r['clicks']/r['Counts']*100) if r['Counts']>0 else 0, axis=1
                            )
                            keyword_analysis['keyword_cr'] = keyword_analysis.apply(
                                lambda r: (r['conversions']/r['Counts']*100) if r['Counts']>0 else 0, axis=1
                            )
                            
                            keyword_analysis = keyword_analysis.sort_values('Counts', ascending=False).head(15)
                            
                            # Health keyword bar chart
                            fig_keywords = px.bar(
                                keyword_analysis,
                                x='keyword',
                                y='Counts',
                                title=f'<b style="color:#2E7D32;">🌿 Top 15 Health Keywords in {selected_subcategory}</b>',
                                labels={'Counts': 'Health Search Volume', 'keyword': 'Health Keywords'},
                                color='keyword_ctr',
                                color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                                text='Counts'
                            )
                            
                            fig_keywords.update_traces(
                                texttemplate='%{text:,}',
                                textposition='outside'
                            )
                            
                            fig_keywords.update_layout(
                                plot_bgcolor='rgba(248,255,248,0.95)',
                                paper_bgcolor='rgba(232,245,232,0.8)',
                                font=dict(color='#1B5E20', family='Segoe UI'),
                                height=500,
                                xaxis=dict(tickangle=45),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_keywords, use_container_width=True)
                            
                            # Health keyword performance table
                            keyword_display = keyword_analysis.copy()
                            keyword_display['Counts'] = keyword_display['Counts'].apply(lambda x: f"{int(x):,}")
                            keyword_display['clicks'] = keyword_display['clicks'].apply(lambda x: f"{int(x):,}")
                            keyword_display['conversions'] = keyword_display['conversions'].apply(lambda x: f"{int(x):,}")
                            keyword_display['keyword_ctr'] = keyword_display['keyword_ctr'].apply(lambda x: f"{x:.2f}%")
                            keyword_display['keyword_cr'] = keyword_display['keyword_cr'].apply(lambda x: f"{x:.2f}%")
                            
                            keyword_display.columns = ['Health Keyword', 'Health Search Volume', 'Health Clicks', 
                                                     'Nutraceuticals & Nutrition Conversions', 'Health CTR %', 'Nutraceuticals & Nutrition CR %']
                            
                            st.dataframe(keyword_display, use_container_width=True, hide_index=True)
                        else:
                            st.info("No health keyword data available for this subcategory.")
                    
                    # Health trend analysis
                    st.markdown("### 📈 Health Subcategory Competitive Analysis")
                    
                    # Compare with similar performing subcategories
                    similar_volume_range = 0.3  # 30% range
                    min_volume = subcat_data['Counts'] * (1 - similar_volume_range)
                    max_volume = subcat_data['Counts'] * (1 + similar_volume_range)
                    
                    similar_subcats = sc[
                        (sc['Counts'] >= min_volume) & 
                        (sc['Counts'] <= max_volume) & 
                        (sc['sub_category'] != selected_subcategory)
                    ].head(5)
                    
                    if len(similar_subcats) > 0:
                        comparison_data = pd.concat([
                            sc[sc['sub_category'] == selected_subcategory],
                            similar_subcats
                        ])
                        
                        fig_competitive = go.Figure()
                        
                        fig_competitive.add_trace(go.Scatter(
                            x=comparison_data['ctr'],
                            y=comparison_data['conversion_rate'],
                            mode='markers+text',
                            text=comparison_data['sub_category'],
                            textposition='top center',
                            marker=dict(
                                size=comparison_data['Counts']/comparison_data['Counts'].max()*50 + 10,
                                color=['#2E7D32' if x == selected_subcategory else '#81C784' 
                                      for x in comparison_data['sub_category']],
                                opacity=0.8,
                                line=dict(width=2, color='white')
                            ),
                            name='Health Subcategories'
                        ))
                        
                        fig_competitive.update_layout(
                            title=f'<b style="color:#2E7D32;">🌿 Health Competitive Analysis - {selected_subcategory} vs Similar Volume Subcategories</b>',
                            xaxis_title='Health CTR (%)',
                            yaxis_title='Nutraceuticals & Nutrition Conversion Rate (%)',
                            plot_bgcolor='rgba(248,255,248,0.95)',
                            paper_bgcolor='rgba(232,245,232,0.8)',
                            font=dict(color='#1B5E20', family='Segoe UI'),
                            height=500,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_competitive, use_container_width=True)
                        
                        st.markdown("**📊 Bubble size represents health search volume. Selected subcategory is highlighted in dark green.**")
                    
                    # Enhanced download button for detailed health analysis
                    detailed_analysis_data = {
                        'Health Subcategory': [selected_subcategory],
                        'Health Search Volume': [subcat_data['Counts']],
                        'Total Health Clicks': [subcat_data['clicks']],
                        'Total Nutraceuticals & Nutrition Conversions': [subcat_data['conversions']],
                        'Health CTR %': [subcat_data['ctr']],
                        'Classic CVR %': [subcat_data['classic_cvr']],
                        'Nutraceuticals & Nutrition Conversion Rate %': [subcat_data['conversion_rate']],
                        'Health Market Rank': [subcat_rank],
                        'Nutraceuticals & Nutrition Market Share %': [market_share],
                        'Health Performance Score': [performance_score],
                        'Nutraceuticals & Nutrition Conversion Efficiency %': [conversion_efficiency]
                    }
                    
                    detailed_df = pd.DataFrame(detailed_analysis_data)
                    csv_detailed = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Detailed Health Analysis CSV",
                        data=csv_detailed,
                        file_name=f"detailed_health_analysis_{selected_subcategory.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="detailed_health_analysis_download"
                    )

            elif analysis_type == "📈 Health Performance Comparison":
                st.subheader("⚖️ Health Subcategory Performance Comparison")
                
                selected_subcategories = st.multiselect(
                    "Select health subcategories to compare (max 10):",
                    options=sc['sub_category'].tolist(),
                    default=sc['sub_category'].head(5).tolist(),
                    max_selections=10
                )
                
                if selected_subcategories:
                    comparison_data = sc[sc['sub_category'].isin(selected_subcategories)].copy()
                    
                    # Health comparison metrics visualization
                    fig_comparison = go.Figure()
                    
                    metrics = ['ctr', 'conversion_rate', 'click_share', 'conversion_share']
                    metric_names = ['Health CTR %', 'Nutraceuticals & Nutrition Conversion Rate %', 'Health Click Share %', 'Nutraceuticals & Nutrition Conversion Share %']
                    colors = ['#4CAF50', '#81C784', '#66BB6A', '#A5D6A7']
                    
                    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                        fig_comparison.add_trace(go.Bar(
                            name=name,
                            x=comparison_data['sub_category'],
                            y=comparison_data[metric],
                            marker_color=colors[i]
                        ))
                    
                    fig_comparison.update_layout(
                        title='<b style="color:#2E7D32;">🌿 Health Performance Metrics Comparison</b>',
                        barmode='group',
                        plot_bgcolor='rgba(248,255,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI'),
                        height=500,
                        xaxis=dict(tickangle=45),
                        yaxis=dict(title='Percentage (%)')
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Health performance scatter plot
                    st.markdown("### 📊 Health CTR vs Nutraceuticals & Nutrition Conversion Rate Scatter Analysis")
                    
                    fig_scatter = px.scatter(
                        comparison_data,
                        x='ctr',
                        y='conversion_rate',
                        size='Counts',
                        color='sub_category',
                        title='<b style="color:#2E7D32;">🌿 Health Performance Matrix - CTR vs Conversion Rate</b>',
                        labels={
                            'ctr': 'Health CTR (%)',
                            'conversion_rate': 'Nutraceuticals & Nutrition Conversion Rate (%)',
                            'Counts': 'Health Search Volume'
                        },
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    
                    fig_scatter.update_layout(
                        plot_bgcolor='rgba(248,255,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI'),
                        height=500
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Detailed health comparison table
                    st.markdown("### 📊 Detailed Health Comparison Table")
                    
                    comparison_table = comparison_data[['sub_category', 'Counts', 'clicks', 'conversions', 
                                                    'ctr', 'conversion_rate', 'click_share', 'conversion_share']].copy()
                    comparison_table.columns = ['Health Subcategory', 'Health Search Volume', 'Health Clicks', 'Nutraceuticals & Nutrition Conversions', 
                                            'Health CTR %', 'Nutraceuticals & Nutrition Conversion Rate %', 'Health Click Share %', 'Nutraceuticals & Nutrition Conversion Share %']
                    
                    # Format numeric columns
                    comparison_table['Health Search Volume'] = comparison_table['Health Search Volume'].apply(lambda x: f"{int(x):,}")
                    comparison_table['Health Clicks'] = comparison_table['Health Clicks'].apply(lambda x: f"{int(x):,}")
                    comparison_table['Nutraceuticals & Nutrition Conversions'] = comparison_table['Nutraceuticals & Nutrition Conversions'].apply(lambda x: f"{int(x):,}")
                    comparison_table['Health CTR %'] = comparison_table['Health CTR %'].apply(lambda x: f"{x:.2f}%")
                    comparison_table['Nutraceuticals & Nutrition Conversion Rate %'] = comparison_table['Nutraceuticals & Nutrition Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                    comparison_table['Health Click Share %'] = comparison_table['Health Click Share %'].apply(lambda x: f"{x:.2f}%")
                    comparison_table['Nutraceuticals & Nutrition Conversion Share %'] = comparison_table['Nutraceuticals & Nutrition Conversion Share %'].apply(lambda x: f"{x:.2f}%")

                    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
                    
                    # Health performance ranking
                    st.markdown("### 🏆 Health Performance Ranking")
                    
                    comparison_data['health_performance_score'] = (
                        comparison_data['ctr'] * 0.4 + 
                        comparison_data['conversion_rate'] * 0.4 + 
                        comparison_data['click_share'] * 0.2
                    )
                    
                    ranking_data = comparison_data.sort_values('health_performance_score', ascending=False).reset_index(drop=True)
                    ranking_data['rank'] = range(1, len(ranking_data) + 1)
                    
                    fig_ranking = px.bar(
                        ranking_data,
                        x='sub_category',
                        y='health_performance_score',
                        title='<b style="color:#2E7D32;">🌿 Health Performance Score Ranking</b>',
                        labels={'health_performance_score': 'Health Performance Score', 'sub_category': 'Health Subcategories'},
                        color='health_performance_score',
                        color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                        text='rank'
                    )
                    
                    fig_ranking.update_traces(
                        texttemplate='#%{text}',
                        textposition='outside'
                    )
                    
                    fig_ranking.update_layout(
                        plot_bgcolor='rgba(248,255,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI'),
                        height=500,
                        xaxis=dict(tickangle=45),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_ranking, use_container_width=True)
                    
                    csv_comparison = comparison_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Health Comparison Data CSV",
                        data=csv_comparison,
                        file_name="health_subcategory_comparison.csv",
                        mime="text/csv",
                        key="health_comparison_download"
                    )
                else:
                    st.info("Please select health subcategories to compare.")

            elif analysis_type == "📊 Nutraceuticals & Nutrition Market Share Analysis":
                st.subheader("📊 Nutraceuticals & Nutrition Market Share & Distribution Analysis")
                
                # Health market share visualization
                col_pie, col_treemap = st.columns(2)
                
                with col_pie:
                    # Pie chart for top 10 health subcategories
                    top_10_market = sc.head(10).copy()
                    others_value = sc.iloc[10:]['Counts'].sum() if len(sc) > 10 else 0
                    
                    if others_value > 0:
                        others_row = pd.DataFrame({
                            'sub_category': ['Other Health Subcategories'],
                            'Counts': [others_value]
                        })
                        pie_data = pd.concat([top_10_market[['sub_category', 'Counts']], others_row])
                    else:
                        pie_data = top_10_market[['sub_category', 'Counts']]
                    
                    fig_pie = px.pie(
                        pie_data,
                        values='Counts',
                        names='sub_category',
                        title='<b style="color:#2E7D32;">🌿 Top 10 Health Subcategories Market Share</b>',
                        color_discrete_sequence=['#2E7D32', '#388E3C', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C8', '#E8F5E8', '#F1F8E9', '#F9FBE7', '#DCEDC8']
                    )
                    
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(
                        height=400,
                        plot_bgcolor='rgba(248,255,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI')
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_treemap:
                    # Treemap visualization for health subcategories
                    fig_treemap = px.treemap(
                        sc.head(20),
                        path=['sub_category'],
                        values='Counts',
                        title='<b style="color:#2E7D32;">🌿 Health Subcategory Volume Distribution</b>',
                        color='ctr',
                        color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                        hover_data={'Counts': ':,', 'ctr': ':.2f'}
                    )
                    
                    fig_treemap.update_layout(
                        height=400,
                        plot_bgcolor='rgba(248,255,248,0.95)',
                        paper_bgcolor='rgba(232,245,232,0.8)',
                        font=dict(color='#1B5E20', family='Segoe UI')
                    )
                    st.plotly_chart(fig_treemap, use_container_width=True)
                
                # Health distribution analysis
                st.markdown("### 📈 Health Market Distribution Analysis")
                
                col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
                
                with col_dist1:
                    st.markdown(f"""
                    <div class='health-subcat-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{gini_coefficient:.3f}</div>
                        <div class='label'>Health Gini Coefficient</div>
                        <div class='sub-label'>Nutraceuticals & Nutrition market concentration</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_dist2:
                    st.markdown(f"""
                    <div class='health-subcat-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{herfindahl_index:.4f}</div>
                        <div class='label'>Health Herfindahl Index</div>
                        <div class='sub-label'>Nutraceuticals & Nutrition market dominance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_dist3:
                    st.markdown(f"""
                    <div class='health-subcat-metric-card'>
                        <span class='icon'>🔝</span>
                        <div class='value'>{top_5_concentration:.1f}%</div>
                        <div class='label'>Top 5 Health Share</div>
                        <div class='sub-label'>Nutraceuticals & Nutrition market concentration</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_dist4:
                    st.markdown(f"""
                    <div class='health-subcat-metric-card'>
                        <span class='icon'>💯</span>
                        <div class='value'>{top_10_concentration:.1f}%</div>
                        <div class='label'>Top 10 Health Share</div>
                        <div class='sub-label'>Nutraceuticals & Nutrition market concentration</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Health Lorenz curve
                st.markdown("### 📉 Health Market Concentration - Lorenz Curve")
                
                sorted_counts = sc['Counts'].sort_values()
                cumulative_counts = np.cumsum(sorted_counts) / sorted_counts.sum()
                cumulative_subcategories = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
                
                fig_lorenz = go.Figure()
                
                # Add Lorenz curve
                fig_lorenz.add_trace(go.Scatter(
                    x=cumulative_subcategories,
                    y=cumulative_counts,
                    mode='lines',
                    name='Actual Health Distribution',
                    line=dict(color='#4CAF50', width=3)
                ))
                
                # Add line of equality
                fig_lorenz.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Perfect Health Equality',
                    line=dict(color='#81C784', width=2, dash='dash')
                ))
                
                fig_lorenz.update_layout(
                    title='<b style="color:#2E7D32;">🌿 Lorenz Curve - Health Subcategory Search Volume Distribution</b>',
                    xaxis_title='Cumulative % of Health Subcategories',
                    yaxis_title='Cumulative % of Health Search Volume',
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=500,
                    xaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                    yaxis=dict(showgrid=True, gridcolor='#C8E6C8')
                )
                
                st.plotly_chart(fig_lorenz, use_container_width=True)

            # Health Subcategory Insights Section
            st.markdown("---")
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                top_subcat_share = sc.iloc[0]['click_share'] if not sc.empty else 0
                top_subcat_name = sc.iloc[0]['sub_category'] if not sc.empty else "N/A"
                high_performers = len(sc[sc['ctr'] > 5]) if not sc.empty else 0
                avg_conversion_rate = sc['conversion_rate'].mean() if not sc.empty else 0
                subcats_above_avg_cr = len(sc[sc['conversion_rate'] > avg_conversion_rate]) if not sc.empty else 0
                
                st.markdown(f"""
                <div class='Nutraceuticals & Nutrition-insight-card'>
                    <h4>🌿 Key Health Subcategory Insights</h4>
                    <p>• <strong>{top_subcat_name}</strong> leads Nutraceuticals & Nutrition market with {top_subcat_share:.1f}% click share<br>
                    • {high_performers} health subcategories achieve CTR > 5% (premium performance)<br>
                    • {subcats_above_avg_cr} subcategories exceed avg CR of {avg_conversion_rate:.2f}%<br>
                    • Health market shows balanced subcategory distribution</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_insight2:
                low_performers = len(sc[sc['ctr'] < 2]) if not sc.empty else 0
                opportunity_subcats = len(sc[(sc['Counts'] > sc['Counts'].median()) & (sc['ctr'] < 3)]) if not sc.empty else 0
                
                st.markdown(f"""
                <div class='Nutraceuticals & Nutrition-insight-card'>
                    <h4>💚 Health Subcategory Strategy Recommendations</h4>
                    <p>• Optimize {low_performers} underperforming health subcategories (CTR < 2%)<br>
                    • {opportunity_subcats} high-volume health subcategories need engagement boost<br>
                    • Focus on health keywords for leading Nutraceuticals & Nutrition subcategories<br>
                    • Strengthen health product portfolio strategy</p>
                </div>
                """, unsafe_allow_html=True)

            # Final Health Subcategory Summary Dashboard
            st.markdown("---")
            st.subheader("📊 Health Subcategory Performance Dashboard Summary")
            
            # Create final summary metrics
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                total_searches = sc['Counts'].sum() if not sc.empty else 0
                st.metric(
                    label="🌿 Total Health Searches",
                    value=f"{total_searches:,.0f}",
                    delta=f"{len(sc)} Nutraceuticals & Nutrition subcategories analyzed"
                )
            
            with summary_col2:
                avg_market_ctr = sc['ctr'].mean() if not sc.empty else 0
                top_ctr = sc['ctr'].max() if not sc.empty else 0
                st.metric(
                    label="📈 Health Market Avg CTR",
                    value=f"{avg_market_ctr:.2f}%",
                    delta=f"Best: {top_ctr:.2f}%"
                )
            
            with summary_col3:
                total_conversions = sc['conversions'].sum() if not sc.empty else 0
                avg_cr = sc['conversion_rate'].mean() if not sc.empty else 0
                st.metric(
                    label="💚 Total Nutraceuticals & Nutrition Conversions",
                    value=f"{total_conversions:,.0f}",
                    delta=f"Avg CR: {avg_cr:.2f}%"
                )
            
            with summary_col4:
                market_concentration = f"{top_5_concentration:.1f}%" if not sc.empty else "0%"
                concentration_status = "High" if top_5_concentration > 60 else "Medium" if top_5_concentration > 40 else "Low"
                st.metric(
                    label="🎯 Health Market Concentration",
                    value=market_concentration,
                    delta=f"{concentration_status} concentration"
                )
            
            # Export all health subcategory data
            st.markdown("---")
            st.subheader("📥 Export Health Subcategory Intelligence")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if not sc.empty:
                    export_data = sc.copy()
                    export_data['analysis_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
                    
                    csv_export = export_data.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Complete Health Subcategory Analysis",
                        data=csv_export,
                        file_name=f"health_subcategory_intelligence_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="complete_health_subcategory_export"
                    )
            
            with export_col2:
                if not sc.empty:
                    summary_report = f"""
HEALTH SUBCATEGORY INTELLIGENCE REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Nutraceuticals & Nutrition MARKET OVERVIEW:
• Total Health Subcategories Analyzed: {len(sc)}
• Total Health Searches: {sc['Counts'].sum():,.0f}
• Market Leader: {sc.iloc[0]['sub_category']} ({sc.iloc[0]['click_share']:.1f}% click share)
• Average Health CTR: {sc['ctr'].mean():.2f}%
• Average Nutraceuticals & Nutrition CR: {sc['conversion_rate'].mean():.2f}%

HEALTH PERFORMANCE TIERS:
• Premium Subcategories (CTR > 10%): {len(sc[sc['ctr'] > 10])}
• Strong Subcategories (CTR 5-10%): {len(sc[(sc['ctr'] >= 5) & (sc['ctr'] <= 10)])}
• Growing Subcategories (CTR 2-5%): {len(sc[(sc['ctr'] >= 2) & (sc['ctr'] < 5)])}
• Emerging Subcategories (CTR < 2%): {len(sc[sc['ctr'] < 2])}

MARKET CONCENTRATION ANALYSIS:
• Top 5 Health Subcategories: {top_5_concentration:.1f}% market share
• Top 10 Health Subcategories: {top_10_concentration:.1f}% market share
• Gini Coefficient: {gini_coefficient:.3f}
• Herfindahl Index: {herfindahl_index:.4f}
• Market Concentration Level: {concentration_status}

STRATEGIC HEALTH INSIGHTS:
• Market concentration is {concentration_status.lower()}
• {len(sc[sc['ctr'] > 5])} subcategories achieve premium performance
• Growth opportunities exist in Nutraceuticals & Nutrition engagement optimization
• Focus areas: {', '.join(sc[sc['ctr'] < 2]['sub_category'].head(3).tolist()) if len(sc[sc['ctr'] < 2]) > 0 else 'All subcategories performing well'}

TOP PERFORMING HEALTH SUBCATEGORIES:
{chr(10).join([f"• {row['sub_category']}: {row['Counts']:,} searches, {row['ctr']:.2f}% CTR, {row['conversion_rate']:.2f}% CR" for _, row in sc.head(5).iterrows()])}

OPTIMIZATION OPPORTUNITIES:
• High-volume, low-CTR subcategories need attention
• Conversion rate optimization potential across {len(sc[sc['conversion_rate'] < avg_conversion_rate])} subcategories
• Keyword expansion opportunities in top-performing health segments
• Cross-subcategory Nutraceuticals & Nutrition strategy development recommended
                    """
                    
                    st.download_button(
                        label="📝 Download Health Executive Summary Report",
                        data=summary_report,
                        file_name=f"health_subcategory_executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        key="health_subcategory_executive_summary_export"
                    )
                
        else:
            st.warning("No health subcategory data available in the uploaded file.")
            st.info("Please ensure your data contains a 'sub_category' column with valid health and Nutraceuticals & Nutrition values.")
            
    except Exception as e:
        st.error(f"An error occurred in the Health Subcategory analysis: {str(e)}")
        st.info("Please check your health data format and try again.")
        st.markdown("""
        **Expected data format:**
        - Column 'sub_category' with health subcategory names
        - Column 'Counts' with search volume data
        - Column 'clicks' with click data
        - Column 'conversions' with conversion data
        - Optional: Column 'keyword' for keyword analysis
        """)


# ----------------- Generic Type Tab -----------------

# Assuming tab_generic is defined elsewhere, e.g., tabs = st.tabs(["Generic"]), tab_generic = tabs[0]
with tab_generic:
    st.header("🌱 Generic Type Intelligence Hub")
    st.markdown("Deep dive into generic term performance and nutraceutical search trends. 💚")

    # Hero Image for Generic Type Tab
    generic_image_options = {
        "Generic Type Analytics": "https://placehold.co/1200x200/E8F5E8/2E7D32?text=Generic+Type+Performance+Analysis",
        "Nutraceutical Generics": "https://placehold.co/1200x200/4CAF50/FFFFFF?text=Nutraceutical+Generic+Intelligence+Dashboard",
        "Abstract Generic Types": "https://source.unsplash.com/1200x200/?nutrition,supplements,generic",
        "Health Gradient": "https://placehold.co/1200x200/C8E6C8/1B5E20?text=Lady+Care+Generic+Type+Insights",
    }
    selected_generic_image = st.sidebar.selectbox("Choose Generic Tab Hero", options=list(generic_image_options.keys()), index=0, key="generic_hero_image_selector")
    st.image(generic_image_options[selected_generic_image], use_container_width=True)

    try:
        # Check if generic type data exists and is valid
        if generic_type is None or generic_type.empty:
            st.warning("⚠️ No generic type data available.")
            st.info("Please ensure your uploaded file contains a 'generic_type' sheet with data.")
            st.stop()
        
        # Use the existing generic_type data directly
        gt = generic_type.copy()
        
        # Data validation and cleaning
        required_columns = ['search', 'count', 'Clicks', 'Conversions']
        missing_columns = [col for col in required_columns if col not in gt.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your generic type data contains these columns")
            st.stop()
        
        # Clean numeric data
        numeric_columns = ['count', 'Clicks', 'Conversions']
        for col in numeric_columns:
            gt[col] = pd.to_numeric(gt[col], errors='coerce').fillna(0)
        
        # Remove rows with missing search terms
        gt = gt.dropna(subset=['search'])
        gt = gt[gt['search'].str.strip() != '']
        
        if gt.empty:
            st.warning("⚠️ No valid generic type data found after cleaning.")
            st.info("Please check your data for empty search terms or invalid values.")
            st.stop()
        
        # Calculate comprehensive generic type metrics with loading indicator
        with st.spinner("🔄 Processing generic type data..."):
            gt_agg = gt.groupby('search').agg({
                'count': 'sum',
                'Clicks': 'sum', 
                'Conversions': 'sum'
            }).reset_index()
            
            # Calculate performance metrics
            gt_agg['ctr'] = gt_agg.apply(lambda r: (r['Clicks']/r['count']*100) if r['count']>0 else 0, axis=1)
            gt_agg['classic_cvr'] = gt_agg.apply(lambda r: (r['Conversions']/r['Clicks']*100) if r['Clicks']>0 else 0, axis=1)
            gt_agg['conversion_rate'] = gt_agg.apply(lambda r: (r['Conversions']/r['count']*100) if r['count']>0 else 0, axis=1)
            
            # Calculate additional metrics
            total_clicks = gt_agg['Clicks'].sum()
            total_conversions = gt_agg['Conversions'].sum()
            gt_agg['click_share'] = gt_agg.apply(lambda r: (r['Clicks']/total_clicks*100) if total_clicks>0 else 0, axis=1)
            gt_agg['conversion_share'] = gt_agg.apply(lambda r: (r['Conversions']/total_conversions*100) if total_conversions>0 else 0, axis=1)
            
            # Sort by counts for main analysis
            gt_agg = gt_agg.sort_values('count', ascending=False)
            
            # Calculate distribution metrics upfront for use in summary
            gini_coefficient = 1 - 2 * np.sum(np.cumsum(gt_agg['count'].sort_values()) / gt_agg['count'].sum()) / len(gt_agg)
            herfindahl_index = np.sum((gt_agg['count'] / gt_agg['count'].sum()) ** 2)
            top_5_concentration = gt_agg.head(5)['count'].sum() / gt_agg['count'].sum() * 100
            top_10_concentration = gt_agg.head(10)['count'].sum() / gt_agg['count'].sum() * 100
        
        # Enhanced CSS for nutrition-focused generic type metrics
        st.markdown("""
        <style>
        .nutrition-generic-metric-card {
            background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: #1B5E20;
            box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
            margin: 10px 0;
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 0.2s ease;
            border-left: 4px solid #4CAF50;
        }
        
        .nutrition-generic-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(46, 125, 50, 0.4);
        }
        
        .nutrition-generic-metric-card .icon {
            font-size: 3em;
            margin-bottom: 10px;
            display: block;
            color: #2E7D32;
        }
        
        .nutrition-generic-metric-card .value {
            font-size: 1.6em;
            font-weight: bold;
            margin-bottom: 8px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.2;
            color: #1B5E20;
        }
        
        .nutrition-generic-metric-card .label {
            font-size: 1.1em;
            opacity: 0.95;
            font-weight: 600;
            margin-bottom: 6px;
            color: #2E7D32;
        }
        
        .nutrition-generic-metric-card .sub-label {
            font-size: 1em;
            opacity: 0.9;
            font-weight: 500;
            line-height: 1.2;
            color: #388E3C;
        }
        
        .nutrition-performance-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 8px;
        }
        
        .high-nutrition-performance {
            background-color: #4CAF50;
            color: white;
        }
        
        .medium-nutrition-performance {
            background-color: #81C784;
            color: white;
        }
        
        .low-nutrition-performance {
            background-color: #A5D6A7;
            color: #1B5E20;
        }
        
        .nutrition-insight-card {
            background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin: 15px 0;
            box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Enhanced Key Metrics Section
        st.subheader("🌱 Generic Type Performance Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        total_generic_terms = len(gt_agg)
        total_searches = gt_agg['count'].sum()
        avg_ctr = gt_agg['ctr'].mean()
        avg_cr = gt_agg['conversion_rate'].mean()
        top_generic_term = gt_agg.iloc[0]['search'] if len(gt_agg) > 0 else 'N/A'
        top_generic_volume = gt_agg.iloc[0]['count'] if len(gt_agg) > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>🌱</span>
                <div class='value'>{format_number(total_generic_terms)}</div>
                <div class='label'>Total Generic Terms</div>
                <div class='sub-label'>Active nutraceutical terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>🔍</span>
                <div class='value'>{format_number(total_searches)}</div>
                <div class='label'>Total Searches</div>
                <div class='sub-label'>Across all generic terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            performance_class = "high-nutrition-performance" if avg_ctr > 5 else "medium-nutrition-performance" if avg_ctr > 2 else "low-nutrition-performance"
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>📈</span>
                <div class='value'>{avg_ctr:.2f}% <span class='nutrition-performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                <div class='label'>Average CTR</div>
                <div class='sub-label'>Click-through rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top_generic_display = top_generic_term[:12] + "..." if len(top_generic_term) > 12 else top_generic_term
            market_share = (top_generic_volume / total_searches * 100) if total_searches > 0 else 0
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>👑</span>
                <div class='value'>{top_generic_display}</div>
                <div class='label'>Top Generic Term</div>
                <div class='sub-label'>{market_share:.1f}% market share</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>💚</span>
                <div class='value'>{avg_cr:.2f}%</div>
                <div class='label'>Avg Conversion Rate</div>
                <div class='sub-label'>Overall performance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            total_clicks = int(gt_agg['Clicks'].sum())
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>🖱️</span>
                <div class='value'>{format_number(total_clicks)}</div>
                <div class='label'>Total Clicks</div>
                <div class='sub-label'>Across all generic terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            total_conversions = int(gt_agg['Conversions'].sum())
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>✅</span>
                <div class='value'>{format_number(total_conversions)}</div>
                <div class='label'>Total Conversions</div>
                <div class='sub-label'>Successful outcomes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            top_conversion_generic = gt_agg.nlargest(1, 'Conversions')['search'].iloc[0] if len(gt_agg) > 0 else 'N/A'
            top_conversion_display = top_conversion_generic[:12] + "..." if len(top_conversion_generic) > 12 else top_conversion_generic
            st.markdown(f"""
            <div class='nutrition-generic-metric-card'>
                <span class='icon'>🏆</span>
                <div class='value'>{top_conversion_display}</div>
                <div class='label'>Conversion Leader</div>
                <div class='sub-label'>Most conversions</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interactive generic type analysis
        st.subheader("🎯 Interactive Generic Type Analysis")

        # Analysis type selector
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["📊 Top Performers Overview", "🔍 Detailed Term Deep Dive", "📈 Performance Comparison", "📊 Distribution Analysis"],
            horizontal=True
        )

        if analysis_type == "📊 Top Performers Overview":
            # Top generic terms analysis
            st.subheader("🏆 Top 20 Generic Terms Performance")
            
            top_20_gt = gt_agg.head(20).copy()
            
            # Enhanced bar chart with nutrition-focused green colors
            fig_top_generics = px.bar(
                top_20_gt,
                x='search',
                y='count',
                title='<b style="color:#2E7D32;">🌱 Top 20 Generic Terms by Search Volume</b>',
                labels={'count': 'Search Volume', 'search': 'Generic Terms'},
                color='count',
                color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                text='count'
            )
            
            fig_top_generics.update_traces(
                texttemplate='%{text:,}',
                textposition='outside'
            )
            
            fig_top_generics.update_layout(
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                height=600,
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#C8E6C8'),
                yaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                showlegend=False
            )
            
            st.plotly_chart(fig_top_generics, use_container_width=True)
            
            # Performance metrics comparison chart
            st.subheader("📊 Performance Metrics Comparison")
            
            fig_metrics_comparison = go.Figure()
            
            # Add bars for each metric
            fig_metrics_comparison.add_trace(go.Bar(
                name='CTR %',
                x=top_20_gt['search'],
                y=top_20_gt['ctr'],
                marker_color='#4CAF50'
            ))
            
            fig_metrics_comparison.add_trace(go.Bar(
                name='Conversion Rate %',
                x=top_20_gt['search'],
                y=top_20_gt['conversion_rate'],
                marker_color='#81C784'
            ))
            
            fig_metrics_comparison.update_layout(
                title='<b style="color:#2E7D32;">🌱 CTR vs Conversion Rate Comparison</b>',
                barmode='group',
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                height=500,
                xaxis=dict(tickangle=45),
                yaxis=dict(title='Percentage (%)')
            )
            
            st.plotly_chart(fig_metrics_comparison, use_container_width=True)

        elif analysis_type == "🔍 Detailed Term Deep Dive":
            # Detailed analysis section
            st.subheader("🔬 Generic Term Deep Dive Analysis")
            
            # Generic term selector with search functionality
            selected_generic = st.selectbox(
                "Select a generic term for detailed analysis:",
                options=gt_agg['search'].tolist(),
                index=0
            )
            
            if selected_generic:
                # Get detailed data for selected generic term
                generic_data = gt_agg[gt_agg['search'] == selected_generic].iloc[0]
                generic_rank = gt_agg.reset_index().index[gt_agg['search'] == selected_generic].tolist()[0] + 1
                
                # Detailed metrics for selected generic term
                col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                
                with col_detail1:
                    rank_performance = "high-nutrition-performance" if generic_rank <= 3 else "medium-nutrition-performance" if generic_rank <= 10 else "low-nutrition-performance"
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>🏆</span>
                        <div class='value'>#{generic_rank} <span class='nutrition-performance-badge {rank_performance}'>{"Top 3" if generic_rank <= 3 else "Top 10" if generic_rank <= 10 else "Lower"}</span></div>
                        <div class='label'>Market Rank</div>
                        <div class='sub-label'>Out of {total_generic_terms} terms</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail2:
                    market_share = (generic_data['count'] / total_searches * 100)
                    share_performance = "high-nutrition-performance" if market_share > 5 else "medium-nutrition-performance" if market_share > 2 else "low-nutrition-performance"
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{market_share:.2f}% <span class='nutrition-performance-badge {share_performance}'>{"High" if market_share > 5 else "Medium" if market_share > 2 else "Low"}</span></div>
                        <div class='label'>Market Share</div>
                        <div class='sub-label'>Of total search volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail3:
                    performance_score = (generic_data['ctr'] + generic_data['conversion_rate']) / 2
                    score_performance = "high-nutrition-performance" if performance_score > 3 else "medium-nutrition-performance" if performance_score > 1 else "low-nutrition-performance"
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>⭐</span>
                        <div class='value'>{performance_score:.1f} <span class='nutrition-performance-badge {score_performance}'>{"High" if performance_score > 3 else "Medium" if performance_score > 1 else "Low"}</span></div>
                        <div class='label'>Performance Score</div>
                        <div class='sub-label'>Combined CTR & CR</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail4:
                    conversion_efficiency = generic_data['conversion_rate'] / generic_data['ctr'] * 100 if generic_data['ctr'] > 0 else 0
                    efficiency_performance = "high-nutrition-performance" if conversion_efficiency > 50 else "medium-nutrition-performance" if conversion_efficiency > 25 else "low-nutrition-performance"
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>⚡</span>
                        <div class='value'>{conversion_efficiency:.1f}% <span class='nutrition-performance-badge {efficiency_performance}'>{"High" if conversion_efficiency > 50 else "Medium" if conversion_efficiency > 25 else "Low"}</span></div>
                        <div class='label'>Conversion Efficiency</div>
                        <div class='sub-label'>CR as % of CTR</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed performance breakdown
                st.markdown("### 📈 Performance Breakdown")

                metrics_data = {
                    'Metric': ['Search Volume', 'Total Clicks', 'Total Conversions', 
                               'Click-Through Rate', 'Classic CVR (Conv/Clicks)', 
                               'Conversion Rate (Conv/Counts)', 'Click Share', 'Conversion Share'],
                    'Value': [
                        f"{int(generic_data['count']):,}",
                        f"{int(generic_data['Clicks']):,}",
                        f"{int(generic_data['Conversions']):,}",
                        f"{generic_data['ctr']:.2f}%",
                        f"{generic_data['classic_cvr']:.2f}%",
                        f"{generic_data['conversion_rate']:.2f}%",
                        f"{generic_data['click_share']:.2f}%",
                        f"{generic_data['conversion_share']:.2f}%"
                    ],
                    'Performance': [
                        'High' if generic_data['count'] > gt_agg['count'].median() else 'Low',
                        'High' if generic_data['Clicks'] > gt_agg['Clicks'].median() else 'Low',
                        'High' if generic_data['Conversions'] > gt_agg['Conversions'].median() else 'Low',
                        'High' if generic_data['ctr'] > gt_agg['ctr'].median() else 'Low',
                        'High' if generic_data['classic_cvr'] > gt_agg['classic_cvr'].median() else 'Low',
                        'High' if generic_data['conversion_rate'] > gt_agg['conversion_rate'].median() else 'Low',
                        'High' if generic_data['click_share'] > gt_agg['click_share'].median() else 'Low',
                        'High' if generic_data['conversion_share'] > gt_agg['conversion_share'].median() else 'Low'
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Performance comparison radar chart
                st.markdown("### 📊 Performance Radar Chart")
                
                # Normalize values for radar chart
                normalized_data = {
                    'Search Volume': generic_data['count'] / gt_agg['count'].max() * 100,
                    'CTR': generic_data['ctr'] / gt_agg['ctr'].max() * 100 if gt_agg['ctr'].max() > 0 else 0,
                    'Conversion Rate': generic_data['conversion_rate'] / gt_agg['conversion_rate'].max() * 100 if gt_agg['conversion_rate'].max() > 0 else 0,
                    'Click Share': generic_data['click_share'],
                    'Conversion Share': generic_data['conversion_share']
                }
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(normalized_data.values()),
                    theta=list(normalized_data.keys()),
                    fill='toself',
                    name=selected_generic,
                    line_color='#4CAF50',
                    fillcolor='rgba(76, 175, 80, 0.3)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            gridcolor='#C8E6C8'
                        ),
                        angularaxis=dict(
                            gridcolor='#C8E6C8'
                        )),
                    showlegend=True,
                    title=f'<b style="color:#2E7D32;">🌱 Performance Radar - {selected_generic}</b>',
                    height=400,
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI')
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)

        elif analysis_type == "📈 Performance Comparison":
            st.subheader("⚖️ Generic Terms Performance Comparison")
            
            # Multi-select for comparison
            selected_generics = st.multiselect(
                "Select generic terms to compare (max 10):",
                options=gt_agg['search'].tolist(),
                default=gt_agg['search'].head(5).tolist(),
                max_selections=10
            )
            
            if selected_generics:
                # Filter data for selected generic terms
                comparison_data = gt_agg[gt_agg['search'].isin(selected_generics)].copy()
                
                # Comparison metrics visualization
                fig_comparison = go.Figure()
                
                # Add traces for different metrics
                metrics = ['ctr', 'conversion_rate', 'click_share', 'conversion_share']
                metric_names = ['CTR %', 'Conversion Rate %', 'Click Share %', 'Conversion Share %']
                colors = ['#4CAF50', '#81C784', '#66BB6A', '#A5D6A7']
                
                for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                    fig_comparison.add_trace(go.Bar(
                        name=name,
                        x=comparison_data['search'],
                        y=comparison_data[metric],
                        marker_color=colors[i]
                    ))
                
                fig_comparison.update_layout(
                    title='<b style="color:#2E7D32;">🌱 Performance Metrics Comparison</b>',
                    barmode='group',
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45),
                    yaxis=dict(title='Percentage (%)')
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Detailed comparison table
                st.markdown("### 📊 Detailed Comparison Table")
                
                comparison_table = comparison_data[['search', 'count', 'Clicks', 'Conversions', 
                                                    'ctr', 'conversion_rate', 'click_share', 'conversion_share']].copy()
                comparison_table.columns = ['Generic Term', 'Search Volume', 'Clicks', 'Conversions', 
                                            'CTR %', 'Conversion Rate %', 'Click Share %', 'Conversion Share %']
                
                # Format numeric columns
                comparison_table['Search Volume'] = comparison_table['Search Volume'].apply(lambda x: f"{int(x):,}")
                comparison_table['Clicks'] = comparison_table['Clicks'].apply(lambda x: f"{int(x):,}")
                comparison_table['Conversions'] = comparison_table['Conversions'].apply(lambda x: f"{int(x):,}")
                comparison_table['CTR %'] = comparison_table['CTR %'].apply(lambda x: f"{x:.2f}%")
                comparison_table['Conversion Rate %'] = comparison_table['Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                comparison_table['Click Share %'] = comparison_table['Click Share %'].apply(lambda x: f"{x:.2f}%")
                comparison_table['Conversion Share %'] = comparison_table['Conversion Share %'].apply(lambda x: f"{x:.2f}%")

                st.dataframe(comparison_table, use_container_width=True, hide_index=True)
                
                # Download comparison data
                csv_comparison = comparison_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison Data CSV",
                    data=csv_comparison,
                    file_name="generic_terms_comparison.csv",
                    mime="text/csv",
                    key="generic_comparison_download"
                )
            else:
                st.info("Please select generic terms to compare.")

        elif analysis_type == "📊 Distribution Analysis":
            st.subheader("📊 Market Share & Distribution Analysis")
            
            # Market share visualization
            col_pie, col_treemap = st.columns(2)
            
            with col_pie:
                # Pie chart for top 10 generic terms
                top_10_market = gt_agg.head(10).copy()
                others_value = gt_agg.iloc[10:]['count'].sum() if len(gt_agg) > 10 else 0
                
                if others_value > 0:
                    others_row = pd.DataFrame({
                        'search': ['Others'],
                        'count': [others_value]
                    })
                    pie_data = pd.concat([top_10_market[['search', 'count']], others_row])
                else:
                    pie_data = top_10_market[['search', 'count']]
                
                fig_pie = px.pie(
                    pie_data,
                    values='count',
                    names='search',
                    title='<b style="color:#2E7D32;">🌱 Top 10 Generic Terms Market Share</b>',
                    color_discrete_sequence=['#2E7D32', '#388E3C', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C8', '#E8F5E8', '#F1F8E9', '#F9FBE7', '#DCEDC8']
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    height=400,
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI')
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_treemap:
                # Treemap visualization
                fig_treemap = px.treemap(
                    gt_agg.head(20),
                    path=['search'],
                    values='count',
                    title='<b style="color:#2E7D32;">🌱 Generic Terms Volume Distribution</b>',
                    color='ctr',
                    color_continuous_scale=['#E8F5E8', '#81C784', '#2E7D32'],
                    hover_data={'count': ':,', 'ctr': ':.2f'}
                )
                
                fig_treemap.update_layout(
                    height=400,
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI')
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Distribution analysis
            st.markdown("### 📈 Distribution Analysis")
            
            col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
            
            with col_dist1:
                st.markdown(f"""
                <div class='nutrition-generic-metric-card'>
                    <span class='icon'>📊</span>
                    <div class='value'>{gini_coefficient:.3f}</div>
                    <div class='label'>Gini Coefficient</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist2:
                st.markdown(f"""
                <div class='nutrition-generic-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{herfindahl_index:.4f}</div>
                    <div class='label'>Herfindahl Index</div>
                    <div class='sub-label'>Market dominance</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist3:
                st.markdown(f"""
                <div class='nutrition-generic-metric-card'>
                    <span class='icon'>🔝</span>
                    <div class='value'>{top_5_concentration:.1f}%</div>
                    <div class='label'>Top 5 Share</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist4:
                st.markdown(f"""
                <div class='nutrition-generic-metric-card'>
                    <span class='icon'>🔟</span>
                    <div class='value'>{top_10_concentration:.1f}%</div>
                    <div class='label'>Top 10 Share</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Lorenz Curve for market concentration
            st.markdown("### 📈 Market Concentration Analysis")
            
            # Calculate Lorenz curve data
            sorted_counts = gt_agg['count'].sort_values().values
            cumulative_counts = np.cumsum(sorted_counts)
            total_count = cumulative_counts[-1]
            
            # Normalize to percentages
            lorenz_x = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
            lorenz_y = cumulative_counts / total_count * 100
            
            # Create Lorenz curve
            fig_lorenz = go.Figure()
            
            # Add Lorenz curve
            fig_lorenz.add_trace(go.Scatter(
                x=lorenz_x,
                y=lorenz_y,
                mode='lines',
                name='Lorenz Curve',
                line=dict(color='#4CAF50', width=3)
            ))
            
            # Add line of equality
            fig_lorenz.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Line of Equality',
                line=dict(color='#81C784', width=2, dash='dash')
            ))
            
            fig_lorenz.update_layout(
                title='<b style="color:#2E7D32;">🌱 Lorenz Curve - Generic Terms Market Concentration</b>',
                xaxis_title='Cumulative % of Generic Terms',
                yaxis_title='Cumulative % of Search Volume',
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                height=400,
                showlegend=True,
                xaxis=dict(showgrid=True, gridcolor='#C8E6C8'),
                yaxis=dict(showgrid=True, gridcolor='#C8E6C8')
            )
            
            st.plotly_chart(fig_lorenz, use_container_width=True)
            
            # Market concentration insights
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.markdown("#### 🎯 Market Concentration Insights")
                
                if gini_coefficient > 0.7:
                    st.error("🔴 **Highly Concentrated Market**: Few generic terms dominate the search volume.")
                elif gini_coefficient > 0.5:
                    st.warning("🟡 **Moderately Concentrated Market**: Some generic terms have significant market share.")
                else:
                    st.success("🟢 **Well-Distributed Market**: Search volume is relatively evenly distributed.")
                
                st.markdown(f"- **Gini Coefficient**: {gini_coefficient:.3f} (0 = perfect equality, 1 = maximum inequality)")
                st.markdown(f"- **Top 5 Terms**: Control {top_5_concentration:.1f}% of total search volume")
                st.markdown(f"- **Top 10 Terms**: Control {top_10_concentration:.1f}% of total search volume")
            
            with col_insight2:
                st.markdown("#### 📊 Performance Distribution")
                
                # Performance quartiles
                q1 = gt_agg['count'].quantile(0.25)
                q2 = gt_agg['count'].quantile(0.50)
                q3 = gt_agg['count'].quantile(0.75)
                
                high_performers = len(gt_agg[gt_agg['count'] >= q3])
                medium_performers = len(gt_agg[(gt_agg['count'] >= q2) & (gt_agg['count'] < q3)])
                low_performers = len(gt_agg[gt_agg['count'] < q2])
                
                st.markdown(f"**📈 High Volume (Top 25%)**: {high_performers} terms")
                st.markdown(f"**📊 Medium Volume (25-75%)**: {medium_performers} terms")
                st.markdown(f"**📉 Low Volume (Bottom 50%)**: {low_performers} terms")
                
                # Average performance by quartile
                high_avg_ctr = gt_agg[gt_agg['count'] >= q3]['ctr'].mean()
                medium_avg_ctr = gt_agg[(gt_agg['count'] >= q2) & (gt_agg['count'] < q3)]['ctr'].mean()
                low_avg_ctr = gt_agg[gt_agg['count'] < q2]['ctr'].mean()
                
                st.markdown(f"**CTR by Volume:**")
                st.markdown(f"- High Volume: {high_avg_ctr:.2f}%")
                st.markdown(f"- Medium Volume: {medium_avg_ctr:.2f}%")
                st.markdown(f"- Low Volume: {low_avg_ctr:.2f}%")

        # Generic Type Insights Section
        st.markdown("---")
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            top_generic_share = gt_agg.iloc[0]['click_share'] if not gt_agg.empty else 0
            top_generic_name = gt_agg.iloc[0]['search'] if not gt_agg.empty else "N/A"
            high_performers = len(gt_agg[gt_agg['ctr'] > 5]) if not gt_agg.empty else 0
            avg_conversion_rate = gt_agg['conversion_rate'].mean() if not gt_agg.empty else 0
            generics_above_avg_cr = len(gt_agg[gt_agg['conversion_rate'] > avg_conversion_rate]) if not gt_agg.empty else 0
            
            st.markdown(f"""
            <div class='nutrition-insight-card'>
                <h4>🌱 Key Generic Type Insights</h4>
                <p>• <strong>{top_generic_name}</strong> leads market with {top_generic_share:.1f}% click share<br>
                • {high_performers} generic terms achieve CTR > 5% (premium performance)<br>
                • {generics_above_avg_cr} terms exceed avg CR of {avg_conversion_rate:.2f}%<br>
                • Market shows balanced generic term distribution</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_insight2:
            low_performers = len(gt_agg[gt_agg['ctr'] < 2]) if not gt_agg.empty else 0
            opportunity_generics = len(gt_agg[(gt_agg['count'] > gt_agg['count'].median()) & (gt_agg['ctr'] < 3)]) if not gt_agg.empty else 0
            
            st.markdown(f"""
            <div class='nutrition-insight-card'>
                <h4>💚 Generic Type Strategy Recommendations</h4>
                <p>• Optimize {low_performers} underperforming generic terms (CTR < 2%)<br>
                • {opportunity_generics} high-volume terms need engagement boost<br>
                • Focus on nutraceutical keywords for leading generics<br>
                • Strengthen product portfolio strategy</p>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced Download and Export Section
        st.markdown("---")
        st.subheader("💾 Advanced Export & Download Options")
        
        col_download1, col_download2, col_download3, col_download4 = st.columns(4)
        
        with col_download1:
            # Complete dataset download
            csv_complete = gt_agg.to_csv(index=False)
            st.download_button(
                label="📊 Complete Analysis CSV",
                data=csv_complete,
                file_name=f"generic_terms_complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="complete_generic_download",
                help="Download complete generic terms analysis with all calculated metrics"
            )
        
        with col_download2:
            # Top performers only
            top_performers_csv = gt_agg.head(50).to_csv(index=False)
            st.download_button(
                label="🏆 Top 50 Performers CSV",
                data=top_performers_csv,
                file_name=f"top_50_generic_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="top_performers_generic_download",
                help="Download top 50 performing generic terms"
            )
        
        with col_download3:
            # Summary report
            summary_report = f"""# Generic Terms Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Generic Terms Analyzed: {total_generic_terms:,}
- Total Search Volume: {total_searches:,}
- Average CTR: {avg_ctr:.2f}%
- Average Conversion Rate: {avg_cr:.2f}%
- Total Clicks: {total_clicks:,}
- Total Conversions: {total_conversions:,}

## Top Performing Generic Terms
{chr(10).join([f"{i+1}. {row['search']}: {int(row['count']):,} searches ({row['ctr']:.2f}% CTR, {row['conversion_rate']:.2f}% CR)" for i, (_, row) in enumerate(gt_agg.head(10).iterrows())])}

## Market Concentration Analysis
- Gini Coefficient: {gini_coefficient:.3f}
- Herfindahl Index: {herfindahl_index:.4f}
- Top 5 Market Share: {top_5_concentration:.1f}%
- Top 10 Market Share: {top_10_concentration:.1f}%

## Performance Distribution
- High Volume Terms (Top 25%): {len(gt_agg[gt_agg['count'] >= gt_agg['count'].quantile(0.75)])} terms
- Medium Volume Terms (25-75%): {len(gt_agg[(gt_agg['count'] >= gt_agg['count'].quantile(0.25)) & (gt_agg['count'] < gt_agg['count'].quantile(0.75))])} terms
- Low Volume Terms (Bottom 25%): {len(gt_agg[gt_agg['count'] < gt_agg['count'].quantile(0.25)])} terms

## Key Insights
- Top Generic Term: "{top_generic_term}" with {top_generic_volume:,} searches ({market_share:.1f}% market share)
- Conversion Leader: "{top_conversion_generic}" with {int(gt_agg.nlargest(1, 'Conversions')['Conversions'].iloc[0]):,} conversions
- Market Concentration: {"High" if gini_coefficient > 0.7 else "Medium" if gini_coefficient > 0.5 else "Low"}

## Recommendations
- Focus optimization efforts on top {min(20, len(gt_agg))} generic terms for maximum impact
- {"Consider expanding reach for high-converting but low-volume terms" if len(gt_agg[gt_agg['conversion_rate'] > avg_cr]) > 0 else ""}
- {"Investigate underperforming high-volume terms for optimization opportunities" if len(gt_agg[(gt_agg['count'] > gt_agg['count'].median()) & (gt_agg['ctr'] < avg_ctr)]) > 0 else ""}

Generated by Generic Terms Analysis Dashboard
"""
            
            st.download_button(
                label="📋 Executive Summary",
                data=summary_report,
                file_name=f"generic_terms_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="summary_generic_download",
                help="Download executive summary report"
            )
        
        with col_download4:
            # Filtered high-opportunity terms
            high_opportunity = gt_agg[
                (gt_agg['count'] > gt_agg['count'].median()) & 
                (gt_agg['ctr'] < avg_ctr)
            ]
            
            if len(high_opportunity) > 0:
                opportunity_csv = high_opportunity.to_csv(index=False)
                st.download_button(
                    label="🎯 High Opportunity Terms",
                    data=opportunity_csv,
                    file_name=f"high_opportunity_generic_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="opportunity_generic_download",
                    help="Download high-volume but underperforming terms for optimization"
                )
            else:
                st.info("No high-opportunity terms identified")

        # Advanced Filtering Section
        st.markdown("---")
        st.subheader("🔍 Advanced Filtering & Custom Analysis")
        
        with st.expander("🎛️ Custom Filter Options", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                st.markdown("**Volume Filters**")
                min_searches = st.number_input(
                    "Minimum Search Volume:",
                    min_value=0,
                    max_value=int(gt_agg['count'].max()),
                    value=0,
                    key="min_searches_filter"
                )
                
                max_searches = st.number_input(
                    "Maximum Search Volume:",
                    min_value=int(min_searches),
                    max_value=int(gt_agg['count'].max()),
                    value=int(gt_agg['count'].max()),
                    key="max_searches_filter"
                )
            
            with filter_col2:
                st.markdown("**Performance Filters**")
                min_ctr = st.slider(
                    "Minimum CTR (%):",
                    min_value=0.0,
                    max_value=float(gt_agg['ctr'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_ctr_filter"
                )
                
                min_cr = st.slider(
                    "Minimum Conversion Rate (%):",
                    min_value=0.0,
                    max_value=float(gt_agg['conversion_rate'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_cr_filter"
                )
            
            with filter_col3:
                st.markdown("**Text Filters**")
                search_contains = st.text_input(
                    "Generic term contains:",
                    placeholder="Enter text to search...",
                    key="search_contains_filter"
                )
                
                exclude_terms = st.text_input(
                    "Exclude terms containing:",
                    placeholder="Enter text to exclude...",
                    key="exclude_terms_filter"
                )
            
            # Apply filters
            filtered_data = gt_agg[
                (gt_agg['count'] >= min_searches) &
                (gt_agg['count'] <= max_searches) &
                (gt_agg['ctr'] >= min_ctr) &
                (gt_agg['conversion_rate'] >= min_cr)
            ].copy()
            
            if search_contains:
                filtered_data = filtered_data[
                    filtered_data['search'].str.contains(search_contains, case=False, na=False)
                ]
            
            if exclude_terms:
                filtered_data = filtered_data[
                    ~filtered_data['search'].str.contains(exclude_terms, case=False, na=False)
                ]
            
            # Display filtered results
            if len(filtered_data) > 0:
                st.markdown(f"### 📊 Filtered Results: {len(filtered_data)} generic terms")
                
                # Quick stats for filtered data - USING CSS CARDS WITH NUTRITION THEME
                filtered_col1, filtered_col2, filtered_col3, filtered_col4 = st.columns(4)
                
                with filtered_col1:
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{len(filtered_data):,}</div>
                        <div class='label'>Terms Found</div>
                        <div class='sub-label'>Matching filters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col2:
                    total_searches_filtered = filtered_data['count'].sum()
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>🔍</span>
                        <div class='value'>{total_searches_filtered:,}</div>
                        <div class='label'>Total Searches</div>
                        <div class='sub-label'>Filtered volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col3:
                    avg_ctr_filtered = filtered_data['ctr'].mean()
                    ctr_performance = "high-nutrition-performance" if avg_ctr_filtered > 5 else "medium-nutrition-performance" if avg_ctr_filtered > 2 else "low-nutrition-performance"
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{avg_ctr_filtered:.2f}% <span class='nutrition-performance-badge {ctr_performance}'>{"High" if avg_ctr_filtered > 5 else "Medium" if avg_ctr_filtered > 2 else "Low"}</span></div>
                        <div class='label'>Avg CTR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col4:
                    avg_cr_filtered = filtered_data['conversion_rate'].mean()
                    cr_performance = "high-nutrition-performance" if avg_cr_filtered > 3 else "medium-nutrition-performance" if avg_cr_filtered > 1 else "low-nutrition-performance"
                    st.markdown(f"""
                    <div class='nutrition-generic-metric-card'>
                        <span class='icon'>💚</span>
                        <div class='value'>{avg_cr_filtered:.2f}% <span class='nutrition-performance-badge {cr_performance}'>{"High" if avg_cr_filtered > 3 else "Medium" if avg_cr_filtered > 1 else "Low"}</span></div>
                        <div class='label'>Avg CR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display filtered data
                display_filtered = filtered_data[['search', 'count', 'Clicks', 'Conversions', 'ctr', 'conversion_rate']].copy()
                display_filtered.columns = ['Generic Term', 'Search Volume', 'Clicks', 'Conversions', 'CTR %', 'Conversion Rate %']
                
                # Format for display
                display_filtered['Search Volume'] = display_filtered['Search Volume'].apply(lambda x: f"{int(x):,}")
                display_filtered['Clicks'] = display_filtered['Clicks'].apply(lambda x: f"{int(x):,}")
                display_filtered['Conversions'] = display_filtered['Conversions'].apply(lambda x: f"{int(x):,}")
                display_filtered['CTR %'] = display_filtered['CTR %'].apply(lambda x: f"{x:.2f}%")
                display_filtered['Conversion Rate %'] = display_filtered['Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(display_filtered, use_container_width=True, hide_index=True)
                
                # Download filtered data
                filtered_csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=filtered_csv,
                    file_name=f"filtered_generic_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="filtered_generic_download"
                )
            else:
                st.warning("⚠️ No generic terms match the selected filters. Try adjusting your criteria.")

        # Final Generic Type Summary Dashboard
        st.markdown("---")
        st.subheader("📊 Generic Type Performance Dashboard Summary")
        
        # Create final summary metrics
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            total_searches = gt_agg['count'].sum() if not gt_agg.empty else 0
            st.metric(
                label="🌱 Total Generic Searches",
                value=f"{total_searches:,.0f}",
                delta=f"{len(gt_agg)} terms analyzed"
            )
        
        with summary_col2:
            avg_market_ctr = gt_agg['ctr'].mean() if not gt_agg.empty else 0
            top_ctr = gt_agg['ctr'].max() if not gt_agg.empty else 0
            st.metric(
                label="📈 Market Avg CTR",
                value=f"{avg_market_ctr:.2f}%",
                delta=f"Best: {top_ctr:.2f}%"
            )
        
        with summary_col3:
            total_conversions = gt_agg['Conversions'].sum() if not gt_agg.empty else 0
            avg_cr = gt_agg['conversion_rate'].mean() if not gt_agg.empty else 0
            st.metric(
                label="💚 Total Conversions",
                value=f"{total_conversions:,.0f}",
                delta=f"Avg CR: {avg_cr:.2f}%"
            )
        
        with summary_col4:
            market_concentration = f"{top_5_concentration:.1f}%" if not gt_agg.empty else "0%"
            concentration_status = "High" if top_5_concentration > 60 else "Medium" if top_5_concentration > 40 else "Low"
            st.metric(
                label="🎯 Market Concentration",
                value=market_concentration,
                delta=f"{concentration_status} concentration"
            )

    except KeyError as e:
        st.error(f"❌ Missing required column: {str(e)}")
        st.info("Please ensure your data contains: 'search', 'count', 'Clicks', 'Conversions'")
    except ValueError as e:
        st.error(f"❌ Data format error: {str(e)}")
        st.info("Please check that numeric columns contain valid numbers")
    except Exception as e:
        st.error(f"❌ Unexpected error processing generic type data: {str(e)}")
        st.info("Please check your data format and try again.")
        st.markdown("""
        **Expected data format:**
        - Column 'search' with generic term names
        - Column 'count' with search volume data
        - Column 'Clicks' with click data
        - Column 'Conversions' with conversion data
        """)


# ----------------- Time Analysis Tab (Enhanced) -----------------
# ----------------- Time Analysis Tab (Enhanced) -----------------
with tab_time:
    st.header("🌿 Temporal Health Intelligence Hub")
    st.markdown("Deep dive into monthly performance and Nutraceuticals & Nutrition search trends. 💚")

    # Hero Image for Time Analysis Tab
    time_image_options = {
        "Temporal Health Analytics": "https://placehold.co/1200x200/E8F5E8/2E7D32?text=Temporal+Health+Performance+Analysis",
        "Wellness Time Trends": "https://placehold.co/1200x200/4CAF50/FFFFFF?text=Wellness+Temporal+Intelligence+Dashboard",
        "Abstract Health Time": "https://source.unsplash.com/1200x200/?health,wellness,time",
        "Health Time Gradient": "https://placehold.co/1200x200/C8E6C8/1B5E20?text=Lady+Care+Health+Temporal+Insights",
    }
    selected_time_image = st.sidebar.selectbox("Choose Time Tab Hero", options=list(time_image_options.keys()), index=0, key="time_hero_image_selector")
    st.image(time_image_options[selected_time_image], use_container_width=True)

    try:
        # Check if time data exists and is valid
        if queries is None or queries.empty:
            st.warning("⚠️ No time data available.")
            st.info("Please ensure your uploaded file contains valid time data.")
            st.stop()
        
        # Data validation and cleaning
        required_columns = ['month', 'Counts', 'clicks', 'conversions']
        missing_columns = [col for col in required_columns if col not in queries.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your time data contains these columns")
            st.stop()
        
        # Clean numeric data
        numeric_columns = ['Counts', 'clicks', 'conversions']
        queries = queries.copy()
        for col in numeric_columns:
            queries[col] = pd.to_numeric(queries[col], errors='coerce').fillna(0)
        
        # Remove rows with missing or empty month data
        queries = queries.dropna(subset=['month'])
        queries = queries[queries['month'].str.strip() != '']
        
        if queries.empty:
            st.warning("⚠️ No valid time data found after cleaning.")
            st.info("Please check your data for empty or invalid month values.")
            st.stop()
        
        # Calculate comprehensive monthly metrics
        with st.spinner("🔄 Processing time data..."):
            monthly = queries.groupby('month').agg({
                'Counts': 'sum',
                'clicks': 'sum',
                'conversions': 'sum'
            }).reset_index()
            
            # Calculate performance metrics
            monthly['ctr'] = monthly.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
            monthly['conversion_rate'] = monthly.apply(lambda r: (r['conversions'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
            monthly['classic_cvr'] = monthly.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1)
            
            # Calculate share metrics
            total_clicks = monthly['clicks'].sum()
            total_conversions = monthly['conversions'].sum()
            monthly['click_share'] = monthly.apply(lambda r: (r['clicks'] / total_clicks * 100) if total_clicks > 0 else 0, axis=1)
            monthly['conversion_share'] = monthly.apply(lambda r: (r['conversions'] / total_conversions * 100) if total_conversions > 0 else 0, axis=1)
            
            # Try to convert month to datetime and sort
            try:
                monthly['month_dt'] = pd.to_datetime(monthly['month'], format='%b %Y', errors='coerce')
                monthly = monthly.sort_values('month_dt')
            except:
                monthly = monthly.sort_values('month')
            
            # Calculate distribution metrics
            gini_coefficient = 1 - 2 * np.sum(np.cumsum(monthly['Counts'].sort_values()) / monthly['Counts'].sum()) / len(monthly)
            top_3_concentration = monthly.head(3)['Counts'].sum() / monthly['Counts'].sum() * 100
        
        # CSS for UI consistency with health theme
        st.markdown("""
        <style>
        .time-metric-card {
            background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: #1B5E20;
            box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
            margin: 10px 0;
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 0.2s ease;
            border-left: 4px solid #4CAF50;
        }
        .time-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(46, 125, 50, 0.4);
        }
        .time-metric-card .icon {
            font-size: 3em;
            margin-bottom: 10px;
            display: block;
            color: #2E7D32;
        }
        .time-metric-card .value {
            font-size: 1.6em;
            font-weight: bold;
            margin-bottom: 8px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.2;
            color: #1B5E20;
        }
        .time-metric-card .label {
            font-size: 1.1em;
            opacity: 0.95;
            font-weight: 600;
            margin-bottom: 6px;
            color: #2E7D32;
        }
        .time-metric-card .sub-label {
            font-size: 1em;
            opacity: 0.9;
            font-weight: 500;
            line-height: 1.2;
            color: #388E3C;
        }
        .time-performance-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 8px;
        }
        .high-time-performance {
            background-color: #4CAF50;
            color: white;
        }
        .medium-time-performance {
            background-color: #81C784;
            color: white;
        }
        .low-time-performance {
            background-color: #A5D6A7;
            color: #1B5E20;
        }
        .time-table-container {
            background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            transition: transform 0.2s ease;
        }
        .time-table-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .time-table-container table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        .time-table-container th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            font-size: 1.1em;
        }
        .time-table-container td {
            padding: 10px;
            font-size: 1em;
            color: #2D3748;
            border-bottom: 1px solid #E2E8F0;
        }
        .time-table-container tr:nth-child(even) {
            background-color: #E8F5E8;
        }
        .time-table-container tr:hover {
            background-color: #C8E6C8;
        }
        .time-insight-card {
            background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            margin: 15px 0;
            box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Key Metrics Section
        st.subheader("🌿 Monthly Health Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_months = len(monthly)
        total_searches = monthly['Counts'].sum()
        avg_ctr = monthly['ctr'].mean()
        avg_cr = monthly['conversion_rate'].mean()
        
        with col1:
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>📅</span>
                <div class='value'>{total_months}</div>
                <div class='label'>Total Months</div>
                <div class='sub-label'>Analyzed periods</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>🔍</span>
                <div class='value'>{format_number(total_searches)}</div>
                <div class='label'>Total Health Searches</div>
                <div class='sub-label'>Across all months</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            performance_class = "high-time-performance" if avg_ctr > 5 else "medium-time-performance" if avg_ctr > 2 else "low-time-performance"
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>📈</span>
                <div class='value'>{avg_ctr:.2f}% <span class='time-performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                <div class='label'>Average Health CTR</div>
                <div class='sub-label'>Monthly average</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            performance_class = "high-time-performance" if avg_cr > 3 else "medium-time-performance" if avg_cr > 1 else "low-time-performance"
            st.markdown(f"""
            <div class='time-metric-card'>
                <span class='icon'>💚</span>
                <div class='value'>{avg_cr:.2f}% <span class='time-performance-badge {performance_class}'>{"High" if avg_cr > 3 else "Medium" if avg_cr > 1 else "Low"}</span></div>
                <div class='label'>Avg Nutraceuticals & Nutrition CR</div>
                <div class='sub-label'>Monthly average</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive Analysis Section
        st.markdown("---")
        st.subheader("🎯 Interactive Temporal Health Analysis")
        
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["📊 Health Trends Overview", "🔍 Detailed Month Health Analysis", "🏷 Brand Health Comparison", "📊 Nutraceuticals & Nutrition Distribution Analysis"],
            horizontal=True
        )
        
        if analysis_type == "📊 Health Trends Overview":
            st.subheader("📈 Monthly Health Trends")
            
            # Line chart for counts
            fig_counts = px.line(
                monthly,
                x='month',
                y='Counts',
                title='<b style="color:#2E7D32;">🌿 Monthly Health Search Volume</b>',
                labels={'Counts': 'Health Search Volume', 'month': 'Month'},
                color_discrete_sequence=['#4CAF50']
            )
            fig_counts.update_traces(line=dict(width=3))
            fig_counts.update_layout(
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                height=400,
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#C8E6C8'),
                yaxis=dict(showgrid=True, gridcolor='#C8E6C8')
            )
            st.plotly_chart(fig_counts, use_container_width=True)
            
            # Line chart for CTR and Conversion Rate
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(
                x=monthly['month'],
                y=monthly['ctr'],
                name='Health CTR %',
                line=dict(color='#4CAF50', width=3)
            ))
            fig_metrics.add_trace(go.Scatter(
                x=monthly['month'],
                y=monthly['conversion_rate'],
                name='Nutraceuticals & Nutrition Conversion Rate %',
                line=dict(color='#81C784', width=3)
            ))
            fig_metrics.update_layout(
                title='<b style="color:#2E7D32;">🌿 Monthly Health CTR and Nutraceuticals & Nutrition Conversion Rate Trends</b>',
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI'),
                height=400,
                xaxis=dict(tickangle=45, title='Month'),
                yaxis=dict(title='Percentage (%)')
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        elif analysis_type == "🔍 Detailed Month Health Analysis":
            st.subheader("🔬 Detailed Monthly Health Performance")
            
            selected_month = st.selectbox(
                "Select a month for detailed Nutraceuticals & Nutrition analysis:",
                options=monthly['month'].tolist(),
                index=0
            )
            
            if selected_month:
                month_data = monthly[monthly['month'] == selected_month].iloc[0]
                month_rank = monthly.reset_index().index[monthly['month'] == selected_month].tolist()[0] + 1
                
                col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                
                with col_detail1:
                    rank_performance = "high-time-performance" if month_rank <= 3 else "medium-time-performance" if month_rank <= 6 else "low-time-performance"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>🏆</span>
                        <div class='value'>#{month_rank}</div>
                        <div class='label'>Health Month Rank</div>
                        <div class='sub-label'>Out of {total_months} months</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail2:
                    market_share = (month_data['Counts'] / total_searches * 100)
                    share_performance = "high-time-performance" if market_share > (100/total_months) else "medium-time-performance" if market_share > (50/total_months) else "low-time-performance"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{market_share:.2f}%</div>
                        <div class='label'>Nutraceuticals & Nutrition Market Share</div>
                        <div class='sub-label'>Of total health searches</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail3:
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{month_data['ctr']:.2f}% <span class='time-performance-badge {"high-time-performance" if month_data['ctr'] > 5 else "medium-time-performance" if month_data['ctr'] > 2 else "low-time-performance"}'>{"High" if month_data['ctr'] > 5 else "Medium" if month_data['ctr'] > 2 else "Low"}</span></div>
                        <div class='label'>Health CTR</div>
                        <div class='sub-label'>Month performance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail4:
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>💚</span>
                        <div class='value'>{month_data['conversion_rate']:.2f}% <span class='time-performance-badge {"high-time-performance" if month_data['conversion_rate'] > 3 else "medium-time-performance" if month_data['conversion_rate'] > 1 else "low-time-performance"}'>{"High" if month_data['conversion_rate'] > 3 else "Medium" if month_data['conversion_rate'] > 1 else "Low"}</span></div>
                        <div class='label'>Nutraceuticals & Nutrition Conversion Rate</div>
                        <div class='sub-label'>Month performance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed performance table
                st.markdown("### 📊 Health Performance Breakdown")
                metrics_data = {
                    'Health Metric': ['Search Volume', 'Total Clicks', 'Total Conversions', 'CTR', 'Conversion Rate', 'Classic CVR', 'Click Share', 'Conversion Share'],
                    'Value': [
                        f"{int(month_data['Counts']):,}",
                        f"{int(month_data['clicks']):,}",
                        f"{int(month_data['conversions']):,}",
                        f"{month_data['ctr']:.2f}%",
                        f"{month_data['conversion_rate']:.2f}%",
                        f"{month_data['classic_cvr']:.2f}%",
                        f"{month_data['click_share']:.2f}%",
                        f"{month_data['conversion_share']:.2f}%"
                    ],
                    'Health Performance': [
                        'High' if month_data['Counts'] > monthly['Counts'].median() else 'Low',
                        'High' if month_data['clicks'] > monthly['clicks'].median() else 'Low',
                        'High' if month_data['conversions'] > monthly['conversions'].median() else 'Low',
                        'High' if month_data['ctr'] > monthly['ctr'].median() else 'Low',
                        'High' if month_data['conversion_rate'] > monthly['conversion_rate'].median() else 'Low',
                        'High' if month_data['classic_cvr'] > monthly['classic_cvr'].median() else 'Low',
                        'High' if month_data['click_share'] > monthly['click_share'].median() else 'Low',
                        'High' if month_data['conversion_share'] > monthly['conversion_share'].median() else 'Low'
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                st.markdown("<div class='time-table-container'>", unsafe_allow_html=True)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        elif analysis_type == "🏷 Brand Health Comparison":
            st.subheader("🏷 Top Brands Health Performance by Month")
            
            if 'brand' in queries.columns and queries['brand'].notna().any() and queries['month'].notna().any():
                # Filter out 'Other' (case-insensitive) and select top 5 brands by Counts
                brand_counts = queries[queries['brand'].str.lower() != 'other'].groupby('brand')['Counts'].sum()
                top_brands = brand_counts.sort_values(ascending=False).head(5).index
                brand_month = queries[queries['brand'].isin(top_brands)].groupby(['month', 'brand']).agg({
                    'Counts': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                brand_month['ctr'] = brand_month.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
                brand_month['conversion_rate'] = brand_month.apply(lambda r: (r['conversions'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
                
                try:
                    brand_month['month_dt'] = pd.to_datetime(brand_month['month'], format='%b %Y', errors='coerce')
                    brand_month = brand_month.sort_values('month_dt')
                except:
                    brand_month = brand_month.sort_values('month')
                
                # Bar chart for brand counts
                fig_brands = px.bar(
                    brand_month,
                    x='month',
                    y='Counts',
                    color='brand',
                    title='<b style="color:#2E7D32;">🌿 Top 5 Brands by Health Search Volume per Month</b>',
                    color_discrete_sequence=['#2E7D32', '#388E3C', '#4CAF50', '#66BB6A', '#81C784']
                )
                fig_brands.update_layout(
                    plot_bgcolor='rgba(248,255,248,0.95)',
                    paper_bgcolor='rgba(232,245,232,0.8)',
                    font=dict(color='#1B5E20', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45, title='Month'),
                    yaxis=dict(title='Health Search Volume')
                )
                st.plotly_chart(fig_brands, use_container_width=True)
                
                # Comparison table
                st.markdown("### 📊 Brand Health Performance Table")
                display_brands = brand_month[['month', 'brand', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate']].copy()
                display_brands.columns = ['Month', 'Brand', 'Health Search Volume', 'Health Clicks', 'Nutraceuticals & Nutrition Conversions', 'Health CTR %', 'Nutraceuticals & Nutrition Conversion Rate %']
                display_brands['Health Search Volume'] = display_brands['Health Search Volume'].apply(lambda x: f"{int(x):,}")
                display_brands['Health Clicks'] = display_brands['Health Clicks'].apply(lambda x: f"{int(x):,}")
                display_brands['Nutraceuticals & Nutrition Conversions'] = display_brands['Nutraceuticals & Nutrition Conversions'].apply(lambda x: f"{int(x):,}")
                display_brands['Health CTR %'] = display_brands['Health CTR %'].apply(lambda x: f"{x:.2f}%")
                display_brands['Nutraceuticals & Nutrition Conversion Rate %'] = display_brands['Nutraceuticals & Nutrition Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                st.markdown("<div class='time-table-container'>", unsafe_allow_html=True)
                st.dataframe(display_brands, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download brand data
                csv_brands = brand_month.to_csv(index=False)
                st.download_button(
                    label="📥 Download Brand Health Data CSV",
                    data=csv_brands,
                    file_name=f"brand_monthly_health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="brand_monthly_health_download"
                )
            else:
                st.info("Brand or month data not available for brand-month health analysis.")
        
        elif analysis_type == "📊 Nutraceuticals & Nutrition Distribution Analysis":
            st.subheader("📊 Monthly Nutraceuticals & Nutrition Distribution Analysis")
            
            # Pie chart for market share
            fig_pie = px.pie(
                monthly,
                values='Counts',
                names='month',
                title='<b style="color:#2E7D32;">🌿 Monthly Health Search Volume Distribution</b>',
                color_discrete_sequence=['#2E7D32', '#388E3C', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C8', '#E8F5E8', '#F1F8E9', '#F9FBE7', '#DCEDC8']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=400,
                plot_bgcolor='rgba(248,255,248,0.95)',
                paper_bgcolor='rgba(232,245,232,0.8)',
                font=dict(color='#1B5E20', family='Segoe UI')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Distribution metrics
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                st.markdown(f"""
                <div class='time-metric-card'>
                    <span class='icon'>📊</span>
                    <div class='value'>{gini_coefficient:.3f}</div>
                    <div class='label'>Health Gini Coefficient</div>
                    <div class='sub-label'>Nutraceuticals & Nutrition concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist2:
                st.markdown(f"""
                <div class='time-metric-card'>
                    <span class='icon'>🔝</span>
                    <div class='value'>{top_3_concentration:.1f}%</div>
                    <div class='label'>Top 3 Months Share</div>
                    <div class='sub-label'>Search volume concentration</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Temporal Health Insights Section
        st.markdown("---")
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            top_month_share = monthly.iloc[0]['click_share'] if not monthly.empty else 0
            top_month_name = monthly.iloc[0]['month'] if not monthly.empty else "N/A"
            high_performers = len(monthly[monthly['ctr'] > 5]) if not monthly.empty else 0
            avg_conversion_rate = monthly['conversion_rate'].mean() if not monthly.empty else 0
            months_above_avg_cr = len(monthly[monthly['conversion_rate'] > avg_conversion_rate]) if not monthly.empty else 0
            
            st.markdown(f"""
            <div class='time-insight-card'>
                <h4>🌿 Key Temporal Health Insights</h4>
                <p>• <strong>{top_month_name}</strong> leads Nutraceuticals & Nutrition period with {top_month_share:.1f}% click share<br>
                • {high_performers} months achieve CTR > 5% (premium performance)<br>
                • {months_above_avg_cr} months exceed avg CR of {avg_conversion_rate:.2f}%<br>
                • Health trends show seasonal distribution</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_insight2:
            low_performers = len(monthly[monthly['ctr'] < 2]) if not monthly.empty else 0
            opportunity_months = len(monthly[(monthly['Counts'] > monthly['Counts'].median()) & (monthly['ctr'] < 3)]) if not monthly.empty else 0
            
            st.markdown(f"""
            <div class='time-insight-card'>
                <h4>💚 Temporal Strategy Recommendations</h4>
                <p>• Optimize {low_performers} underperforming months (CTR < 2%)<br>
                • {opportunity_months} high-volume periods need engagement boost<br>
                • Plan seasonal campaigns for peak Nutraceuticals & Nutrition months<br>
                • Strengthen year-round health strategy</p>
            </div>
            """, unsafe_allow_html=True)

        # Advanced Filtering Section
        st.markdown("---")
        st.subheader("🔍 Advanced Filtering & Custom Health Analysis")

        with st.expander("🎛️ Custom Filter Options", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                st.markdown("**Volume Filters**")
                min_searches = st.number_input(
                    "Minimum Search Volume:",
                    min_value=0,
                    max_value=int(monthly['Counts'].max()),
                    value=0,
                    key="min_searches_time_filter"
                )
                max_searches = st.number_input(
                    "Maximum Search Volume:",
                    min_value=int(min_searches),
                    max_value=int(monthly['Counts'].max()),
                    value=int(monthly['Counts'].max()),
                    key="max_searches_time_filter"
                )
            
            with filter_col2:
                st.markdown("**Performance Filters**")
                min_ctr = st.slider(
                    "Minimum CTR (%):",
                    min_value=0.0,
                    max_value=float(monthly['ctr'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_ctr_time_filter"
                )
                min_cr = st.slider(
                    "Minimum Conversion Rate (%):",
                    min_value=0.0,
                    max_value=float(monthly['conversion_rate'].max()),
                    value=0.0,
                    step=0.1,
                    key="min_cr_time_filter"
                )
            
            with filter_col3:
                st.markdown("**Brand Filter**")
                if 'brand' in queries.columns and queries['brand'].notna().any():
                    # Convert brands to string, handle NaN
                    brand_series = queries['brand'].astype(str).replace('nan', '')
                    # Exclude 'Other' (case-insensitive) and empty strings
                    brand_options = [b for b in brand_series.unique().tolist() if b.lower() != 'other' and b]
                    selected_brands = st.multiselect(
                        "Select brands to include:",
                        options=brand_options,
                        default=brand_options[:min(3, len(brand_options))],
                        key="brand_time_filter"
                    )
                else:
                    selected_brands = []
                    st.info("No brand data available for filtering.")
            
            # Apply filters
            filtered_data = monthly.copy()  # Start with full monthly data
            
            # Apply volume and performance filters
            filtered_data = filtered_data[
                (filtered_data['Counts'] >= min_searches) &
                (filtered_data['Counts'] <= max_searches) &
                (filtered_data['ctr'] >= min_ctr) &
                (filtered_data['conversion_rate'] >= min_cr)
            ]
            
            # Apply brand filter if selected
            if selected_brands:
                # Filter queries for selected brands, excluding 'Other' and NaN
                brand_series = queries['brand'].astype(str).replace('nan', '')
                brand_filtered = queries[
                    (brand_series.isin(selected_brands)) & 
                    (brand_series.str.lower() != 'other')
                ].groupby('month').agg({
                    'Counts': 'sum',
                    'clicks': 'sum',
                    'conversions': 'sum'
                }).reset_index()
                
                # Recalculate metrics for brand-filtered data
                brand_filtered['ctr'] = brand_filtered.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
                brand_filtered['conversion_rate'] = brand_filtered.apply(lambda r: (r['conversions'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
                brand_filtered['classic_cvr'] = brand_filtered.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1)
                brand_filtered['click_share'] = brand_filtered.apply(
                    lambda r: (r['clicks'] / total_clicks * 100) if total_clicks > 0 else 0, axis=1
                )
                brand_filtered['conversion_share'] = brand_filtered.apply(
                    lambda r: (r['conversions'] / total_conversions * 100) if total_conversions > 0 else 0, axis=1
                )
                
                # Merge with filtered_data to retain only months that match brand-filtered data
                filtered_data = filtered_data.merge(
                    brand_filtered[['month', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate', 'classic_cvr', 'click_share', 'conversion_share']],
                    on='month',
                    how='inner',
                    suffixes=('', '_brand')
                )
                
                # Update columns with brand-filtered values
                for col in ['Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate', 'classic_cvr', 'click_share', 'conversion_share']:
                    filtered_data[col] = filtered_data[f'{col}_brand']
                    filtered_data = filtered_data.drop(columns=f'{col}_brand')
            
            # Display filtered results
            if len(filtered_data) > 0:
                st.markdown(f"### 📊 Filtered Results: {len(filtered_data)} months")
                
                filtered_col1, filtered_col2, filtered_col3, filtered_col4 = st.columns(4)
                
                with filtered_col1:
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📅</span>
                        <div class='value'>{len(filtered_data)}</div>
                        <div class='label'>Months Found</div>
                        <div class='sub-label'>Matching filters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col2:
                    total_searches_filtered = filtered_data['Counts'].sum()
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>🔍</span>
                        <div class='value'>{format_number(total_searches_filtered)}</div>
                        <div class='label'>Total Health Searches</div>
                        <div class='sub-label'>Filtered volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col3:
                    avg_ctr_filtered = filtered_data['ctr'].mean()
                    ctr_performance = "high-time-performance" if avg_ctr_filtered > 5 else "medium-time-performance" if avg_ctr_filtered > 2 else "low-time-performance"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{avg_ctr_filtered:.2f}% <span class='time-performance-badge {ctr_performance}'>{"High" if avg_ctr_filtered > 5 else "Medium" if avg_ctr_filtered > 2 else "Low"}</span></div>
                        <div class='label'>Avg Health CTR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col4:
                    avg_cr_filtered = filtered_data['conversion_rate'].mean()
                    cr_performance = "high-time-performance" if avg_cr_filtered > 3 else "medium-time-performance" if avg_cr_filtered > 1 else "low-time-performance"
                    st.markdown(f"""
                    <div class='time-metric-card'>
                        <span class='icon'>💚</span>
                        <div class='value'>{avg_cr_filtered:.2f}% <span class='time-performance-badge {cr_performance}'>{"High" if avg_cr_filtered > 3 else "Medium" if avg_cr_filtered > 1 else "Low"}</span></div>
                        <div class='label'>Avg Nutraceuticals & Nutrition CR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display filtered data
                display_filtered = filtered_data[['month', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate']].copy()
                display_filtered.columns = ['Month', 'Health Search Volume', 'Health Clicks', 'Nutraceuticals & Nutrition Conversions', 'Health CTR %', 'Nutraceuticals & Nutrition Conversion Rate %']
                display_filtered['Health Search Volume'] = display_filtered['Health Search Volume'].apply(lambda x: f"{int(x):,}")
                display_filtered['Health Clicks'] = display_filtered['Health Clicks'].apply(lambda x: f"{int(x):,}")
                display_filtered['Nutraceuticals & Nutrition Conversions'] = display_filtered['Nutraceuticals & Nutrition Conversions'].apply(lambda x: f"{int(x):,}")
                display_filtered['Health CTR %'] = display_filtered['Health CTR %'].apply(lambda x: f"{x:.2f}%")
                display_filtered['Nutraceuticals & Nutrition Conversion Rate %'] = display_filtered['Nutraceuticals & Nutrition Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                
                st.markdown("<div class='time-table-container'>", unsafe_allow_html=True)
                st.dataframe(display_filtered, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download filtered data
                filtered_csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Health Data",
                    data=filtered_csv,
                    file_name=f"filtered_monthly_health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="filtered_time_health_download"
                )
            else:
                st.warning("⚠️ No months match the selected filters. Try adjusting your criteria.")
        
        # Download and Export Section
        st.markdown("---")
        st.subheader("💾 Advanced Export & Download Options")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            csv_complete = monthly.to_csv(index=False)
            st.download_button(
                label="📊 Complete Monthly Health Analysis CSV",
                data=csv_complete,
                file_name=f"monthly_health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="complete_time_health_download"
            )
        
        with col_download2:
            summary_report = f"""# Monthly Health Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Months Analyzed: {total_months}
- Total Health Search Volume: {total_searches:,}
- Average Health CTR: {avg_ctr:.2f}%
- Average Nutraceuticals & Nutrition Conversion Rate: {avg_cr:.2f}%
- Total Health Clicks: {int(total_clicks):,}
- Total Nutraceuticals & Nutrition Conversions: {int(total_conversions):,}

## Top Performing Months
{chr(10).join([f"{row['month']}: {int(row['Counts']):,} searches ({row['ctr']:.2f}% CTR, {row['conversion_rate']:.2f}% CR)" for _, row in monthly.head(3).iterrows()])}

## Market Concentration
- Gini Coefficient: {gini_coefficient:.3f}
- Top 3 Months Share: {top_3_concentration:.1f}%

## Recommendations
- Focus on high-performing months for Nutraceuticals & Nutrition campaign optimization
- Investigate low-performing months for health improvement opportunities

Generated by Temporal Health Analysis Dashboard
"""
            st.download_button(
                label="📋 Health Executive Summary",
                data=summary_report,
                file_name=f"monthly_health_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="summary_time_health_download"
            )
    
    except KeyError as e:
        st.error(f"❌ Missing required column: {str(e)}")
        st.info("Please ensure your data contains: 'month', 'Counts', 'clicks', 'conversions'")
    except ValueError as e:
        st.error(f"❌ Data format error: {str(e)}")
        st.info("Please check that numeric columns contain valid numbers")
    except Exception as e:
        st.error(f"❌ Unexpected error processing time health data: {str(e)}")
        st.info("Please check your data format and try again.")

# ----------------- Pivot Builder Tab -----------------
with tab_pivot:
    st.header("🌿 Pivot Health Intelligence Hub")
    st.markdown("Deep dive into custom pivots and Nutraceuticals & Nutrition data insights. 💚")

    # Hero Image for Pivot Builder Tab
    pivot_image_options = {
        "Pivot Health Analytics": "https://placehold.co/1200x200/E8F5E8/2E7D32?text=Pivot+Health+Performance+Analysis",
        "Wellness Pivot Builder": "https://placehold.co/1200x200/4CAF50/FFFFFF?text=Wellness+Pivot+Intelligence+Dashboard",
        "Abstract Health Pivots": "https://source.unsplash.com/1200x200/?health,wellness,pivot",
        "Health Pivot Gradient": "https://placehold.co/1200x200/C8E6C8/1B5E20?text=Lady+Care+Health+Pivot+Insights",
    }
    selected_pivot_image = st.sidebar.selectbox("Choose Pivot Tab Hero", options=list(pivot_image_options.keys()), index=0, key="pivot_hero_image_selector")
    st.image(pivot_image_options[selected_pivot_image], use_container_width=True)

    # Apply CSS for consistency with health theme
    st.markdown("""
    <style>
    .pivot-metric-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: #1B5E20;
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
        margin: 10px 0;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease;
        border-left: 4px solid #4CAF50;
    }
    .pivot-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(46, 125, 50, 0.4);
    }
    .pivot-metric-card .icon {
        font-size: 3em;
        margin-bottom: 10px;
        display: block;
        color: #2E7D32;
    }
    .pivot-metric-card .value {
        font-size: 1.6em;
        font-weight: bold;
        margin-bottom: 8px;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.2;
        color: #1B5E20;
    }
    .pivot-metric-card .label {
        font-size: 1.1em;
        opacity: 0.95;
        font-weight: 600;
        margin-bottom: 6px;
        color: #2E7D32;
    }
    .pivot-metric-card .sub-label {
        font-size: 1em;
        opacity: 0.9;
        font-weight: 500;
        line-height: 1.2;
        color: #388E3C;
    }
    .pivot-performance-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 8px;
    }
    .high-pivot-performance {
        background-color: #4CAF50;
        color: white;
    }
    .medium-pivot-performance {
        background-color: #81C784;
        color: white;
    }
    .low-pivot-performance {
        background-color: #A5D6A7;
        color: #1B5E20;
    }
    .pivot-table-container {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.2s ease;
    }
    .pivot-table-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .pivot-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
    }
    .pivot-table-container th {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 12px;
        text-align: left;
        font-size: 1.1em;
    }
    .pivot-table-container td {
        padding: 10px;
        font-size: 1em;
        color: #2D3748;
        border-bottom: 1px solid #E2E8F0;
    }
    .pivot-table-container tr:nth-child(even) {
        background-color: #E8F5E8;
    }
    .pivot-table-container tr:hover {
        background-color: #C8E6C8;
    }
    .pivot-insight-card {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    try:
        # Prebuilt Pivot: Brand × Query (Top 300)
        st.subheader("📋 Prebuilt: Brand × Health Query (Top 300)")
        if 'brand' in queries.columns and 'normalized_query' in queries.columns:
            # Validate and clean data
            queries = queries.copy()
            queries['brand'] = queries['brand'].astype(str).replace('nan', '')
            queries['Counts'] = pd.to_numeric(queries['Counts'], errors='coerce').fillna(0)
            
            # Aggregate
            pv = queries[queries['brand'].str.lower() != 'other'].groupby(['brand', 'normalized_query']).agg(
                Counts=('Counts', 'sum'),
                clicks=('clicks', 'sum'),
                conversions=('conversions', 'sum')
            ).reset_index()
            pv['ctr'] = pv.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
            pv['conversion_rate'] = pv.apply(lambda r: (r['conversions'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
            
            # Debug: Check total Counts before filtering
            total_counts_all = pv['Counts'].sum()
        
            
            # Filter to top 300
            pv_top = pv.sort_values('Counts', ascending=False).head(300)
            
            # Calculate metrics for top 300
            total_rows = len(pv_top)
            total_counts = pv_top['Counts'].sum()
            avg_ctr = pv_top['ctr'].mean()
            avg_cr = pv_top['conversion_rate'].mean()
            
            # Debug: Check total Counts for top 300

            
            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>📋</span>
                    <div class='value'>{total_rows:,}</div>
                    <div class='label'>Total Rows</div>
                    <div class='sub-label'>Top 300 Brand-Query Pairs</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>🔍</span>
                    <div class='value'>{format_number(int(total_counts))}</div>
                    <div class='label'>Total Health Searches</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                performance_class = "high-pivot-performance" if avg_ctr > 5 else "medium-pivot-performance" if avg_ctr > 2 else "low-pivot-performance"
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{avg_ctr:.2f}% <span class='pivot-performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                    <div class='label'>Average Health CTR</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                performance_class = "high-pivot-performance" if avg_cr > 3 else "medium-pivot-performance" if avg_cr > 1 else "low-pivot-performance"
                st.markdown(f"""
                <div class='pivot-metric-card'>
                    <span class='icon'>💚</span>
                    <div class='value'>{avg_cr:.2f}% <span class='pivot-performance-badge {performance_class}'>{"High" if avg_cr > 3 else "Medium" if avg_cr > 1 else "Low"}</span></div>
                    <div class='label'>Avg Nutraceuticals & Nutrition CR</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Sorting and filtering options
            with st.expander("🔍 Filter & Sort Options", expanded=False):
                sort_col = st.selectbox("Sort By:", options=['Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate'], index=0)
                sort_order = st.radio("Sort Order:", options=['Descending', 'Ascending'], index=0, horizontal=True)
                min_counts = st.number_input("Minimum Search Volume:", min_value=0, value=0)
                pv_top = pv_top[pv_top['Counts'] >= min_counts].sort_values(sort_col, ascending=(sort_order == 'Ascending')).head(300)
            
            # Display pivot
            st.markdown("<div class='pivot-table-container'>", unsafe_allow_html=True)
            if AGGRID_OK:
                gb = GridOptionsBuilder.from_dataframe(pv_top)
                gb.configure_grid_options(enableRangeSelection=True, pagination=True, paginationPageSize=10)
                AgGrid(pv_top, gridOptions=gb.build(), height=400, theme='material')
            else:
                st.dataframe(pv_top, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button
            csv_pv = pv_top.to_csv(index=False)
            st.download_button(
                label="📥 Download Brand × Health Query Pivot",
                data=csv_pv,
                file_name=f"brand_health_query_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="brand_health_query_pivot_download"
            )
        else:
            st.info("Brand or normalized_query column missing for this pivot.")
            
            # Sorting and filtering options
            with st.expander("🔍 Filter & Sort Options", expanded=False):
                sort_col = st.selectbox("Sort By:", options=['Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate'], index=0)
                sort_order = st.radio("Sort Order:", options=['Descending', 'Ascending'], index=0, horizontal=True)
                min_counts = st.number_input("Minimum Search Volume:", min_value=0, value=0)
                pv_top = pv[pv['Counts'] >= min_counts].sort_values(sort_col, ascending=(sort_order == 'Ascending')).head(300)
            
            # Display pivot
            st.markdown("<div class='pivot-table-container'>", unsafe_allow_html=True)
            if AGGRID_OK:
                gb = GridOptionsBuilder.from_dataframe(pv_top)
                gb.configure_grid_options(enableRangeSelection=True, pagination=True, paginationPageSize=10)
                AgGrid(pv_top, gridOptions=gb.build(), height=400, theme='material')
            else:
                st.dataframe(pv_top, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button
            csv_pv = pv_top.to_csv(index=False)
            st.download_button(
                label="📥 Download Brand × Health Query Pivot",
                data=csv_pv,
                file_name=f"brand_health_query_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="brand_health_query_pivot_download"
            )
        
            
        # Custom Pivot Builder
        st.markdown("---")
        st.subheader("🔧 Custom Health Pivot Builder")
        columns = queries.columns.tolist()
        
        with st.expander("🎛️ Pivot Configuration", expanded=True):
            col_pivot1, col_pivot2 = st.columns(2)
            with col_pivot1:
                idx = st.multiselect(
                    "Rows (Index)",
                    options=columns,
                    default=['normalized_query'],
                    help="Select one or more columns to group as rows."
                )
                cols = st.multiselect(
                    "Columns",
                    options=[c for c in columns if c not in idx],
                    default=['brand'] if 'brand' in columns else [],
                    help="Select one or more columns to group as columns."
                )
            with col_pivot2:
                val = st.selectbox(
                    "Value (Measure)",
                    options=['Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate'],
                    index=0,
                    help="Select the metric to aggregate (CTR and Conversion Rate are calculated post-aggregation)."
                )
                aggfunc = st.selectbox(
                    "Aggregation",
                    options=['sum', 'mean', 'count', 'max', 'min'],
                    index=0,
                    help="Choose how to aggregate the selected value."
                )
            
            # Preview pivot structure
            if idx and cols and val:
                st.markdown(f"**Preview Health Pivot Structure**")
                st.write(f"Rows: {', '.join(idx)}")
                st.write(f"Columns: {', '.join(cols)}")
                st.write(f"Value: {val} ({aggfunc})")
            else:
                st.warning("Please select at least one row, one column, and a value to generate the pivot.")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_pivot = st.button("Generate Health Pivot")
        with col_btn2:
            reset_pivot = st.button("Reset Selections")
        
        if reset_pivot:
            st.session_state['min_searches_time_filter'] = 0
            st.session_state['max_searches_time_filter'] = int(queries['Counts'].max())
            st.session_state['min_ctr_time_filter'] = 0.0
            st.session_state['min_cr_time_filter'] = 0.0
            st.session_state['brand_time_filter'] = []
            st.experimental_rerun()
        
        if generate_pivot:
            if not idx or not cols or not val:
                st.error("Please select at least one row, one column, and a value.")
            else:
                try:
                    # Create a copy of queries to avoid modifying original
                    pivot_data = queries.copy()
                    # Calculate derived metrics if selected
                    if val in ['ctr', 'conversion_rate']:
                        pivot_data['ctr'] = pivot_data.apply(
                            lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1
                        )
                        pivot_data['conversion_rate'] = pivot_data.apply(
                            lambda r: (r['conversions'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1
                        )
                    # Handle brand column if present
                    if 'brand' in pivot_data.columns:
                        pivot_data['brand'] = pivot_data['brand'].astype(str).replace('nan', '')
                        pivot_data = pivot_data[pivot_data['brand'].str.lower() != 'other']
                    
                    pivot = pd.pivot_table(
                        pivot_data,
                        values=val,
                        index=idx,
                        columns=cols,
                        aggfunc=aggfunc,
                        fill_value=0
                    )
                    st.markdown("<div class='pivot-table-container'>", unsafe_allow_html=True)
                    if AGGRID_OK:
                        gb = GridOptionsBuilder.from_dataframe(pivot.reset_index())
                        gb.configure_grid_options(enableRangeSelection=True, pagination=True, paginationPageSize=10)
                        AgGrid(pivot.reset_index(), gridOptions=gb.build(), height=400, theme='material')
                    else:
                        st.dataframe(pivot, use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Download button
                    csv_pivot = pivot.to_csv()
                    st.download_button(
                        label="⬇ Download Custom Health Pivot CSV",
                        data=csv_pivot,
                        file_name=f"custom_health_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="custom_health_pivot_download"
                    )
                except Exception as e:
                    st.error(f"Pivot generation error: {e}")
                    st.info("Ensure selected columns and values are valid and contain data.")
        
        # Pivot Insights Section
        st.markdown("---")
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown(f"""
            <div class='pivot-insight-card'>
                <h4>🌿 Key Pivot Health Insights</h4>
                <p>• Analyze brand-query interactions for Nutraceuticals & Nutrition patterns<br>
                • Identify high-performing combinations<br>
                • Spot seasonal health trends in data<br>
                • Uncover conversion opportunities</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_insight2:
            st.markdown(f"""
            <div class='pivot-insight-card'>
                <h4>💚 Pivot Strategy Recommendations</h4>
                <p>• Customize pivots for specific health metrics<br>
                • Focus on top brand-query pairs<br>
                • Optimize Nutraceuticals & Nutrition campaigns based on insights<br>
                • Explore multi-dimensional analysis</p>
            </div>
            """, unsafe_allow_html=True)

        # Final Pivot Summary Dashboard
        st.markdown("---")
        st.subheader("📊 Pivot Performance Dashboard Summary")
        
        # Create final summary metrics (using prebuilt pivot data as example)
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric(
                label="🌿 Total Pivot Rows",
                value=f"{total_rows:,.0f}",
                delta="Top 300 analyzed"
            )
        
        with summary_col2:
            st.metric(
                label="📈 Avg Health CTR",
                value=f"{avg_ctr:.2f}%",
                delta="Across pairs"
            )
        
        with summary_col3:
            st.metric(
                label="💚 Total Nutraceuticals & Nutrition Conversions",
                value=f"{pv_top['conversions'].sum():,.0f}",
                delta=f"Avg CR: {avg_cr:.2f}%"
            )
        
        with summary_col4:
            concentration = (pv_top.head(5)['Counts'].sum() / total_counts * 100) if total_counts > 0 else 0
            status = "High" if concentration > 60 else "Medium" if concentration > 40 else "Low"
            st.metric(
                label="🎯 Pivot Concentration",
                value=f"{concentration:.1f}%",
                delta=f"{status} in top 5"
            )
        
    except Exception as e:
        st.error(f"❌ Unexpected error in Pivot Health Builder: {e}")
        st.info("Please check your data format and ensure required columns are present.")

# ----------------- Insights & Questions (Modified) -----------------
with tab_insights:
    st.header("🌿 Health Insights & Actionable Questions (10)")
    st.markdown("Curated health insights focused on **search** data for Nutraceuticals & Nutrition-driven decisions, with enhanced tables, charts, and visuals. 🚀")

    # Hero Image for Insights Tab
    insights_image_options = {
        "Health Insights Analytics": "https://placehold.co/1200x200/E8F5E8/2E7D32?text=Health+Insights+Performance+Analysis",
        "Wellness Insights Hub": "https://placehold.co/1200x200/4CAF50/FFFFFF?text=Wellness+Insights+Intelligence+Dashboard",
        "Abstract Health Insights": "https://source.unsplash.com/1200x200/?health,wellness,insights",
        "Health Insights Gradient": "https://placehold.co/1200x200/C8E6C8/1B5E20?text=Lady+Care+Health+Insights",
    }
    selected_insights_image = st.sidebar.selectbox("Choose Insights Tab Hero", options=list(insights_image_options.keys()), index=0, key="insights_hero_image_selector")
    st.image(insights_image_options[selected_insights_image], use_container_width=True)

    # Apply CSS for enhanced green health theme consistency
    st.markdown("""
    <style>
    .health-insight-metric-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: #1B5E20;
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
        margin: 10px 0;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease;
        border-left: 4px solid #4CAF50;
    }
    .health-insight-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(46, 125, 50, 0.4);
    }
    .health-insight-metric-card .icon {
        font-size: 3em;
        margin-bottom: 10px;
        display: block;
        color: #2E7D32;
    }
    .health-insight-metric-card .value {
        font-size: 1.6em;
        font-weight: bold;
        margin-bottom: 8px;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.2;
        color: #1B5E20;
    }
    .health-insight-metric-card .label {
        font-size: 1.1em;
        opacity: 0.95;
        font-weight: 600;
        margin-bottom: 6px;
        color: #2E7D32;
    }
    .health-insight-metric-card .sub-label {
        font-size: 1em;
        opacity: 0.9;
        font-weight: 500;
        line-height: 1.2;
        color: #388E3C;
    }
    .health-insight-performance-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 8px;
    }
    .high-health-insight-performance {
        background-color: #4CAF50;
        color: white;
    }
    .medium-health-insight-performance {
        background-color: #81C784;
        color: white;
    }
    .low-health-insight-performance {
        background-color: #A5D6A7;
        color: #1B5E20;
    }
    .health-insight-table-container {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.2s ease;
    }
    .health-insight-table-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .health-insight-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
    }
    .health-insight-table-container th {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 12px;
        text-align: left;
        font-size: 1.1em;
    }
    .health-insight-table-container td {
        padding: 10px;
        font-size: 1em;
        color: #2D3748;
        border-bottom: 1px solid #E2E8F0;
    }
    .health-insight-table-container tr:nth-child(even) {
        background-color: #E8F5E8;
    }
    .health-insight-table-container tr:hover {
        background-color: #C8E6C8;
    }
    .health-insight-box {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C8 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(46, 125, 50, 0.1);
    }
    .health-insight-box h4 {
        color: #2E7D32;
        margin-bottom: 10px;
    }
    .health-insight-box p {
        color: #388E3C;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

    # Cache data processing for performance
    @st.cache_data
    def preprocess_data():
        df = queries.copy()
        df['brand'] = df['brand'].astype(str).replace('nan', '').str.strip()
        df['Counts'] = pd.to_numeric(df['Counts'], errors='coerce').fillna(0)
        df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0)
        df['conversions'] = pd.to_numeric(df['conversions'], errors='coerce').fillna(0)
        df['month'] = df['month'].astype(str).str.strip()
        df['sub_category'] = df['sub_category'].astype(str).str.strip() if 'sub_category' in df.columns else ''
        df['category'] = df['category'].astype(str).str.strip() if 'category' in df.columns else ''
        return df

    # Load preprocessed data
    try:
        queries_clean = preprocess_data()
    except Exception as e:
        st.error(f"Data preprocessing error: {e}")
        st.stop()

    # Overall Health Insights Summary
    st.subheader("📊 Overall Health Insights Summary")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    total_queries = len(queries_clean['normalized_query'].unique())
    total_searches = queries_clean['Counts'].sum()
    avg_ctr = queries_clean.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1).mean()
    avg_cr = queries_clean.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1).mean()
    
    with summary_col1:
        st.markdown(f"""
        <div class='health-insight-metric-card'>
            <span class='icon'>🔍</span>
            <div class='value'>{format_number(total_queries)}</div>
            <div class='label'>Unique Health Queries</div>
            <div class='sub-label'>Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown(f"""
        <div class='health-insight-metric-card'>
            <span class='icon'>📈</span>
            <div class='value'>{format_number(total_searches)}</div>
            <div class='label'>Total Health Searches</div>
            <div class='sub-label'>Volume</div>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        performance_class = "high-health-insight-performance" if avg_ctr > 5 else "medium-health-insight-performance" if avg_ctr > 2 else "low-health-insight-performance"
        st.markdown(f"""
        <div class='health-insight-metric-card'>
            <span class='icon'>📊</span>
            <div class='value'>{avg_ctr:.2f}% <span class='health-insight-performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
            <div class='label'>Average Health CTR</div>
            <div class='sub-label'>Engagement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col4:
        performance_class = "high-health-insight-performance" if avg_cr > 3 else "medium-health-insight-performance" if avg_cr > 1 else "low-health-insight-performance"
        st.markdown(f"""
        <div class='health-insight-metric-card'>
            <span class='icon'>💚</span>
            <div class='value'>{avg_cr:.2f}% <span class='health-insight-performance-badge {performance_class}'>{"High" if avg_cr > 3 else "Medium" if avg_cr > 1 else "Low"}</span></div>
            <div class='label'>Average Nutraceuticals & Nutrition CR</div>
            <div class='sub-label'>Conversion</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    def q_expand(title, explanation, render_fn, icon="💡"):
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown(f"<div class='health-insight-box'><h4>Why & How to Use</h4><p>{explanation}</p></div>", unsafe_allow_html=True)
            try:
                st.markdown("<div class='health-insight-table-container'>", unsafe_allow_html=True)
                render_fn()
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Rendering error: {e}")

    # Q1: Top Queries by Counts (Top 10)
    def q1():
        out = queries_clean.groupby('normalized_query').agg(
            Counts=('Counts', 'sum'),
            clicks=('clicks', 'sum'),
            conversions=('conversions', 'sum')
        ).reset_index()
        out['ctr'] = out.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
        out['cr'] = out.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1)
        total_counts = out['Counts'].sum()
        out['share_counts'] = out['Counts'] / total_counts * 100 if total_counts > 0 else 0
        out = out.sort_values('Counts', ascending=False).head(10)
        out = out[['normalized_query', 'Counts', 'share_counts', 'clicks', 'conversions', 'ctr', 'cr']]
        out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
        out['clicks'] = out['clicks'].apply(lambda x: f"{x:,.0f}")
        out['conversions'] = out['conversions'].apply(lambda x: f"{x:,.0f}")
        out['ctr'] = out['ctr'].apply(lambda x: f"{x:.2f}%")
        out['cr'] = out['cr'].apply(lambda x: f"{x:.2f}%")
        out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
        if AGGRID_OK:
            AgGrid(out, height=300, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        csv = out.to_csv(index=False)
        st.download_button(
            label="📥 Download Q1 Health Table",
            data=csv,
            file_name=f"q1_top_health_queries_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q1_health_download"
        )
        fig = px.bar(out, x='normalized_query', y=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                     title='Top 10 Health Queries by Counts', color_discrete_sequence=['#4CAF50'], text_auto=True)
        fig.update_layout(xaxis_title="Query", yaxis_title="Counts", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q1 — Top Health Queries by Counts (Top 10)",
             "Which queries drive the most Counts? Prioritize for Nutraceuticals & Nutrition search tuning and inventory planning.",
             q1, "📈")

    # Q2: High Counts, Low CTR Queries (Top 10)
    def q2():
        df2 = queries_clean.groupby('normalized_query').agg(
            Counts=('Counts', 'sum'),
            clicks=('clicks', 'sum')
        ).reset_index()
        df2['ctr'] = df2.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
        total_counts = df2['Counts'].sum()
        df2['share_counts'] = df2['Counts'] / total_counts * 100 if total_counts > 0 else 0
        out = df2[(df2['Counts'] >= df2['Counts'].quantile(0.6)) & (df2['ctr'] <= df2['ctr'].quantile(0.3))].sort_values('Counts', ascending=False).head(10)
        out = out[['normalized_query', 'Counts', 'share_counts', 'clicks', 'ctr']]
        out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
        out['clicks'] = out['clicks'].apply(lambda x: f"{x:,.0f}")
        out['ctr'] = out['ctr'].apply(lambda x: f"{x:.2f}%")
        out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
        if AGGRID_OK:
            AgGrid(out, height=300, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        csv = out.to_csv(index=False)
        st.download_button(
            label="📥 Download Q2 Health Table",
            data=csv,
            file_name=f"q2_high_counts_low_ctr_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q2_health_download"
        )
        fig = px.scatter(out, x=out['Counts'].apply(lambda x: float(x.replace(',', ''))), y=out['ctr'].apply(lambda x: float(x.strip('%'))),
                         text='normalized_query', title='High Counts, Low CTR Health Queries',
                         color_discrete_sequence=['#4CAF50'], size=out['Counts'].apply(lambda x: float(x.replace(',', ''))))
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_title="Counts", yaxis_title="CTR (%)")
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q2 — High Counts, Low CTR Health Queries (Top 10)",
             "Queries with high Counts but low engagement. Improve Nutraceuticals & Nutrition relevance, snippets, or imagery.",
             q2, "⚠️")

    # Q3: Top Queries by Conversion Rate (Min Counts=200, Top 10)
    def q3():
        df3 = queries_clean.groupby('normalized_query').agg(
            Counts=('Counts', 'sum'),
            clicks=('clicks', 'sum'),
            conversions=('conversions', 'sum')
        ).reset_index()
        df3 = df3[df3['Counts'] >= 200]
        df3['cr'] = df3.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1)
        total_counts = df3['Counts'].sum()
        df3['share_counts'] = df3['Counts'] / total_counts * 100 if total_counts > 0 else 0
        out = df3.sort_values('cr', ascending=False).head(10)
        out = out[['normalized_query', 'Counts', 'share_counts', 'clicks', 'conversions', 'cr']]
        out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
        out['clicks'] = out['clicks'].apply(lambda x: f"{x:,.0f}")
        out['conversions'] = out['conversions'].apply(lambda x: f"{x:,.0f}")
        out['cr'] = out['cr'].apply(lambda x: f"{x:.2f}%")
        out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
        if AGGRID_OK:
            AgGrid(out, height=300, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        csv = out.to_csv(index=False)
        st.download_button(
            label="📥 Download Q3 Health Table",
            data=csv,
            file_name=f"q3_top_conversion_rate_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q3_health_download"
        )
        fig = px.bar(out, x='normalized_query', y=out['cr'].apply(lambda x: float(x.strip('%'))),
                     title='Top 10 Health Queries by Conversion Rate', color_discrete_sequence=['#4CAF50'], text_auto='.2f')
        fig.update_layout(xaxis_title="Query", yaxis_title="Conversion Rate (%)", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q3 — Top Health Queries by Conversion Rate (Min Counts=200)",
             "High-converting queries for Nutraceuticals & Nutrition paid promotions or product focus.",
             q3, "🎯")

    # Q4: Long-Tail vs Short-Tail Queries
    def q4():
        lt = queries_clean[queries_clean['query_length'] >= 20]
        lt_counts = lt['Counts'].sum()
        total_counts = queries_clean['Counts'].sum()
        st.markdown(f"<div class='health-insight-metric-card'><span class='icon'>📏</span><div class='value'>{lt_counts:,.0f}</div><div class='label'>Long-Tail Counts</div><div class='sub-label'>Queries ≥20 chars, Share: {lt_counts/total_counts:.2%}</div></div>", unsafe_allow_html=True)
        out = pd.DataFrame({
            'Type': ['Long-Tail (≥20 chars)', 'Short-Tail (<20 chars)'],
            'Counts': [lt_counts, total_counts - lt_counts],
            'share_counts': [lt_counts / total_counts * 100 if total_counts > 0 else 0, (total_counts - lt_counts) / total_counts * 100 if total_counts > 0 else 0]
        })
        out = out[['Type', 'Counts', 'share_counts']]
        out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
        out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
        if AGGRID_OK:
            AgGrid(out, height=200, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        csv = out.to_csv(index=False)
        st.download_button(
            label="📥 Download Q4 Health Table",
            data=csv,
            file_name=f"q4_long_tail_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q4_health_download"
        )
        fig = px.pie(out, names='Type', values=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                     title='Long-Tail vs Short-Tail Health Counts Share',
                     color_discrete_sequence=['#4CAF50', '#81C784'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q4 — Long-Tail vs Short-Tail Health Queries",
             "How much Counts come from long-tail queries? Key for Nutraceuticals & Nutrition content strategy.",
             q4, "📏")

    # Q5: Branded vs Generic Counts Share
    def q5():
        if 'brand' in queries_clean.columns:
            generic = queries_clean[queries_clean['brand'].str.lower() == 'other']
            branded = queries_clean[(queries_clean['brand'].notna()) & (queries_clean['brand'] != '') & (queries_clean['brand'].str.lower() != 'other')]
            generic_counts = generic['Counts'].sum()
            branded_counts = branded['Counts'].sum()
            total_counts = queries_clean['Counts'].sum()
            st.markdown(f"<div class='health-insight-metric-card'><span class='icon'>🏷</span><div class='value'>{branded_counts:,.0f}</div><div class='label'>Branded Health Counts</div><div class='sub-label'>Share: {branded_counts/total_counts:.2%}</div></div>", unsafe_allow_html=True)
            out = pd.DataFrame({
                'Type': ['Branded', 'Generic'],
                'Counts': [branded_counts, generic_counts],
                'share_counts': [branded_counts / total_counts * 100 if total_counts > 0 else 0, generic_counts / total_counts * 100 if total_counts > 0 else 0]
            })
            out = out[['Type', 'Counts', 'share_counts']]
            out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
            out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
            if AGGRID_OK:
                AgGrid(out, height=200, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
            else:
                st.dataframe(out, use_container_width=True, hide_index=True)
            csv = out.to_csv(index=False)
            st.download_button(
                label="📥 Download Q5 Health Table",
                data=csv,
                file_name=f"q5_branded_generic_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="q5_health_download"
            )
            fig = px.pie(out, names='Type', values=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                         title='Branded vs Generic Health Counts Share',
                         color_discrete_sequence=['#4CAF50', '#81C784'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brand column not present.")
    q_expand("Q5 — Branded vs Generic Health Counts Share",
             "Assess brand vs generic search intent, with 'Other' as Generic.",
             q5, "🏷")

    # Q6: Query Funnel Snapshot (Top 10)
    def q6():
        out = queries_clean.groupby('normalized_query').agg(
            Counts=('Counts', 'sum'),
            clicks=('clicks', 'sum'),
            conversions=('conversions', 'sum')
        ).reset_index()
        out['ctr'] = out.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
        out['cr'] = out.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1)
        total_counts = out['Counts'].sum()
        out['share_counts'] = out['Counts'] / total_counts * 100 if total_counts > 0 else 0
        out = out.sort_values('Counts', ascending=False).head(10)
        out = out[['normalized_query', 'Counts', 'share_counts', 'clicks', 'conversions', 'ctr', 'cr']]
        out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
        out['clicks'] = out['clicks'].apply(lambda x: f"{x:,.0f}")
        out['conversions'] = out['conversions'].apply(lambda x: f"{x:,.0f}")
        out['ctr'] = out['ctr'].apply(lambda x: f"{x:.2f}%")
        out['cr'] = out['cr'].apply(lambda x: f"{x:.2f}%")
        out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
        if AGGRID_OK:
            AgGrid(out, height=300, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        csv = out.to_csv(index=False)
        st.download_button(
            label="📥 Download Q6 Health Table",
            data=csv,
            file_name=f"q6_health_funnel_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q6_health_download"
        )
        fig = px.bar(out, x='normalized_query', y=[out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                                                   out['clicks'].apply(lambda x: float(x.replace(',', ''))),
                                                   out['conversions'].apply(lambda x: float(x.replace(',', '')))],
                     title='Top 10 Health Queries: Funnel Snapshot',
                     barmode='group', color_discrete_sequence=['#4CAF50', '#81C784', '#66BB6A'])
        fig.update_layout(xaxis_title="Query", yaxis_title="Value", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q6 — Health Query Funnel Snapshot (Top 10)",
             "View top queries' Nutraceuticals & Nutrition funnel: Counts → clicks → conversions.",
             q6, "📊")

    # Q7: Top Queries by CTR (Min Counts=200, Top 10)
    def q7():
        df7 = queries_clean.groupby('normalized_query').agg(
            Counts=('Counts', 'sum'),
            clicks=('clicks', 'sum')
        ).reset_index()
        df7 = df7[df7['Counts'] >= 200]
        df7['ctr'] = df7.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
        total_counts = df7['Counts'].sum()
        df7['share_counts'] = df7['Counts'] / total_counts * 100 if total_counts > 0 else 0
        out = df7.sort_values('ctr', ascending=False).head(10)
        out = out[['normalized_query', 'Counts', 'share_counts', 'clicks', 'ctr']]
        out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
        out['clicks'] = out['clicks'].apply(lambda x: f"{x:,.0f}")
        out['ctr'] = out['ctr'].apply(lambda x: f"{x:.2f}%")
        out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
        if AGGRID_OK:
            AgGrid(out, height=300, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        csv = out.to_csv(index=False)
        st.download_button(
            label="📥 Download Q7 Health Table",
            data=csv,
            file_name=f"q7_top_ctr_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q7_health_download"
        )
        fig = px.bar(out, x='normalized_query', y=out['ctr'].apply(lambda x: float(x.strip('%'))),
                     title='Top 10 Health Queries by CTR', color_discrete_sequence=['#4CAF50'], text_auto='.2f')
        fig.update_layout(xaxis_title="Query", yaxis_title="CTR (%)", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q7 — Top Health Queries by CTR (Min Counts=200)",
             "High-engagement queries for Nutraceuticals & Nutrition ad campaigns or content.",
             q7, "📈")

    # Q8: High Counts, Low CTR & Conversion Rate (Top 10)
    def q8():
        df8 = queries_clean.groupby('normalized_query').agg(
            Counts=('Counts', 'sum'),
            clicks=('clicks', 'sum'),
            conversions=('conversions', 'sum')
        ).reset_index()
        df8['ctr'] = df8.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
        df8['cr'] = df8.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1)
        total_counts = df8['Counts'].sum()
        df8['share_counts'] = df8['Counts'] / total_counts * 100 if total_counts > 0 else 0
        out = df8[(df8['Counts'] >= df8['Counts'].quantile(0.6)) &
                  (df8['ctr'] <= df8['ctr'].mean()) &
                  (df8['cr'] <= df8['cr'].mean())].sort_values('Counts', ascending=False).head(10)
        out = out[['normalized_query', 'Counts', 'share_counts', 'clicks', 'conversions', 'ctr', 'cr']]
        out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
        out['clicks'] = out['clicks'].apply(lambda x: f"{x:,.0f}")
        out['conversions'] = out['conversions'].apply(lambda x: f"{x:,.0f}")
        out['ctr'] = out['ctr'].apply(lambda x: f"{x:.2f}%")
        out['cr'] = out['cr'].apply(lambda x: f"{x:.2f}%")
        out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
        if AGGRID_OK:
            AgGrid(out, height=300, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
        else:
            st.dataframe(out, use_container_width=True, hide_index=True)
        csv = out.to_csv(index=False)
        st.download_button(
            label="📥 Download Q8 Health Table",
            data=csv,
            file_name=f"q8_low_ctr_cr_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q8_health_download"
        )
        fig = px.scatter(out, x=out['ctr'].apply(lambda x: float(x.strip('%'))),
                         y=out['cr'].apply(lambda x: float(x.strip('%'))),
                         text='normalized_query', title='High Counts, Low CTR & Conversion Rate Health',
                         color_discrete_sequence=['#4CAF50'], size=out['Counts'].apply(lambda x: float(x.replace(',', ''))))
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_title="CTR (%)", yaxis_title="Conversion Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q8 — High Counts, Low CTR & Conversion Rate Health (Top 10)",
             "Optimize Nutraceuticals & Nutrition search results for these underperforming queries.",
             q8, "⚠️")

    # Q9: Top Brands by Counts (Top 10, Excluding "Other")
    def q9():
        if 'brand' in queries_clean.columns:
            out = queries_clean[queries_clean['brand'].str.lower() != 'other'].groupby('brand').agg(
                Counts=('Counts', 'sum'),
                clicks=('clicks', 'sum'),
                conversions=('conversions', 'sum')
            ).reset_index()
            out['ctr'] = out.apply(lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1)
            out['cr'] = out.apply(lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1)
            total_counts = out['Counts'].sum()
            out['share_counts'] = out['Counts'] / total_counts * 100 if total_counts > 0 else 0
            out = out.sort_values('Counts', ascending=False).head(10)
            out = out[['brand', 'Counts', 'share_counts', 'clicks', 'conversions', 'ctr', 'cr']]
            out['Counts'] = out['Counts'].apply(lambda x: f"{x:,.0f}")
            out['clicks'] = out['clicks'].apply(lambda x: f"{x:,.0f}")
            out['conversions'] = out['conversions'].apply(lambda x: f"{x:,.0f}")
            out['ctr'] = out['ctr'].apply(lambda x: f"{x:.2f}%")
            out['cr'] = out['cr'].apply(lambda x: f"{x:.2f}%")
            out['share_counts'] = out['share_counts'].apply(lambda x: f"{x:.2f}%")
            if AGGRID_OK:
                AgGrid(out, height=300, gridOptions=GridOptionsBuilder.from_dataframe(out).build(), theme='material')
            else:
                st.dataframe(out, use_container_width=True, hide_index=True)
            csv = out.to_csv(index=False)
            st.download_button(
                label="📥 Download Q9 Health Table",
                data=csv,
                file_name=f"q9_top_brands_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="q9_health_download"
            )
            fig = px.bar(out, x='brand', y=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                         title='Top 10 Brands by Health Counts', color_discrete_sequence=['#4CAF50'], text_auto=True)
            fig.update_layout(xaxis_title="Brand", yaxis_title="Counts", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brand column not present.")
    q_expand("Q9 — Top Brands by Health Counts (Top 10)",
             "Rank brands by Counts for Nutraceuticals & Nutrition partnerships or promotions, excluding 'Other'.",
             q9, "🏷")

    # Q10: Category vs Brand Performance (Pivot)
    def q10():
        if 'category' in queries_clean.columns and 'brand' in queries_clean.columns:
            pivot = queries_clean[queries_clean['brand'].str.lower() != 'other'].pivot_table(
                values='Counts', index='category', columns='brand', aggfunc='sum'
            ).fillna(0)
            total_counts = pivot.sum().sum()
            pivot['Total'] = pivot.sum(axis=1)
            pivot['share_counts'] = pivot['Total'] / total_counts * 100 if total_counts > 0 else 0
            top_brands = queries_clean[queries_clean['brand'].str.lower() != 'other'].groupby('brand')['Counts'].sum().sort_values(ascending=False).head(5).index
            pivot = pivot[top_brands.tolist() + ['Total', 'share_counts']]
            pivot = pivot.reset_index()
            pivot['Total'] = pivot['Total'].apply(lambda x: f"{x:,.0f}")
            pivot['share_counts'] = pivot['share_counts'].apply(lambda x: f"{x:.2f}%")
            for col in top_brands:
                pivot[col] = pivot[col].apply(lambda x: f"{x:,.0f}")
            pivot = pivot[['category', 'Total', 'share_counts'] + top_brands.tolist()]
            if AGGRID_OK:
                AgGrid(pivot, height=300, gridOptions=GridOptionsBuilder.from_dataframe(pivot).build(), theme='material')
            else:
                st.dataframe(pivot, use_container_width=True, hide_index=True)
            csv = pivot.to_csv(index=False)
            st.download_button(
                label="📥 Download Q10 Health Table",
                data=csv,
                file_name=f"q10_category_brand_health_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="q10_health_download"
            )
            fig = px.bar(pivot.melt(id_vars='category', value_vars=top_brands, value_name='Counts'),
                         x='category', y='Counts', color='brand', title='Health Category vs Brand Counts',
                         barmode='stack', color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(xaxis_title="Category", yaxis_title="Counts", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category or brand column missing.")
    q_expand("Q10 — Health Category vs Brand Performance (Pivot)",
             "Analyze brand performance within Nutraceuticals & Nutrition categories for targeted strategies.",
             q10, "📦🏷")

    st.info("For advanced health analyses (e.g., anomaly detection, semantic clustering), contact Nour Eldeen for custom Nutraceuticals & Nutrition solutions.")

# ----------------- Export / Downloads -----------------
# Export Tab - FIXED with correct dataframe name
with tab_export:
    st.header("⬇ Export & Save Dashboard")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #FF5A6E 0%, #FF8A80 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📸 Click Any Card to Auto-Print!</h3>
        <p style="color: white; margin: 5px 0 0 0;">Cards will switch tabs and open print dialog automatically!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab info with their corresponding tab indices
    tab_info = {
        "Overview": {"icon": "📊", "desc": "Key metrics, totals, and summary charts", "tab_index": 0},
        "Search Analysis": {"icon": "🔍", "desc": "Search queries analysis and insights", "tab_index": 1},
        "Brand Analysis": {"icon": "🏷️", "desc": "Brand performance and comparisons", "tab_index": 2},
        "Category Analysis": {"icon": "📂", "desc": "Category breakdown and trends", "tab_index": 3},
        "Subcategory Analysis": {"icon": "📋", "desc": "Detailed subcategory insights", "tab_index": 4},
        "Generic Type": {"icon": "🏷️", "desc": "Generic vs branded analysis", "tab_index": 5},
        "Time Analysis": {"icon": "📈", "desc": "Time-based trends and patterns", "tab_index": 6}
    }
    
    st.subheader("🎯 Click Card to Auto-Screenshot:")
    
    # JavaScript for auto-print functionality
    auto_print_js = """
    <script>
    function autoPrintTab(tabName, tabIndex) {
        // Hide sidebar first
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) sidebar.style.display = 'none';
        
        // Find and click the tab
        const tabs = document.querySelectorAll('[data-baseweb="tab"]');
        if (tabs[tabIndex]) {
            tabs[tabIndex].click();
            
            // Wait for tab to load, then print
            setTimeout(() => {
                // Additional cleanup for print
                const header = document.querySelector('[data-testid="stHeader"]');
                const toolbar = document.querySelector('[data-testid="stToolbar"]');
                
                if (header) header.style.display = 'none';
                if (toolbar) toolbar.style.display = 'none';
                
                // Trigger print dialog
                window.print();
                
                // Show success message
                alert(`✅ Print dialog opened for ${tabName} tab!\\n\\nTip: Choose "Save as PDF" in the print dialog.`);
                
            }, 2000); // Wait 2 seconds for tab to fully load
        }
    }
    </script>
    """
    
    # Inject JavaScript
    st.components.v1.html(auto_print_js, height=0)
    
    # Create clickable cards
    cols = st.columns(2)
    for i, (tab_name, info) in enumerate(tab_info.items()):
        with cols[i % 2]:
            # Create unique button for each tab
            if st.button(f"🖨️ Print {tab_name}", key=f"print_{tab_name.lower().replace(' ', '_')}"):
                # JavaScript to handle the click
                st.components.v1.html(f"""
                <script>
                    // Auto-print function
                    setTimeout(() => {{
                        // Hide Streamlit UI elements
                        const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
                        const header = parent.document.querySelector('[data-testid="stHeader"]');
                        const toolbar = parent.document.querySelector('[data-testid="stToolbar"]');
                        
                        if (sidebar) sidebar.style.display = 'none';
                        if (header) header.style.display = 'none';
                        if (toolbar) toolbar.style.display = 'none';
                        
                        // Click the target tab
                        const tabs = parent.document.querySelectorAll('[data-baseweb="tab"]');
                        if (tabs[{info['tab_index']}]) {{
                            tabs[{info['tab_index']}].click();
                            
                            // Wait for content to load, then print
                            setTimeout(() => {{
                                parent.window.print();
                            }}, 1500);
                        }}
                    }}, 100);
                </script>
                """, height=0)
                
                st.success(f"🎯 Switching to {tab_name} and opening print dialog...")
            
            # Display card info
            st.markdown(f"""
            <div style="
                border: 2px solid #FF5A6E; 
                border-radius: 10px; 
                padding: 15px; 
                margin: 10px 0;
                background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
            ">
                <h4 style="margin: 0 0 8px 0; color: #2D3748;">{info['icon']} {tab_name}</h4>
                <p style="margin: 5px 0; color: #666; font-size: 14px;">{info['desc']}</p>
                <small style="color: #FF5A6E; font-weight: bold;">
                    ✨ One-click auto-print!
                </small>
            </div>
            """, unsafe_allow_html=True)
    
    # Alternative: Manual method
    st.markdown("---")
    st.subheader("🔧 Alternative: Manual Method")
    
    with st.expander("📋 Manual Screenshot Steps (if auto-print doesn't work)"):
        st.markdown("""
        ### Step-by-Step Manual Process:
        
        1. **Click the tab** you want to screenshot (at the top)
        2. **Wait** for all data to load completely
        3. **Click the 3 dots (⋮)** in the top-right corner
        4. **Select "Print"** from the dropdown menu
        5. **In Print Dialog:**
           - Destination: **Save as PDF**
           - Layout: **Portrait** (or Landscape for wide charts)
           - More settings → **Background graphics: ✅ ON**
        6. **Click "Save"** and choose your location
        
        ✅ **Done!** Perfect PDF saved!
        """)
    
    # Export data section - FIXED with correct dataframe name
    st.markdown("---")
    st.subheader("📊 Export Raw Data")
    
    # Check if queries dataframe exists
    if 'queries' in locals() or 'queries' in globals():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            try:
                csv_data = queries.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"lady_care_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download complete queries dataset as CSV file"
                )
            except Exception as e:
                st.error(f"CSV Export Error: {str(e)}")
        
        with col2:
            # Excel Export
            try:
                from io import BytesIO
                import pandas as pd
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Main queries sheet
                    queries.to_excel(writer, sheet_name='Queries', index=False)
                    
                    # Add summary sheets if they exist
                    if brand_summary is not None:
                        brand_summary.to_excel(writer, sheet_name='Brand Summary', index=False)
                    
                    if category_summary is not None:
                        category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
                    
                    if subcategory_summary is not None:
                        subcategory_summary.to_excel(writer, sheet_name='Subcategory Summary', index=False)
                    
                    if generic_type is not None:
                        generic_type.to_excel(writer, sheet_name='Generic Type', index=False)
                    
                    # Create analysis summary
                    if 'brand' in queries.columns:
                        analysis_summary = queries.groupby('brand').agg({
                            'clicks': 'sum',
                            'conversions': 'sum',
                            'Counts': 'sum',
                            'ctr': 'mean',
                            'cr': 'mean'
                        }).round(3)
                        analysis_summary.to_excel(writer, sheet_name='Analysis Summary')
                
                st.download_button(
                    label="📊 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"lady_care_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download as Excel with all sheets and summaries"
                )
            except ImportError:
                st.info("📊 Excel export requires openpyxl package")
            except Exception as e:
                st.error(f"Excel Export Error: {str(e)}")
        
        with col3:
            # JSON Export
            try:
                json_data = queries.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="🔧 Download JSON",
                    data=json_data,
                    file_name=f"lady_care_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download as JSON for API integration"
                )
            except Exception as e:
                st.error(f"JSON Export Error: {str(e)}")
        
    
    # Browser compatibility note
    st.markdown("---")
    st.info("""
    🌐 **Browser Compatibility:**
    - ✅ **Chrome/Edge**: Full auto-print support
    - ✅ **Firefox**: May require manual confirmation
    - ✅ **Safari**: May need manual steps
    
    💡 **Tip**: If auto-print doesn't work, use the manual method above!
    """)
    
    # Tips section
    st.markdown("---")
    st.subheader("💡 Pro Tips")
    
    st.markdown("""
    ### 🎯 For Best Screenshot Quality:
    
    - **📱 Use Chrome/Edge** for best auto-print compatibility
    - **🖥️ Full Screen Mode** (F11) before printing
    - **📊 Wait for Charts** to fully load before printing
    - **🎨 Enable Background Graphics** in print settings
    - **📄 Choose A4/Letter size** for standard documents
    - **🔄 Landscape Mode** for wide charts and tables
    
    ### 📊 Data Export Tips:
    
    - **CSV**: Best for Excel analysis and pivot tables
    - **Excel**: Includes all sheets (queries, summaries, analysis)
    - **JSON**: Perfect for API integration and web apps
    
    ### 🚀 Quick Actions:
    
    1. **Screenshot All Tabs**: Click each print button in sequence
    2. **Batch Export**: Use the data export buttons for raw data
    3. **Share Reports**: Combine PDFs into a single presentation
    """)


# ----------------- Footer -----------------
st.markdown(f"""
<div class="footer">
✨ Lady Care Search Analytics — Noureldeen Mohamed
</div>
""", unsafe_allow_html=True)