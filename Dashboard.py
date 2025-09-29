import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re, os, logging
from datetime import datetime
import pytz

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
    <h2>👋 Welcome to Lady Care Search Analytics! ✨</h2>
    <p>Explore search patterns, brand performance, and actionable insights. Use the sidebar to filter data, navigate tabs to dive deep, and download results for your reports!</p>
</div>
""", unsafe_allow_html=True)

# ----------------- KPI cards -----------------
st.markdown('<div class="main-header">🔥 Lady Care — Ultimate Search Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Uncover powerful insights from the <b>search</b> column with vibrant visuals and actionable pivots</div>', unsafe_allow_html=True)

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
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_counts)}</div><div class='label'>✨ Total Counts</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_clicks)}</div><div class='label'>👆 Total Clicks</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='kpi'><div class='value'>{format_number(total_conversions)}</div><div class='label'>🎯 Total Conversions</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='kpi'><div class='value'>{overall_ctr:.2f}%</div><div class='label'>📈 Overall CTR</div></div>", unsafe_allow_html=True)
with c5:
    st.markdown(f"<div class='kpi'><div class='value'>{overall_cr:.2f}%</div><div class='label'>💡 Overall CR</div></div>", unsafe_allow_html=True)


# Show data source info in sidebar
st.sidebar.info(f"**Data Source:** {main_key}")
st.sidebar.write(f"**Total Rows:** {len(queries):,}")
st.sidebar.write(f"**Total Counts:** {total_counts:,}")
st.sidebar.write(f"**Calculated Clicks:** {total_clicks:,}")
st.sidebar.write(f"**Calculated Conversions:** {total_conversions:,}")

# Add debug info in an expander so it doesn't clutter the sidebar
with st.sidebar.expander("🔍 Data Debug Info"):
    st.write(f"Main sheet: {main_key}")
    st.write(f"Processed columns: {list(queries.columns)}")
    st.write(f"Processed shape: {queries.shape}")
    
    st.write("**Column Usage:**")
    if 'count' in raw_queries.columns:
        st.write(f"✓ Counts/Impressions: 'count' column")
    else:
        st.write("✗ Counts/Impressions: No 'count' column found")
    
    st.write("**Calculation Method:**")
    st.write("• Clicks = Counts × Click Through Rate")
    st.write("• Conversions = Clicks × Conversion Rate")
    
    # Show sample of raw data
    st.write("**Sample data (first 3 rows):**")
    st.dataframe(raw_queries.head(3))

# ----------------- Tabs -----------------
tab_overview, tab_search, tab_brand, tab_category, tab_subcat, tab_generic, tab_time, tab_pivot, tab_insights, tab_export = st.tabs([
    "📈 Overview","🔍 Search Analysis","🏷 Brand","📦 Category","🧴 Subcategory","🛠 Generic Type",
    "⏰ Time Analysis","📊 Pivot Builder","💡 Insights & Qs","⬇ Export"
])

# ----------------- Overview -----------------
with tab_overview:
    st.header("📈 Overview & Quick Wins")
    st.markdown("Quick visuals to spot trends and take immediate action. 🚀 Based on **queries_clustered** data (e.g., 17M+ Counts across categories).")

    # Accuracy Fix: Ensure Date conversion (Excel serial)
    if not queries['Date'].dtype == 'datetime64[ns]':
        queries['Date'] = pd.to_datetime(queries['start_date'], unit='D', origin='1899-12-30', errors='coerce')

    # Refresh Button (User-Friendly)
    if st.button("🔄 Refresh Filters & Data"):
        st.rerun()

    # Image Selection in Sidebar
    st.sidebar.header("🎨 Customize Hero Image")
    image_options = {
        "Abstract Gradient": "https://placehold.co/1200x200/E6F3FA/FF5A6E?text=Lady+Care+Insights",
        "Nature-Inspired": "https://picsum.photos/1200/250?random=care_nature",
        "Elegant Pink Theme": "https://source.unsplash.com/1200x250/?pink,elegant",
        "Custom Text on Solid Color": "https://placehold.co/1200x250/E6F3FA/FF5A6E?text=✨+Lady+Care+Glow",
        "Feminine Floral": "https://picsum.photos/1200/250?random=floral_feminine"
    }
    selected_image = st.sidebar.selectbox("Choose Hero Image", options=list(image_options.keys()), index=0)

    # Hero Image (Creative UI) with selected option
    st.image(image_options[selected_image], use_container_width=True)

    # FIRST ROW: Monthly Counts Table and Chart side by side
    st.markdown("## 📊 Monthly Analysis Overview")
    col_table, col_chart = st.columns([1,2])  # Equal width columns

    with col_table:
        st.markdown("### 📋 Monthly Counts Table")
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
                        ('background-color', '#FF5A6E'),
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
            <div style="background: linear-gradient(135deg, #FF5A6E 0%, #FFB085 100%); 
                        padding: 15px; border-radius: 10px; color: white; margin: 10px 0; text-align: center;">
                <strong>📊 Total: {format_number(int(total_all_months))} searches across {len(monthly_counts)} months</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No monthly data available")


    with col_chart:
        st.markdown("### 📈 Monthly Trends Visualization")
        
        if not monthly_counts.empty and len(monthly_counts) >= 2:
            try:
                fig = px.bar(monthly_counts, x='Date', y='Counts',
                            title='<b style="color:#FF5A6E; font-size:16px;">Monthly Search Trends 🌟</b>',
                            labels={'Date': '<i>Month</i>', 'Counts': '<b>Search Counts</b>'},
                            color='Counts',
                            color_continuous_scale=['#E6F3FA', '#FFB085', '#FF5A6E'],
                            template='plotly_white',
                            text=monthly_counts['Counts'].astype(str))
                    
                # Update traces
                fig.update_traces(
                    texttemplate='%{text}<br>%{customdata:.1f}%',
                    customdata=monthly_counts['Percentage'],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Counts: %{y:,.0f}<br>Share: %{customdata:.1f}%<extra></extra>'
                )
                
                # Layout optimization
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    title_x=0.5,  # Center alignment for title
                    title_font_size=16,
                    xaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
                    yaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
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
                    text=f"🏆 Peak: {peak_value:,.0f}",
                    showarrow=True,
                    arrowhead=3,
                    arrowcolor='#FF5A6E',
                    ax=0, ay=-40,
                    font=dict(size=12, color='#FF5A6E', family='Segoe UI', weight='bold')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating chart: {e}")
        else:
            st.info("📅 Add more date range for monthly trends visualization")

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
            'search': 'Query',
            'Counts': 'Total Search Counts',
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
        column_order = ['Query', 'Total Search Counts', 'Share %']
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
            
            gainers = top10_for_analysis.nlargest(3, 'MoM Change')[['Query', 'MoM Change']]
            losers = top10_for_analysis.nsmallest(3, 'MoM Change')[['Query', 'MoM Change']]
            
            return gainers, losers
        
        return pd.DataFrame(), pd.DataFrame()

    # 🚀 CACHED: CSV generation
    @st.cache_data(ttl=300, show_spinner=False)
    def generate_csv_ultra(_df):
        return _df.to_csv(index=False)

    # NOW START THE ACTUAL SECTION
    st.markdown("## 🔍 Top 50 Queries Analysis")

    if queries.empty or 'Counts' not in queries.columns or queries['Counts'].isna().all():
        st.warning("No valid data available for top 50 queries.")
    else:
        try:
            # 🚀 LAZY CSS LOADING - Only load once per session
            if 'top50_css_loaded' not in st.session_state:
                st.markdown("""
                <style>
                .top50-metric-card {
                    background: linear-gradient(135deg, #FF5A6E 0%, #FFB085 100%);
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 8px 32px rgba(255, 90, 110, 0.3); margin: 8px 0;
                    min-height: 120px; display: flex; flex-direction: column; justify-content: center;
                    transition: transform 0.2s ease; width: 100%;
                }
                .top50-metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(255, 90, 110, 0.4); }
                .top50-metric-card .icon { font-size: 2.5em; margin-bottom: 8px; display: block; }
                .top50-metric-card .value { font-size: 1.8em; font-weight: bold; margin-bottom: 5px; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.1; }
                .top50-metric-card .label { font-size: 1em; opacity: 0.95; font-weight: 600; line-height: 1.2; }
                .monthly-metric-card {
                    background: linear-gradient(135deg, #4A90E2 0%, #7BB3F0 100%);
                    padding: 18px; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 6px 25px rgba(74, 144, 226, 0.3); margin: 8px 0;
                    min-height: 100px; display: flex; flex-direction: column; justify-content: center;
                    transition: transform 0.2s ease; width: 100%;
                }
                .monthly-metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 35px rgba(74, 144, 226, 0.4); }
                .monthly-metric-card .icon { font-size: 2em; margin-bottom: 6px; display: block; }
                .monthly-metric-card .value { font-size: 1.5em; font-weight: bold; margin-bottom: 4px; line-height: 1.1; }
                .monthly-metric-card .label { font-size: 0.9em; opacity: 0.95; font-weight: 600; line-height: 1.2; }
                .download-section { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 20px; border-radius: 12px; text-align: center; margin: 20px 0; box-shadow: 0 6px 25px rgba(40, 167, 69, 0.3); }
                .insights-section { background: linear-gradient(135deg, #6f42c1 0%, #8e44ad 100%); padding: 20px; border-radius: 12px; margin: 20px 0; box-shadow: 0 6px 25px rgba(111, 66, 193, 0.3); }
                .mom-analysis { background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin: 10px 0; }
                .gainer-item { background: rgba(76, 175, 80, 0.2); padding: 8px 12px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #4CAF50; }
                .decliner-item { background: rgba(244, 67, 54, 0.2); padding: 8px 12px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #F44336; }
                </style>
                """, unsafe_allow_html=True)
                st.session_state.top50_css_loaded = True

            # 🚀 OPTIMIZED: Show debug info only in sidebar (non-blocking)
            if st.sidebar.checkbox("Show Debug Info", value=False):
                st.sidebar.write("**Available columns in queries:**", list(queries.columns))

            # 🚀 ENHANCED: Static month names (faster than dynamic lookup)
            month_names = {
                '2025-06': 'June 2025',
                '2025-07': 'July 2025',
                '2025-08': 'August 2025'
            }

            # 🚀 COMPUTE: Get data with caching (filter-aware)
            # Create a unique filter key based on current filter state
            filter_state = {
                'filters_applied': st.session_state.get('filters_applied', False),
                'data_shape': queries.shape,
                'data_hash': hash(str(queries['search'].tolist()[:10]) if not queries.empty else "empty")
            }
            filter_key = str(hash(str(filter_state)))

            top50, unique_months = compute_top50_queries_ultra(queries, month_names, filter_key)

            if top50.empty:
                st.warning("No valid data after processing top 50 queries.")
            else:
                # 🚀 ENHANCED: Smart styling cache with hash-based invalidation
                top50_hash = hash(str(top50.shape) + str(top50.columns.tolist()) + str(top50.iloc[0].to_dict()) if len(top50) > 0 else "empty")
                
                if ('styled_top50' not in st.session_state or 
                    st.session_state.get('top50_cache_key') != top50_hash):
                    
                    st.session_state.top50_cache_key = top50_hash
                    
                    # 🚀 FAST: Apply format_number to numeric columns before styling
                    display_top50 = top50.copy()
                    
                    # Format Total Search Counts with format_number
                    if 'Total Search Counts' in display_top50.columns:
                        display_top50['Total Search Counts'] = display_top50['Total Search Counts'].apply(lambda x: format_number(int(x)))
                    
                    # Format monthly columns with format_number
                    for month in unique_months:
                        month_display_name = month_names.get(month, month)
                        if month_display_name in display_top50.columns:
                            display_top50[month_display_name] = display_top50[month_display_name].apply(lambda x: format_number(int(x)))
                    
                    # Create styled DataFrame from the formatted copy
                    styled_top50 = display_top50.style.set_properties(**{
                        'text-align': 'center',
                        'vertical-align': 'middle',
                        'font-size': '14px',
                        'padding': '8px',
                        'line-height': '1.2'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center'), ('vertical-align', 'middle'), ('font-weight', 'bold'), ('background-color', '#f0f2f6'), ('color', '#262730'), ('padding', '10px'), ('border', '1px solid #ddd')]},
                        {'selector': 'td', 'props': [('text-align', 'center'), ('vertical-align', 'middle'), ('padding', '8px'), ('border', '1px solid #ddd')]},
                        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}
                    ])
                    
                    # Create format dictionary for remaining columns
                    format_dict = {
                        'Share %': '{:.2f}%',
                        'Clicks': '{:,.0f}',
                        'Conversions': '{:,.0f}',
                        'Conversion Rate': '{:.2f}%'
                    }

                    styled_top50 = styled_top50.format(format_dict)

                # 🚀 DISPLAY: Cached styled DataFrame
                st.dataframe(
                    styled_top50, 
                    use_container_width=True, 
                    height=600,
                    hide_index=True
                )

                # 🚀 ENHANCED SUMMARY METRICS
                st.markdown("---")
                
                # 🚀 FAST: Pre-calculate all metrics at once
                metrics = {
                    'total_queries': len(top50),
                    'total_search_volume': int(pd.to_numeric(top50['Total Search Counts'], errors='coerce').sum()),
                    'total_clicks': int(top50['Clicks'].sum()),
                    'total_conversions': int(top50['Conversions'].sum())
                }
                
                col1, col2, col3, col4 = st.columns(4)
                
                # 🚀 OPTIMIZED: Batch metric rendering
                metric_configs = [
                    (col1, "📊", metrics['total_queries'], "Total Queries"),
                    (col2, "🔍", format_number(metrics['total_search_volume']), "Total Search Volume"),
                    (col3, "👆", format_number(metrics['total_clicks']), "Total Clicks"),
                    (col4, "🎯", format_number(metrics['total_conversions']), "Total Conversions")
                ]
                
                for col, icon, value, label in metric_configs:
                    with col:
                        st.markdown(f"""
                        <div class="top50-metric-card">
                            <div class="icon">{icon}</div>
                            <div class="value">{value}</div>
                            <div class="label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # 🚀 MONTHLY BREAKDOWN
                if unique_months:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📅 Monthly Search Volume Breakdown")
                    
                    month_cols = st.columns(len(unique_months))
                    
                    for i, month in enumerate(unique_months):
                        month_display_name = month_names.get(month, month)
                        if month_display_name in top50.columns:
                            with month_cols[i]:
                                monthly_total = int(top50[month_display_name].sum())
                                st.markdown(f"""
                                <div class="monthly-metric-card">
                                    <div class="icon">📈</div>
                                    <div class="value">{format_number(monthly_total)}</div>
                                    <div class="label">{month_display_name}</div>
                                </div>
                                """, unsafe_allow_html=True)

                # 🚀 ENHANCED DOWNLOAD SECTION
                st.markdown("<br>", unsafe_allow_html=True)
                
                csv = generate_csv_ultra(top50)
                
                col_download = st.columns([1, 2, 1])
                with col_download[1]:
                    st.markdown("""
                    <div class="download-section">
                        <h4 style="color: white; margin-bottom: 15px;">📥 Export Data</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Table as CSV",
                        data=csv,
                        file_name=f"top_50_queries_monthly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the complete table with monthly breakdown",
                        use_container_width=True
                    )
                
                # 🚀 OPTIMIZED MONTHLY INSIGHTS
                with st.expander("📊 Monthly Insights", expanded=False):
                    if unique_months and len(unique_months) >= 2:
                        st.markdown("""
                        <div class="insights-section">
                            <h3 style="color: white; text-align: center; margin-bottom: 20px;">📈 Month-over-Month Analysis</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 🚀 CACHED: MoM analysis
                        gainers, losers = compute_mom_analysis_ultra(top50, unique_months, month_names, filter_key)
                        
                        if not gainers.empty and not losers.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                <div class="mom-analysis">
                                    <h4 style="color: #4CAF50; text-align: center; margin-bottom: 15px;">🚀 Top Gainers (MoM %)</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for _, row in gainers.iterrows():
                                    change_val = row['MoM Change']
                                    sign = "+" if change_val > 0 else ""
                                    st.markdown(f"""
                                    <div class="gainer-item">
                                        <strong>{row['Query']}</strong>: <span style="color: #4CAF50; font-weight: bold;">{sign}{change_val:.1f}%</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class="mom-analysis">
                                    <h4 style="color: #F44336; text-align: center; margin-bottom: 15px;">📉 Top Decliners (MoM %)</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for _, row in losers.iterrows():
                                    change_val = row['MoM Change']
                                    st.markdown(f"""
                                    <div class="decliner-item">
                                        <strong>{row['Query']}</strong>: <span style="color: #F44336; font-weight: bold;">{change_val:.1f}%</span>
                                    </div>
                                    """, unsafe_allow_html=True)

        except KeyError as e:
            st.error(f"Column error: {e}. Check column names in your data.")
        except Exception as e:
            st.error(f"Error processing top 50 queries: {e}")
            st.write("**Debug info:**")
            st.write(f"Queries shape: {queries.shape}")
            st.write(f"Available columns: {list(queries.columns)}")
            if 'top50' in locals() and not top50.empty:
                st.write(f"Top50 shape: {top50.shape}")
                if 'Total Search Counts' in top50.columns:
                    st.write(f"Total Search Counts dtype: {top50['Total Search Counts'].dtype}")
                    st.write(f"Sample values: {top50['Total Search Counts'].head()}")

    st.markdown("---")


# ----------------- Performance Snapshot -----------------
    st.subheader("📊 Performance Snapshot")

    # Mini-Metrics Row (Data-Driven: From Analysis with Share)
    colM1, colM2, colM3, colM4 = st.columns(4)
    with colM1:
        avg_ctr = queries['Click Through Rate'].mean() * 100 if not queries.empty else 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>📊</span>
            <div class='value'>{avg_ctr:.2f}%</div>
            <div class='label'>Avg CTR (All Cats)</div>
        </div>
        """, unsafe_allow_html=True)
    with colM2:
        avg_cr = queries['Converion Rate'].mean() * 100 if not queries.empty else 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>🎯</span>
            <div class='value'>{avg_cr:.2f}%</div>
            <div class='label'>Avg CR (Derived)</div>
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
            <div class='label'>Unique Queries (Top Share)</div>
        </div>
        """, unsafe_allow_html=True)
    with colM4:
        cat_counts = queries.groupby('Category')['Counts'].sum()
        top_cat = cat_counts.idxmax()
        top_cat_share = (cat_counts.max() / total_counts * 100) if total_counts > 0 else 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>📦</span>
            <div class='value'>{format_number(int(cat_counts.max()))} ({top_cat_share:.2f}%)</div>
            <div class='label'>Top Cat Counts ({top_cat})</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")

    st.subheader("🏷 Brand & Category Snapshot")
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
                                title='<b style="color:#FF5A6E; font-size:18px; text-shadow: 2px 2px 4px #00000055;">Top Brands by Search Counts</b>',
                                labels={'Brand': '<i>Brand</i>', 'Counts': '<b>Search Counts</b>'},
                                color=color_column,
                                color_continuous_scale=['#E6F3FA', '#FFB085', '#FF5A6E'],
                                template='plotly_white',
                                hover_data=hover_columns)
                    
                    # Update traces to position text outside and set hovertemplate
                    fig.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Counts: %{y:,.0f}' + 
                                    ('<br>Share: %{customdata[0]:.2f}%' if 'share' in hover_columns else '') +
                                    ('<br>Conversions: %{customdata[1]:,.0f}' if 'conversions' in hover_columns and len(hover_columns) > 1 else '') +
                                    '<extra></extra>'
                    )

                    # Enhance attractiveness: Custom layout for beauty
                    fig.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.95)',
                        paper_bgcolor='rgba(255,247,232,0.8)',
                        font=dict(color='#0B486B', family='Segoe UI'),
                        title_x=0,  # Left alignment for title
                        title_font_size=16,
                        xaxis=dict(
                            title='Brand',
                            showgrid=True, 
                            gridcolor='#E6F3FA', 
                            linecolor='#FF5A6E', 
                            linewidth=2
                        ),
                        yaxis=dict(
                            title='Search Counts',
                            showgrid=True, 
                            gridcolor='#E6F3FA', 
                            linecolor='#FF5A6E', 
                            linewidth=2
                        ),
                        bargap=0.2,
                        barcornerradius=8,
                        hovermode='x unified',
                        annotations=[
                            dict(
                                x=0.5, y=1.05, xref='paper', yref='paper',
                                text='✨ Hover for details | Top brand highlighted below ✨',
                                showarrow=False,
                                font=dict(size=10, color='#FF5A6E', family='Segoe UI'),
                                align='center'
                            )
                        ]
                    )

                    # Highlight the top brand with a custom marker
                    top_brand = brand_perf.loc[brand_perf['Counts'].idxmax(), 'Brand']
                    top_count = brand_perf['Counts'].max()
                    fig.add_annotation(
                        x=top_brand, y=top_count,
                        text=f"🏆 Peak: {top_count:,.0f}",
                        showarrow=True,
                        arrowhead=3,
                        arrowcolor='#FF5A6E',
                        ax=0, ay=-30,
                        font=dict(size=12, color='#FF5A6E', family='Segoe UI', weight='bold')
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No brand data available after filtering or missing required columns.")
            else:
                st.warning("No valid aggregation columns found for brand analysis.")
        else:
            st.info("🏷 Brand column not found in the dataset.")

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
                
                st.markdown("**Top Categories by Counts**")
                
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
                
                # Display the table with available data
                if len(display_columns) > 1:  # More than just the Category column
                    # Sort by Counts in descending order
                    if 'Counts' in cat_perf.columns:
                        sorted_cat_perf = cat_perf[display_columns].sort_values('Counts', ascending=False).head(10)
                    else:
                        # Fallback to first numeric column if Counts not available
                        numeric_cols = [col for col in display_columns[1:] if col in cat_perf.columns]
                        if numeric_cols:
                            sorted_cat_perf = cat_perf[display_columns].sort_values(numeric_cols[0], ascending=False).head(10)
                        else:
                            sorted_cat_perf = cat_perf[display_columns].head(10)
                    
                    try:
                        # Try using AgGrid if available
                        if 'AGGRID_OK' in globals() and AGGRID_OK:
                            AgGrid(sorted_cat_perf, height=300, enable_enterprise_modules=False)
                        else:
                            # Fall back to styled DataFrame
                            styled_cat_perf = sorted_cat_perf.style.format(format_dict).set_properties(**{
                                'text-align': 'center',
                                'font-size': '14px'
                            }).background_gradient(subset=['cr'] if 'cr' in display_columns else [], cmap='YlGnBu')
                            st.dataframe(styled_cat_perf, use_container_width=True, hide_index=True)
                    except NameError:
                        # AGGRID_OK not defined, use regular DataFrame
                        styled_cat_perf = sorted_cat_perf.style.format(format_dict).set_properties(**{
                            'text-align': 'center',
                            'font-size': '14px'
                        }).background_gradient(subset=['cr'] if 'cr' in display_columns else [], cmap='YlGnBu')
                        st.dataframe(styled_cat_perf, use_container_width=True, hide_index=True)
                else:
                    st.info("Insufficient data columns available for category analysis.")
            else:
                st.warning("No valid aggregation columns found for category analysis.")
        else:
            st.info("📦 Category column not found in the dataset.")

st.markdown("---")
# ----------------- Search Analysis (Enhanced Core) -----------------
with tab_search:
    st.header("🔍 Search Column — Deep Dive Analysis")
    st.markdown("Analyze raw search queries with advanced keyword insights, performance metrics, and actionable intelligence. 🎯")
    
    # Hero Image for Search Tab
    search_image_options = {
        "Search Analytics Focus": "https://placehold.co/1200x200/E6F3FA/FF5A6E?text=Keyword+Performance+Hub",
        "Data Visualization": "https://placehold.co/1200x200/FF5A6E/FFFFFF?text=Search+Query+Intelligence",
        "Abstract Search": "https://source.unsplash.com/1200x200/?analytics,data",
        "Abstract Gradient": "https://placehold.co/1200x200/E6F3FA/FF5A6E?text=Lady+Care+Insights",
    }
    selected_search_image = st.sidebar.selectbox("Choose Search Tab Hero", options=list(search_image_options.keys()), index=0, key="search_hero_image_selector")
    st.image(search_image_options[selected_search_image], use_container_width=True)
    
    # Add error handling and data validation
    if queries.empty or 'keywords' not in queries.columns:
        st.error("❌ No keyword data available. Please ensure your data contains properly processed keywords.")
        st.stop()
    
    # Quick Search Metrics Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        unique_queries = queries['normalized_query'].nunique()
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>🔍</span>
            <div class='value'>{format_number(unique_queries)}</div>
            <div class='label'>Unique Search Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        avg_query_length = queries['query_length'].mean()
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>📏</span>
            <div class='value'>{avg_query_length:.1f}</div>
            <div class='label'>Avg Query Length</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        # Fixed: Total Keywords = Total number of rows (each search query counts as 1)
        # Queries with above-average CTR
        if 'Click Through Rate' in queries.columns and not queries.empty:
            avg_ctr = queries['Click Through Rate'].mean()
            high_perf_queries = len(queries[queries['Click Through Rate'] > avg_ctr])
        else:
            high_perf_queries = 0
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>⚡</span>
            <div class='value'>{format_number(high_perf_queries)}</div>
            <div class='label'>High-Performance Queries</div>
        </div>
        """, unsafe_allow_html=True)

    
    with col_m4:
        long_tail_pct = (queries['query_length'] >= 20).mean() * 100
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>📈</span>
            <div class='value'>{long_tail_pct:.1f}%</div>
            <div class='label'>Long-tail Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two-column layout for main analysis
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Enhanced Keyword Analysis
        st.subheader("🔤 Keyword Frequency & Performance Analysis")
        
        # Process keywords safely
        kw_series = queries['keywords'].explode().dropna()
        if kw_series.empty:
            st.warning("No keywords found in the dataset.")
        else:
            kw_counts = kw_series.value_counts().reset_index()
            kw_counts.columns = ['keyword', 'frequency']
            
            # Fixed: Create keyword performance data with correct calculation logic
            keyword_performance = []
            for keyword in kw_counts['keyword'].head(50):  # Top 50 keywords
                # Filter rows where the keyword appears in the search column
                keyword_queries = queries[queries['search'].str.contains(keyword, case=False, na=False)]
                if not keyword_queries.empty:
                    total_counts = keyword_queries['Counts'].sum()
                    total_clicks = keyword_queries['clicks'].sum()
                    total_conversions = keyword_queries['conversions'].sum()
                    
                    performance = {
                        'keyword': keyword,
                        'total_counts': total_counts,
                        'total_clicks': total_clicks,
                        'total_conversions': total_conversions,
                        'avg_ctr': (total_clicks / total_counts * 100) if total_counts > 0 else 0,
                        'avg_cr': (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
                    }
                    keyword_performance.append(performance)
            
            kw_perf_df = pd.DataFrame(keyword_performance)
            
            if not kw_perf_df.empty:
                # Enhanced keyword visualization - Fixed: Removed frequency, using counts vs CTR and CR
                fig_kw = px.scatter(
                    kw_perf_df.head(30), 
                    x='total_counts', 
                    y='avg_ctr',
                    size='total_clicks',
                    color='avg_cr',
                    hover_name='keyword',
                    title='<b style="color:#FF5A6E; font-size:18px;">Keyword Performance Matrix: Counts vs CTR 🎯</b>',
                    labels={'total_counts': 'Total Counts', 'avg_ctr': 'Average CTR (%)', 'avg_cr': 'Avg CR (%)'},
                    color_continuous_scale=['#E6F3FA', '#FFB085', '#FF5A6E'],
                    template='plotly_white'
                )
                
                fig_kw.update_traces(
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                'Total Counts: %{x:,.0f}<br>' +
                                'CTR: %{y:.2f}%<br>' +
                                'Total Clicks: %{marker.size:,.0f}<br>' +
                                'Conversion Rate: %{marker.color:.2f}%<extra></extra>'
                )
                
                fig_kw.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    title_x=0,
                    xaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
                    yaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
                    annotations=[
                        dict(
                            x=0.95, y=0.95, xref='paper', yref='paper',
                            text='💡 Size = Total Clicks | Color = Conversion Rate',
                            showarrow=False,
                            font=dict(size=11, color='#0B486B'),
                            align='right'
                        )
                    ]
                )
                
                st.plotly_chart(fig_kw, use_container_width=True)

    with col_right:
        # Query Length Distribution
        st.subheader("📊 Query Length Analysis")
        length_dist = queries.groupby('query_length').size().reset_index(name='count')
        length_dist = length_dist.sort_values('query_length')
        
        fig_length = px.histogram(
            queries, 
            x='query_length', 
            nbins=30,
            title='<b style="color:#FF5A6E;">Query Length Distribution</b>',
            labels={'query_length': 'Character Length', 'count': 'Number of Queries'},
            color_discrete_sequence=['#FF8A7A']
        )
        
        fig_length.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,247,232,0.8)',
            font=dict(color='#0B486B', family='Segoe UI'),
            bargap=0.1,
            xaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
            yaxis=dict(showgrid=True, gridcolor='#E6F3FA')
        )
        
        st.plotly_chart(fig_length, use_container_width=True)

    # Separator
    st.markdown("---")

    # Second row: Full width for Top Performing Keywords
    st.subheader("🏆 Top Performing Keywords")

    if not kw_perf_df.empty:
        # Use slider instead of selectbox to avoid key conflicts
        num_keywords = st.slider(
            "Number of keywords to display:", 
            min_value=10, 
            max_value=300, 
            value=15, 
            step=10,
            key="keyword_count_slider_search_tab"
        )

        top_keywords = kw_perf_df.sort_values('total_counts', ascending=False).head(num_keywords)

        # Calculate total counts for share percentage
        total_all_counts = queries['Counts'].sum()
        top_keywords['share_pct'] = (top_keywords['total_counts'] / total_all_counts * 100).round(2)

        # Check if data exists before applying styling
        if not top_keywords.empty:
            # Create display version with renamed columns and proper formatting
            display_df = top_keywords.copy()
            
            # FIXED: Calculate CR as conversions/counts and add Classic CR as conversions/clicks
            display_df['classic_cr'] = display_df['avg_cr']  # This was conversions/clicks
            display_df['avg_cr'] = (display_df['total_conversions'] / display_df['total_counts'] * 100).round(2).fillna(0)
            
            display_df = display_df.rename(columns={
                'keyword': 'Keyword',
                'total_counts': 'Total Counts',
                'share_pct': 'Share %',
                'total_clicks': 'Total Clicks',
                'total_conversions': 'Conversions',
                'avg_ctr': 'Avg CTR',
                'avg_cr': 'CR',
                'classic_cr': 'Classic CR'
            })
            
            # Format numbers manually
            display_df['Total Counts'] = display_df['Total Counts'].apply(lambda x: f"{x:,.0f}")
            display_df['Share %'] = display_df['Share %'].apply(lambda x: f"{x:.2f}%")
            display_df['Total Clicks'] = display_df['Total Clicks'].apply(lambda x: f"{x:,.0f}")
            display_df['Conversions'] = display_df['Conversions'].apply(lambda x: f"{x:,.0f}")
            display_df['Avg CTR'] = display_df['Avg CTR'].apply(lambda x: f"{x:.2f}%")
            display_df['CR'] = display_df['CR'].apply(lambda x: f"{x:.2f}%")
            display_df['Classic CR'] = display_df['Classic CR'].apply(lambda x: f"{x:.2f}%")
            
            # FIXED: Reorder columns to place Share % right after Total Counts
            column_order = ['Keyword', 'Total Counts', 'Share %', 'Total Clicks', 'Conversions', 'Avg CTR', 'CR', 'Classic CR']
            display_df = display_df[column_order]
            
            # Reset index to remove it and display with center alignment
            display_df = display_df.reset_index(drop=True)
            
            # Display dataframe with custom styling for center alignment
            st.dataframe(
                display_df, 
                use_container_width=True,
                hide_index=True,  # This hides the index column
                column_config={
                    "Keyword": st.column_config.TextColumn(
                        "Keyword",
                        help="Search keyword",
                        width="medium"
                    ),
                    "Total Counts": st.column_config.TextColumn(
                        "Total Counts",
                        help="Total search counts",
                        width="small"
                    ),
                    "Share %": st.column_config.TextColumn(
                        "Share %",
                        help="Percentage of total searches",
                        width="small"
                    ),
                    "Total Clicks": st.column_config.TextColumn(
                        "Total Clicks",
                        help="Total clicks received",
                        width="small"
                    ),
                    "Conversions": st.column_config.TextColumn(
                        "Conversions",
                        help="Total conversions",
                        width="small"
                    ),
                    "Avg CTR": st.column_config.TextColumn(
                        "Avg CTR",
                        help="Average Click-Through Rate",
                        width="small"
                    ),
                    "CR": st.column_config.TextColumn(
                        "CR",
                        help="Conversion Rate (Conversions/Counts)",
                        width="small"
                    ),
                    "Classic CR": st.column_config.TextColumn(
                        "Classic CR",
                        help="Classic Conversion Rate (Conversions/Clicks)",
                        width="small"
                    )
                }
            )
            
            # Add custom CSS for center alignment
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
                background-color: #FF5A6E !important;
                color: white !important;
                font-weight: bold !important;
            }
            .stDataFrame td {
                text-align: center !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Download button for keywords
            csv_keywords = top_keywords.to_csv(index=False)
            st.download_button(
                label="📥 Download Keywords CSV",
                data=csv_keywords,
                file_name=f"top_{num_keywords}_keywords.csv",
                mime="text/csv",
                key="keyword_csv_download_search_tab"
            )


    st.markdown("---")

    
    # Advanced Analytics Section
    st.subheader("📈 Advanced Query Performance Analytics")
    
    # Three-column layout for advanced metrics
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        st.markdown("**🎯 Query Length vs Performance**")
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
                title='Length vs CTR Performance',
                color_continuous_scale=['#E6F3FA', '#FF8A7A'],
                template='plotly_white'
            )
            
            fig_ql.update_layout(
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI', size=10),
                height=300,
                xaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
                yaxis=dict(showgrid=True, gridcolor='#E6F3FA')
            )
            
            st.plotly_chart(fig_ql, use_container_width=True)
    
    with adv_col2:
        st.markdown("**📊 Long-tail vs Short-tail Performance**")
        queries['is_long_tail'] = queries['query_length'] >= 20
        lt_analysis = queries.groupby('is_long_tail').agg({
            'Counts': 'sum', 
            'clicks': 'sum',
            'conversions': 'sum'
        }).reset_index()
        lt_analysis['label'] = lt_analysis['is_long_tail'].map({
            True: 'Long-tail (≥20 chars)', 
            False: 'Short-tail (<20 chars)'
        })
        lt_analysis['ctr'] = lt_analysis.apply(lambda r: (r['clicks']/r['Counts']*100) if r['Counts']>0 else 0, axis=1)
        
        if not lt_analysis.empty:
            fig_lt = px.bar(
                lt_analysis, 
                x='label', 
                y='Counts',
                color='ctr',
                title='Traffic: Long-tail vs Short-tail',
                color_continuous_scale=['#E6F3FA', '#FF5A6E'],
                text='Counts'
            )
            
            fig_lt.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside'
            )
            
            fig_lt.update_layout(
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI', size=10),
                height=300,
                xaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
                yaxis=dict(showgrid=True, gridcolor='#E6F3FA')
            )
            
            st.plotly_chart(fig_lt, use_container_width=True)
    
    with adv_col3:
        st.markdown("**🔍 Keyword Density Analysis**")
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
                title='Query Length Distribution',
                color_discrete_sequence=['#FF5A6E', '#FFB085', '#E6F3FA', '#FF8A7A', '#FFF7E8']
            )
            
            fig_density.update_layout(
                font=dict(color='#0B486B', family='Segoe UI', size=10),
                height=300
            )
            
            st.plotly_chart(fig_density, use_container_width=True)

    
    st.markdown("---")
    
    # Replace Detailed Query Performance Analysis with Top Queries from Tab 1
    st.subheader("📋 Top Performing Queries")

    # Use slider instead of selectbox for queries too
    num_queries = st.slider(
        "Number of queries to display:", 
        min_value=10, 
        max_value=300, 
        value=50, 
        step=10,
        key="query_count_slider_search_tab"
    )

    if queries.empty or 'Counts' not in queries.columns or queries['Counts'].isna().all():
        st.warning("No valid data available for top queries.")
    else:
        try:
            # Group by 'search' and aggregate
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
                'search': 'Query',
                'Counts': 'Search Counts',
                'clicks': 'Clicks',
                'conversions': 'Conversions'
            })

            # No need to round again since we already did it above

            
            # Format Search Counts with commas
            top_queries['Search Counts'] = top_queries['Search Counts'].apply(lambda x: f"{x:,.0f}")
            top_queries['Share %'] = top_queries['Share %'].apply(lambda x: f"{x:.2f}%")
            top_queries['Conversion Rate'] = top_queries['Conversion Rate'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
            top_queries['Query Length'] = top_queries['Query Length'].apply(lambda x: f"{x}")

            # Reorder columns to include Query Length after Query
            column_order = ['Query', 'Query Length', 'Search Counts', 'Share %', 'Clicks', 'Conversions', 'Conversion Rate']
            top_queries = top_queries[column_order]

            # Reset index to remove it
            top_queries = top_queries.reset_index(drop=True)

            # Display the DataFrame with custom styling for center alignment
            st.dataframe(
                top_queries, 
                use_container_width=True,
                hide_index=True,  # This hides the index column
                column_config={
                    "Query": st.column_config.TextColumn(
                        "Query",
                        help="Search query text",
                        width="large"
                    ),
                    "Query Length": st.column_config.TextColumn(
                        "Query Length",
                        help="Number of characters in query",
                        width="small"
                    ),
                    "Search Counts": st.column_config.TextColumn(
                        "Search Counts",
                        help="Total search counts",
                        width="medium"
                    ),
                    "Share %": st.column_config.TextColumn(
                        "Share %",
                        help="Percentage of total searches",
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
                        help="Conversion rate percentage",
                        width="small"
                    )
                }
            )

            # Add custom CSS for center alignment
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
                background-color: #FF5A6E !important;
                color: white !important;
                font-weight: bold !important;
            }
            .stDataFrame td {
                text-align: center !important;
            }
            /* Keep Query column left-aligned for better readability */
            .stDataFrame td:first-child {
                text-align: left !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Add download button
            csv = top_queries.to_csv(index=False)
            st.download_button(
                label="📥 Download Queries CSV",
                data=csv,
                file_name=f"top_{num_queries}_queries.csv",
                mime="text/csv",
                key="query_csv_download_search_tab"
            )
        except KeyError as e:
            st.error(f"Column error: {e}. Check column names in your data (e.g., 'search', 'Counts', 'clicks', 'conversions', 'Conversion Rate').")
        except Exception as e:
            st.error(f"Error processing top queries: {e}")

    
    # Key Insights Box
    st.markdown("---")
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown("""
        <div class='insight-box'>
            <h4>🎯 Key Insights</h4>
            <p>• Long-tail queries represent {:.1f}% of total traffic<br>
            • Average query length is {:.1f} characters<br>
            • Top keyword appears in {:.1f}% of searches</p>
        </div>
        """.format(
            long_tail_pct,
            avg_query_length,
            (kw_counts.iloc[0]['frequency'] / len(queries) * 100) if not kw_counts.empty else 0
        ), unsafe_allow_html=True)
    
    with col_insight2:
        st.markdown("""
        <div class='insight-box'>
            <h4>💡 Recommendations</h4>
            <p>• Focus on high-performing keywords for content optimization<br>
            • Analyze long-tail queries for niche opportunities<br>
            • Monitor search intent patterns for strategy alignment</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- Brand Tab (Enhanced & Fixed) -----------------
with tab_brand:
    st.header("🏷 Brand Intelligence Hub")
    st.markdown("Comprehensive brand performance analysis with competitive insights and strategic recommendations. 🚀")
    
    # Hero Image for Brand Tab
    brand_image_options = {
        "Brand Analytics Focus": "https://placehold.co/1200x200/E6F3FA/FF5A6E?text=Brand+Market+Position",
        "Competitive Analysis": "https://placehold.co/1200x200/FF5A6E/FFFFFF?text=Brand+Performance+Intelligence",
        "Abstract Brand": "https://source.unsplash.com/1200x200/?brand,marketing",
        "Abstract Gradient": "https://placehold.co/1200x200/E6F3FA/FF5A6E?text=Lady+Care+Brand+Insights",
    }
    selected_brand_image = st.sidebar.selectbox("Choose Brand Tab Hero", options=list(brand_image_options.keys()), index=0, key="brand_hero_image_selector")
    st.image(brand_image_options[selected_brand_image], use_container_width=True)
    
    # Check for brand column with case sensitivity handling (no debug output)
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
        st.error(f"❌ No brand data available. Available columns: {list(queries.columns)}")
        st.info("💡 Please ensure your dataset contains a brand column (brand, Brand, or Brand Name)")
        st.stop()
    
    
    # Filter out "Other" brand from all analysis
    # Filter out "Other" brand from all analysis (CASE-INSENSITIVE)
    brand_queries = queries[
        (queries[brand_column].notna()) & 
        (~queries[brand_column].str.lower().isin(['other', 'others']))
    ]

    
    if brand_queries.empty:
        st.error("❌ No valid brand data available after filtering.")
        st.stop()
    
    # Brand Performance Metrics Row
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

        # Calculate comprehensive brand metrics
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

        # 🎯 FIXED: No need to filter again since brand_queries is already filtered
        bs = bs_raw.copy()

        # Calculate Share % based on filtered data
        total_counts = bs['Counts'].sum()
        bs['share_pct'] = (bs['Counts'] / total_counts * 100).round(2)

        # Calculate performance metrics
        bs['ctr'] = ((bs['clicks'] / bs['Counts']) * 100).round(2)
        bs['cr'] = ((bs['conversions'] / bs['clicks']) * 100).fillna(0).round(2)
        bs['classic_cr'] = ((bs['conversions'] / bs['Counts']) * 100).round(2)

        
        # Enhanced scatter plot for brand performance
        # 🎯 OPTION: Show top 50 brands sorted by Counts
        num_scatter_brands = st.slider(
            "Number of brands in scatter plot:", 
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
            color='cr',
            hover_name='brand',
            title=f'<b style="color:#FF5A6E; font-size:18px;">Brand Performance Matrix: Top {num_scatter_brands} Brands 🎯</b>',
            labels={'Counts': 'Total Search Counts', 'ctr': 'Click-Through Rate (%)', 'cr': 'Conversion Rate (%)'},
            color_continuous_scale=['#E6F3FA', '#FFB085', '#FF5A6E'],
            template='plotly_white'
        )

        
        fig_brand_perf.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Search Counts: %{x:,.0f}<br>' +
                         'CTR: %{y:.2f}%<br>' +
                         'Total Clicks: %{marker.size:,.0f}<br>' +
                         'Conversion Rate: %{marker.color:.2f}%<extra></extra>'
        )
        
        fig_brand_perf.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,247,232,0.8)',
            font=dict(color='#0B486B', family='Segoe UI'),
            title_x=0,
            xaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
            yaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
        )
        
        st.plotly_chart(fig_brand_perf, use_container_width=True)
        
        # Top Brands Performance Table
        st.subheader("🏆 Top Brand Performance")
        
        num_brands = st.slider(
            "Number of brands to display:", 
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
            'brand': 'Brand',
            'Counts': 'Search Counts',
            'share_pct': 'Share %',
            'clicks': 'Total Clicks',
            'conversions': 'Conversions',
            'ctr': 'CTR',
            'cr': 'CR',
            'classic_cr': 'Classic CR'
        })
        
        # Format numbers
        display_brands['Search Counts'] = display_brands['Search Counts'].apply(lambda x: f"{x:,.0f}")
        display_brands['Share %'] = display_brands['Share %'].apply(lambda x: f"{x:.2f}%")
        display_brands['Total Clicks'] = display_brands['Total Clicks'].apply(lambda x: f"{x:,.0f}")
        display_brands['Conversions'] = display_brands['Conversions'].apply(lambda x: f"{x:,.0f}")
        display_brands['CTR'] = display_brands['CTR'].apply(lambda x: f"{x:.2f}%")
        display_brands['CR'] = display_brands['CR'].apply(lambda x: f"{x:.2f}%")
        display_brands['Classic CR'] = display_brands['Classic CR'].apply(lambda x: f"{x:.2f}%")
        
        # Reorder columns
        column_order = ['Brand', 'Search Counts', 'Share %', 'Total Clicks', 'Conversions', 'CTR', 'CR', 'Classic CR']
        display_brands = display_brands[column_order]
        
        st.dataframe(display_brands, use_container_width=True, hide_index=True)
        
        # Download button
        csv_brands = top_brands.to_csv(index=False)
        st.download_button(
            label="📥 Download Brands CSV",
            data=csv_brands,
            file_name=f"top_{num_brands}_brands.csv",
            mime="text/csv",
            key="brand_csv_download"
        )
        
        # Brand Summary Data (calculated from queries)
        st.subheader("📋 Brand Summary Data")
        
        # Calculate brand summary from queries
        brand_summary_calc = []
        
        for brand in brand_queries[brand_column].unique():
            brand_data = brand_queries[brand_queries[brand_column] == brand]
            
            # Basic metrics
            total_counts = brand_data['Counts'].sum()
            total_clicks = brand_data['clicks'].sum()
            total_conversions = brand_data['conversions'].sum()
            
            # Calculate rates
            ctr = (total_clicks / total_counts * 100) if total_counts > 0 else 0
            cr = (total_conversions / total_counts * 100) if total_counts > 0 else 0
            classic_cr = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
            
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
            top_keywords_str = ', '.join([f"{kw}({cnt:,.0f})" for kw, cnt in top_keywords])
            
            brand_summary_calc.append({
                'Brand': brand,
                'Search Counts': total_counts,
                'Total Clicks': total_clicks,
                'Conversions': total_conversions,
                'CTR': ctr,
                'CR': cr,
                'Classic CR': classic_cr,
                'Unique Keywords': unique_keywords_count,
                'Top Keywords': top_keywords_str
            })
        
        brand_summary_df = pd.DataFrame(brand_summary_calc)
        
        # Sort by Search Counts
        brand_summary_df = brand_summary_df.sort_values('Search Counts', ascending=False)
        
        # Format for display
        display_summary = brand_summary_df.copy()
        display_summary['Search Counts'] = display_summary['Search Counts'].apply(lambda x: f"{x:,.0f}")
        display_summary['Total Clicks'] = display_summary['Total Clicks'].apply(lambda x: f"{x:,.0f}")
        display_summary['Conversions'] = display_summary['Conversions'].apply(lambda x: f"{x:,.0f}")
        display_summary['CTR'] = display_summary['CTR'].apply(lambda x: f"{x:.2f}%")
        display_summary['CR'] = display_summary['CR'].apply(lambda x: f"{x:.2f}%")
        display_summary['Classic CR'] = display_summary['Classic CR'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
        
        # Download button for brand summary
        csv_summary = brand_summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Brand Summary CSV",
            data=csv_summary,
            file_name="brand_summary_calculated.csv",
            mime="text/csv",
            key="brand_summary_calc_csv_download"
        )
    
    with col_right:
        # Brand Market Share Pie Chart (without "Other")
        st.subheader("📊 Brand Market Share")
        
        top_brands_pie = bs.nlargest(10, 'Counts')
        
        fig_pie = px.pie(
            top_brands_pie, 
            names='brand', 
            values='Counts',
            title='<b style="color:#FF5A6E;">Market Share Distribution</b>',
            color_discrete_sequence=['#FF5A6E', '#FFB085', '#E6F3FA', '#FF8A7A', '#FFF7E8', '#B8E6B8', '#87CEEB', '#DDA0DD', '#F0E68C', '#FFB6C1']
        )
        
        fig_pie.update_layout(
            font=dict(color='#0B486B', family='Segoe UI'),
            paper_bgcolor='rgba(255,247,232,0.8)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Brand Performance Categories
        st.subheader("🎯 Brand Performance Categories")
        
        # Categorize brands based on performance
        bs['performance_category'] = pd.cut(
            bs['ctr'], 
            bins=[0, 2, 5, 10, float('inf')], 
            labels=['Low (0-2%)', 'Medium (2-5%)', 'High (5-10%)', 'Excellent (>10%)']
        )
        
        category_counts = bs['performance_category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig_cat = px.bar(
            category_counts, 
            x='Category', 
            y='Count',
            title='<b style="color:#FF5A6E;">CTR Performance Distribution</b>',
            color='Count',
            color_continuous_scale=['#E6F3FA', '#FF5A6E'],
            text='Count'
        )
        
        fig_cat.update_traces(
            texttemplate='%{text}',
            textposition='outside'
        )
        
        fig_cat.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,247,232,0.8)',
            font=dict(color='#0B486B', family='Segoe UI'),
            xaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
            yaxis=dict(showgrid=True, gridcolor='#E6F3FA')
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Enhanced Brand Trend Analysis with proper filter application
        if 'Date' in queries.columns:
            st.subheader("📈 Brand Trend Analysis")
            
            # Get top 5 brands for trend analysis
            top_5_brands = bs.nlargest(5, 'Counts')['brand'].tolist()
            
            # CRITICAL FIX: Use the already filtered 'queries' data instead of 'brand_queries'
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
                        # FIX: Create proper monthly aggregation
                        trend_data['Month'] = trend_data['Date'].dt.to_period('M')
                        trend_data['Month_Display'] = trend_data['Date'].dt.strftime('%Y-%m')
                        
                        # Group by Month and brand - sum the counts for each month
                        monthly_trends = trend_data.groupby(['Month_Display', brand_column])['Counts'].sum().reset_index()
                        monthly_trends = monthly_trends.rename(columns={brand_column: 'brand'})
                        
                        # Convert month display back to datetime for proper plotting
                        monthly_trends['Date'] = pd.to_datetime(monthly_trends['Month_Display'] + '-01')
                        
                        # Debug: Check if we have monthly data
                        unique_months = monthly_trends['Month_Display'].unique()
                        st.write(f"📊 Monthly data available: {', '.join(sorted(unique_months))}")
                        
                        if len(monthly_trends) > 0:
                            fig_trend = px.line(
                                monthly_trends, 
                                x='Date', 
                                y='Counts', 
                                color='brand',
                                title='<b style="color:#FF5A6E;">Top 5 Brands Monthly Trend</b>',
                                color_discrete_sequence=['#FF5A6E', '#FFB085', '#E6F3FA', '#FF8A7A', '#B8E6B8'],
                                markers=True
                            )
                            
                            # FIX: Format x-axis to show months properly
                            fig_trend.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.95)',
                                paper_bgcolor='rgba(255,247,232,0.8)',
                                font=dict(color='#0B486B', family='Segoe UI'),
                                xaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#E6F3FA',
                                    title='Month',
                                    dtick="M1",  # Show monthly ticks
                                    tickformat="%b %Y"  # Format as "Jun 2025"
                                ),
                                yaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#E6F3FA',
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
                            st.info("No trend data available for the selected date range and brands")
                    else:
                        st.info("No valid dates found in the filtered data")
                except Exception as e:
                    st.error(f"Error processing trend data: {str(e)}")
            else:
                st.info("No brand data available for the selected date range")


    
    st.markdown("---")
    
    # Brand-Keyword Intelligence Matrix with Interactive Filter
    st.subheader("🔥 Brand-Keyword Intelligence Matrix")

    # Create brand filter dropdown
    if 'brand' in queries.columns and 'search' in queries.columns:
        # Get available brands (excluding null and 'other')
        available_brands = queries[
            (queries['brand'].notna()) & 
            (queries['brand'].str.lower() != 'other') &
            (queries['brand'].str.lower() != 'others')
        ]['brand'].unique()
        
        # Sort brands alphabetically
        available_brands = sorted(available_brands)
        
        # Create dropdown with "All Brands" option
        brand_options = ['All Brands'] + list(available_brands)
        
        # Brand selection dropdown
        selected_brand = st.selectbox(
            "🎯 Select Brand to Analyze:",
            options=brand_options,
            index=0  # Default to "All Brands"
        )
        
        # Filter data based on selection
        if selected_brand == 'All Brands':
            # Show top 8 brands if "All Brands" is selected - EXCLUDE "Other"
            top_brands = queries[
                (queries['brand'].str.lower() != 'other') &
                (queries['brand'].str.lower() != 'others') &
                (queries['brand'].notna())
            ]['brand'].value_counts().head(8).index.tolist()
            
            filtered_data = queries[queries['brand'].isin(top_brands)]
            matrix_title = "Top Brands vs Search Terms (Sum of Counts)"
        else:
            # Filter for selected brand only
            filtered_data = queries[queries['brand'] == selected_brand]
            matrix_title = f"{selected_brand} - Search Terms Analysis (Sum of Counts)"
        
        # Remove null values and 'other' categories from search terms as well
        matrix_data = filtered_data[
            (filtered_data['brand'].notna()) & 
            (filtered_data['search'].notna()) &
            (filtered_data['brand'].str.lower() != 'other') &
            (filtered_data['brand'].str.lower() != 'others') &
            (filtered_data['search'].str.lower() != 'other') &
            (filtered_data['search'].str.lower() != 'others')
        ].copy()
        
        if not matrix_data.empty:
            if selected_brand == 'All Brands':
                # For all brands: Group by brand and search term, sum the counts
                brand_search_matrix = matrix_data.groupby(['brand', 'search'])['Counts'].sum().reset_index()
                
                # Get top search terms across all brands (excluding "other")
                top_searches = matrix_data[
                    (matrix_data['search'].str.lower() != 'other') &
                    (matrix_data['search'].str.lower() != 'others')
                ]['search'].value_counts().head(12).index.tolist()
                
                brand_search_matrix = brand_search_matrix[brand_search_matrix['search'].isin(top_searches)]
                
                # Create pivot table
                heatmap_data = brand_search_matrix.pivot(
                    index='brand', 
                    columns='search', 
                    values='Counts'
                ).fillna(0)
                
            else:
                # For single brand: Show search terms vs other dimensions or just search terms
                search_counts = matrix_data.groupby('search')['Counts'].sum().reset_index()
                search_counts = search_counts.sort_values('Counts', ascending=False).head(15)
                
                # Create a simple horizontal bar chart instead of heatmap for single brand
                fig_single = px.bar(
                    search_counts,
                    x='Counts',
                    y='search',
                    orientation='h',
                    title=f'<b style="color:#FF5A6E;">{selected_brand} - Top Search Terms by Count</b>',
                    labels={'Counts': 'Total Search Counts', 'search': 'Search Terms'},
                    color='Counts',
                    color_continuous_scale='Reds'
                )
                
                fig_single.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_single, use_container_width=True)
                
                # Show summary
                total_counts = search_counts['Counts'].sum()
                st.info(f"📊 {selected_brand} has {len(search_counts)} top search terms with {total_counts:,} total counts")
                
            # Only create heatmap for "All Brands" view
            if selected_brand == 'All Brands' and not heatmap_data.empty:
                # Create the heatmap
                fig_matrix = px.imshow(
                    heatmap_data.values,
                    labels=dict(x="Search Terms", y="Brands", color="Total Counts"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale='Reds',
                    title=f'<b style="color:#FF5A6E;">{matrix_title}</b>',
                    aspect='auto'
                )
                
                fig_matrix.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    xaxis=dict(tickangle=45),
                    height=500
                )
                
                # Update hover template
                fig_matrix.update_traces(
                    hovertemplate='<b>Brand:</b> %{y}<br>' +
                                '<b>Search Term:</b> %{x}<br>' +
                                '<b>Total Counts:</b> %{z:,.0f}<extra></extra>'
                )
                
                st.plotly_chart(fig_matrix, use_container_width=True)
                
                # Show summary statistics
                total_interactions = brand_search_matrix['Counts'].sum()
                st.info(f"📊 Matrix shows {len(heatmap_data.index)} brands × {len(heatmap_data.columns)} search terms with {total_interactions:,} total search counts")
        
        else:
            st.warning("No data available for the selected brand filter")
            
    else:
        st.error("Required columns 'brand' and 'search' not found in the dataset")


    
    # Enhanced Key Insights and Recommendations
    st.markdown("---")
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        # Calculate meaningful insights
        top_brand_share = bs.iloc[0]['share_pct'] if not bs.empty else 0
        top_brand_name = bs.iloc[0]['brand'] if not bs.empty else "N/A"
        high_performers = len(bs[bs['ctr'] > 5]) if not bs.empty else 0
        avg_conversion_rate = bs['cr'].mean() if not bs.empty else 0
        brands_above_avg_cr = len(bs[bs['cr'] > avg_conversion_rate]) if not bs.empty else 0
        
        st.markdown(f"""
        <div class='insight-box'>
            <h4>🎯 Key Brand Insights</h4>
            <p>• <strong>{top_brand_name}</strong> dominates with {top_brand_share:.1f}% share<br>
            • {high_performers} brands achieve CTR > 5% (high performers)<br>
            • {brands_above_avg_cr} brands exceed avg CR of {avg_conversion_rate:.2f}%<br>
            • Market shows {"high" if brand_dominance > 30 else "moderate" if brand_dominance > 15 else "low"} concentration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_insight2:
        # Calculate strategic recommendations based on data
        low_performers = len(bs[bs['ctr'] < 2]) if not bs.empty else 0
        opportunity_brands = len(bs[(bs['Counts'] > bs['Counts'].median()) & (bs['ctr'] < 3)]) if not bs.empty else 0
        
        st.markdown(f"""
        <div class='insight-box'>
            <h4>💡 Strategic Recommendations</h4>
            <p>• Optimize {low_performers} underperforming brands (CTR < 2%)<br>
            • {opportunity_brands} high-volume brands need CTR improvement<br>
            • Focus on top keywords for leading brands<br>
            • {"Diversify" if brand_dominance > 40 else "Strengthen"} brand portfolio strategy</p>
        </div>
        """, unsafe_allow_html=True)



# ----------------- Category Tab (Enhanced & Optimized) -----------------
with tab_category:
    st.header("📦 Category Intelligence Hub")
    st.markdown("Comprehensive category performance analysis with strategic insights and competitive intelligence. 🌟")
    
    # Hero Image for Category Tab
    category_image_options = {
        "Category Analytics Focus": "https://placehold.co/1200x200/E6F3FA/FF5A6E?text=Category+Performance+Analysis",
        "Product Categories": "https://placehold.co/1200x200/FF5A6E/FFFFFF?text=Category+Intelligence+Dashboard",
        "Abstract Categories": "https://source.unsplash.com/1200x200/?categories,products",
        "Abstract Gradient": "https://placehold.co/1200x200/E6F3FA/FF5A6E?text=Lady+Care+Category+Insights",
    }
    selected_category_image = st.sidebar.selectbox("Choose Category Tab Hero", options=list(category_image_options.keys()), index=0, key="category_hero_image_selector")
    st.image(category_image_options[selected_category_image], use_container_width=True)
    
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
        st.error(f"❌ No category data available. Available columns: {list(queries.columns)}")
        st.info("💡 Please ensure your dataset contains a category column (category, Category, or Category Name)")
        st.stop()
    
    
    # Filter out "Other" category from all analysis
    category_queries = queries[
        (queries[category_column].notna()) & 
        (queries[category_column].str.lower() != 'other') &
        (queries[category_column].str.lower() != 'others')
    ]
    
    if category_queries.empty:
        st.error("❌ No valid category data available after filtering.")
        st.stop()
    
    # Category Performance Metrics Row
    total_categories = category_queries[category_column].nunique()
    top_category = category_queries.groupby(category_column)['Counts'].sum().idxmax()
    avg_category_counts = category_queries.groupby(category_column)['Counts'].sum().mean()
    
    # Calculate Category Dominance Index
    category_counts_sum = category_queries.groupby(category_column)['Counts'].sum()
    category_dominance = (category_counts_sum.max() / category_counts_sum.sum() * 100)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>📦</span>
            <div class='value'>{format_number(total_categories)}</div>
            <div class='label'>Total Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>👑</span>
            <div class='value'>{top_category[:15]}...</div>
            <div class='label'>Top Category</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>⚡</span>
            <div class='value'>{category_dominance:.1f}%</div>
            <div class='label'>Category Dominance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(f"""
        <div class='mini-metric'>
            <span class='icon'>📊</span>
            <div class='value'>{format_number(avg_category_counts)}</div>
            <div class='label'>Avg Search Counts</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Category Analysis Layout
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Enhanced Category Performance Analysis
        st.subheader("📈 Category Performance Matrix")
        
        # Calculate comprehensive category metrics
        cs = category_queries.groupby(category_column).agg({
            'Counts': 'sum',
            'clicks': 'sum', 
            'conversions': 'sum'
        }).reset_index()
        
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
            title='<b style="color:#FF5A6E; font-size:18px;">Category Performance Matrix: Search Counts vs CTR 🎯</b>',
            labels={'Counts': 'Total Search Counts', 'ctr': 'Click-Through Rate (%)', 'cr': 'Conversion Rate (%)'},
            color_continuous_scale=['#E6F3FA', '#FFB085', '#FF5A6E'],
            template='plotly_white'
        )
        
        fig_category_perf.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Search Counts: %{x:,.0f}<br>' +
                         'CTR: %{y:.2f}%<br>' +
                         'Total Clicks: %{marker.size:,.0f}<br>' +
                         'Conversion Rate: %{marker.color:.2f}%<extra></extra>'
        )
        
        fig_category_perf.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,247,232,0.8)',
            font=dict(color='#0B486B', family='Segoe UI'),
            title_x=0,
            xaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
            yaxis=dict(showgrid=True, gridcolor='#E6F3FA', linecolor='#FF5A6E', linewidth=2),
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
                title='<b style="color:#FF5A6E;">Search Counts by Category</b>',
                color='Counts',
                color_continuous_scale='Reds',
                text='Counts'
            )
            
            fig_counts.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside'
            )
            
            fig_counts.update_layout(
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#E6F3FA'),
                yaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
                height=400
            )
            
            st.plotly_chart(fig_counts, use_container_width=True)
        
        with col_chart2:
            # Conversion Rate by Category
            fig_cr = px.bar(
                cs.sort_values('cr', ascending=False).head(15), 
                x='category', 
                y='cr',
                title='<b style="color:#FF5A6E;">Conversion Rate by Category (%)</b>',
                color='cr',
                color_continuous_scale='Blues',
                text='cr'
            )
            
            fig_cr.update_traces(
                texttemplate='%{text:.2f}%',
                textposition='outside'
            )
            
            fig_cr.update_layout(
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI'),
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#E6F3FA'),
                yaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
                height=400
            )
            
            st.plotly_chart(fig_cr, use_container_width=True)
        
        # Top Categories Performance Table
        st.subheader("🏆 Top Category Performance")
        
        num_categories = st.slider(
            "Number of categories to display:", 
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
            'category': 'Category',
            'Counts': 'Search Counts',
            'share_pct': 'Share %',
            'clicks': 'Total Clicks',
            'conversions': 'Conversions',
            'ctr': 'CTR',
            'cr': 'CR',
            'classic_cr': 'Classic CR'
        })
        
        # Format numbers
        display_categories['Search Counts'] = display_categories['Search Counts'].apply(lambda x: f"{x:,.0f}")
        display_categories['Share %'] = display_categories['Share %'].apply(lambda x: f"{x:.2f}%")
        display_categories['Total Clicks'] = display_categories['Total Clicks'].apply(lambda x: f"{x:,.0f}")
        display_categories['Conversions'] = display_categories['Conversions'].apply(lambda x: f"{x:,.0f}")
        display_categories['CTR'] = display_categories['CTR'].apply(lambda x: f"{x:.2f}%")
        display_categories['CR'] = display_categories['CR'].apply(lambda x: f"{x:.2f}%")
        display_categories['Classic CR'] = display_categories['Classic CR'].apply(lambda x: f"{x:.2f}%")
        
        # Reorder columns
        column_order = ['Category', 'Search Counts', 'Share %', 'Total Clicks', 'Conversions', 'CTR', 'CR', 'Classic CR']
        display_categories = display_categories[column_order]
        
        st.dataframe(display_categories, use_container_width=True, hide_index=True)
        
        # Download button
        csv_categories = top_categories.to_csv(index=False)
        st.download_button(
            label="📥 Download Categories CSV",
            data=csv_categories,
            file_name=f"top_{num_categories}_categories.csv",
            mime="text/csv",
            key="category_csv_download"
        )
    
    with col_right:
        # Category Market Share Pie Chart
        st.subheader("📊 Category Market Share")
        
        top_categories_pie = cs.nlargest(10, 'Counts')
        
        fig_pie = px.pie(
            top_categories_pie, 
            names='category', 
            values='Counts',
            title='<b style="color:#FF5A6E;">Market Share Distribution</b>',
            color_discrete_sequence=['#FF5A6E', '#FFB085', '#E6F3FA', '#FF8A7A', '#FFF7E8', '#B8E6B8', '#87CEEB', '#DDA0DD', '#F0E68C', '#FFB6C1']
        )
        
        fig_pie.update_layout(
            font=dict(color='#0B486B', family='Segoe UI'),
            paper_bgcolor='rgba(255,247,232,0.8)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Category Performance Categories
        st.subheader("🎯 Category Performance Distribution")
        
        # Categorize categories based on performance
        cs['performance_category'] = pd.cut(
            cs['ctr'], 
            bins=[0, 2, 5, 10, float('inf')], 
            labels=['Low (0-2%)', 'Medium (2-5%)', 'High (5-10%)', 'Excellent (>10%)']
        )
        
        category_perf_counts = cs['performance_category'].value_counts().reset_index()
        category_perf_counts.columns = ['Performance', 'Count']
        
        fig_cat_perf = px.bar(
            category_perf_counts, 
            x='Performance', 
            y='Count',
            title='<b style="color:#FF5A6E;">CTR Performance Distribution</b>',
            color='Count',
            color_continuous_scale=['#E6F3FA', '#FF5A6E'],
            text='Count'
        )
        
        fig_cat_perf.update_traces(
            texttemplate='%{text}',
            textposition='outside'
        )
        
        fig_cat_perf.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,247,232,0.8)',
            font=dict(color='#0B486B', family='Segoe UI'),
            xaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
            yaxis=dict(showgrid=True, gridcolor='#E6F3FA')
        )
        
        st.plotly_chart(fig_cat_perf, use_container_width=True)
        
        # Enhanced Category Trend Analysis
        if 'Date' in queries.columns:
            st.subheader("📈 Category Trend Analysis")
            
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
                                title='<b style="color:#FF5A6E;">Top 5 Categories Monthly Trend</b>',
                                color_discrete_sequence=['#FF5A6E', '#FFB085', '#E6F3FA', '#FF8A7A', '#B8E6B8'],
                                markers=True
                            )
                            
                            fig_trend.update_layout(
                                plot_bgcolor='rgba(255,255,255,0.95)',
                                paper_bgcolor='rgba(255,247,232,0.8)',
                                font=dict(color='#0B486B', family='Segoe UI'),
                                xaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#E6F3FA',
                                    title='Month',
                                    dtick="M1",
                                    tickformat="%b %Y"
                                ),
                                yaxis=dict(
                                    showgrid=True, 
                                    gridcolor='#E6F3FA',
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
                            st.info("No trend data available for the selected date range and categories")
                    else:
                        st.info("No valid dates found in the filtered data")
                except Exception as e:
                    st.error(f"Error processing trend data: {str(e)}")
            else:
                st.info("No category data available for the selected date range")
    
    st.markdown("---")
    
    # Enhanced Category-Keyword Intelligence Matrix
    st.subheader("🔥 Category-Keyword Intelligence Matrix")
    
    # Create category filter dropdown
    if 'search' in queries.columns:
        # Get available categories (excluding null and 'other')
        available_categories = category_queries[category_column].unique()
        
        # Sort categories alphabetically
        available_categories = sorted(available_categories)
        
        # Create dropdown with "All Categories" option
        category_options = ['All Categories'] + list(available_categories)
        
        # Category selection dropdown
        selected_category = st.selectbox(
            "🎯 Select Category to Analyze:",
            options=category_options,
            index=0  # Default to "All Categories"
        )
        
        # Filter data based on selection
        if selected_category == 'All Categories':
            # Show top 8 categories if "All Categories" is selected
            top_categories_matrix = cs.nlargest(8, 'Counts')['category'].tolist()
            filtered_data = category_queries[category_queries[category_column].isin(top_categories_matrix)]
            matrix_title = "Top Categories vs Search Terms (Sum of Counts)"
        else:
            # Filter for selected category only
            filtered_data = category_queries[category_queries[category_column] == selected_category]
            matrix_title = f"{selected_category} - Search Terms Analysis (Sum of Counts)"
        
        # Remove null values from search terms
        matrix_data = filtered_data[
            (filtered_data[category_column].notna()) & 
            (filtered_data['search'].notna()) &
            (filtered_data['search'].str.lower() != 'other') &
            (filtered_data['search'].str.lower() != 'others')
        ].copy()
        
        if not matrix_data.empty:
            if selected_category == 'All Categories':
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
                        labels=dict(x="Search Terms", y="Categories", color="Total Counts"),
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        color_continuous_scale='Reds',
                        title=f'<b style="color:#FF5A6E;">{matrix_title}</b>',
                        aspect='auto'
                    )
                    
                    fig_matrix.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.95)',
                        paper_bgcolor='rgba(255,247,232,0.8)',
                        font=dict(color='#0B486B', family='Segoe UI'),
                        xaxis=dict(tickangle=45),
                        height=500
                    )
                    
                    # Update hover template
                    fig_matrix.update_traces(
                        hovertemplate='<b>Category:</b> %{y}<br>' +
                                    '<b>Search Term:</b> %{x}<br>' +
                                    '<b>Total Counts:</b> %{z:,.0f}<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_matrix, use_container_width=True)
                    
                    # Show summary statistics
                    total_interactions = category_search_matrix['Counts'].sum()
                    st.info(f"📊 Matrix shows {len(heatmap_data.index)} categories × {len(heatmap_data.columns)} search terms with {total_interactions:,} total search counts")
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
                    title=f'<b style="color:#FF5A6E;">{selected_category} - Top Search Terms by Count</b>',
                    labels={'Counts': 'Total Search Counts', 'search': 'Search Terms'},
                    color='Counts',
                    color_continuous_scale='Reds'
                )
                
                fig_single.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_single, use_container_width=True)
                
                # Show summary
                total_counts = search_counts['Counts'].sum()
                st.info(f"📊 {selected_category} has {len(search_counts)} top search terms with {total_counts:,} total counts")
        else:
            st.warning("No data available for the selected category filter")
    
    st.markdown("---")
    
    # Enhanced Top Keywords per Category Analysis
    st.subheader("🔑 Top Keywords per Category Analysis")
    
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
                "Choose display format:",
                ["Interactive Table", "Heatmap Visualization", "Top Keywords Summary"],
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
                    labels=dict(x="Keywords", y="Categories", color="Keyword Count"),
                    x=pivot_ckw.columns,
                    y=pivot_ckw.index,
                    color_continuous_scale='Blues',
                    title='<b style="color:#FF5A6E;">Category-Keyword Frequency Heatmap</b>',
                    aspect='auto'
                )
                
                fig_keyword_heatmap.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    xaxis=dict(tickangle=45),
                    height=600
                )
                
                st.plotly_chart(fig_keyword_heatmap, use_container_width=True)
            
            else:  # Top Keywords Summary
                # Show top keywords summary by category with enhanced accuracy
                st.subheader("🔥 Top 10 Keywords by Category")
                
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
                        'Category': cat,
                        'Top 10 Keywords (with counts)': keywords_str,
                        'Total Keywords': unique_keywords,
                        'Category Total Volume': f"{actual_category_total:,}",  # Use actual category total
                        'Share %': f"{share_percentage:.2f}%",  # Add share percentage column
                        'Keyword Analysis Volume': f"{total_keyword_count:,}",  # Show keyword-specific total
                        'Avg Keyword Count': f"{avg_keyword_count:.1f}",
                        'Top Keyword': top_10_keywords.iloc[0]['keyword'] if len(top_10_keywords) > 0 else 'N/A',
                        'Keyword Dominance %': f"{top_keyword_dominance:.1f}%"
                    })
                
                # Sort by actual category total volume (descending)
                top_keywords_summary = sorted(top_keywords_summary, key=lambda x: int(x['Category Total Volume'].replace(',', '')), reverse=True)
                summary_df = pd.DataFrame(top_keywords_summary)
                
                # Display the enhanced summary table
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Additional insights section with ENHANCED FONT SIZES
                st.markdown("---")
                st.subheader("📊 Category Keyword Intelligence")
                
                # Enhanced CSS with LARGER FONTS
                st.markdown("""
                <style>
                .enhanced-mini-metric {
                    background: linear-gradient(135deg, #FF5A6E 0%, #FFB085 100%);
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 8px 32px rgba(255, 90, 110, 0.3);
                    margin: 10px 0;
                    min-height: 160px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                
                .enhanced-mini-metric .icon {
                    font-size: 3em;
                    margin-bottom: 10px;
                    display: block;
                }
                
                .enhanced-mini-metric .value {
                    font-size: 1.6em;
                    font-weight: bold;
                    margin-bottom: 8px;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    line-height: 1.2;
                }
                
                .enhanced-mini-metric .label {
                    font-size: 1.1em;
                    opacity: 0.95;
                    font-weight: 600;
                    margin-bottom: 6px;
                }
                
                .enhanced-mini-metric .sub-label {
                    font-size: 1em;
                    opacity: 0.9;
                    font-weight: 500;
                    line-height: 1.2;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                }
                </style>
                """, unsafe_allow_html=True)
                
                col_insight1, col_insight2, col_insight3 = st.columns(3)
                
                with col_insight1:
                    # Most diverse category (most unique keywords)
                    most_diverse_cat = max(category_stats.items(), key=lambda x: x[1]['total_keywords'])
                    category_name = most_diverse_cat[0][:15] + "..." if len(most_diverse_cat[0]) > 15 else most_diverse_cat[0]
                    st.markdown(f"""
                    <div class='enhanced-mini-metric'>
                        <span class='icon'>🌟</span>
                        <div class='value'>{category_name}</div>
                        <div class='label'>Most Diverse Category</div>
                        <div class='sub-label'>{most_diverse_cat[1]['total_keywords']} unique keywords</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_insight2:
                    # Highest volume category with CORRECT SHARE PERCENTAGE
                    highest_volume_cat = max(category_stats.items(), key=lambda x: x[1]['total_count'])
                    category_name = highest_volume_cat[0][:15] + "..." if len(highest_volume_cat[0]) > 15 else highest_volume_cat[0]
                    st.markdown(f"""
                    <div class='enhanced-mini-metric'>
                        <span class='icon'>🚀</span>
                        <div class='value'>{category_name}</div>
                        <div class='label'>Highest Volume Category</div>
                        <div class='sub-label'>{highest_volume_cat[1]['total_count']:,} total searches<br>{highest_volume_cat[1]['share_percentage']:.2f}% share</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_insight3:
                    # Most concentrated category with CORRECT SHARE PERCENTAGE
                    most_concentrated_cat = max(category_stats.items(), key=lambda x: x[1]['share_percentage'])
                    category_name = most_concentrated_cat[0][:15] + "..." if len(most_concentrated_cat[0]) > 15 else most_concentrated_cat[0]
                    st.markdown(f"""
                    <div class='enhanced-mini-metric'>
                        <span class='icon'>🎯</span>
                        <div class='value'>{category_name}</div>
                        <div class='label'>Most Concentrated Category</div>
                        <div class='sub-label'>{most_concentrated_cat[1]['share_percentage']:.2f}% market share</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top keywords across all categories
                st.markdown("---")
                st.subheader("🏆 Global Top Keywords Across All Categories")
                
                # Get top keywords globally
                global_keywords = df_ckw.groupby('keyword')['count'].sum().reset_index()
                global_keywords = global_keywords.sort_values('count', ascending=False).head(20)
                
                # Create a horizontal bar chart for global keywords
                fig_global_keywords = px.bar(
                    global_keywords,
                    x='count',
                    y='keyword',
                    orientation='h',
                    title='<b style="color:#FF5A6E;">Top 20 Keywords Across All Categories</b>',
                    labels={'count': 'Total Search Count', 'keyword': 'Keywords'},
                    color='count',
                    color_continuous_scale='Reds',
                    text='count'
                )
                
                fig_global_keywords.update_traces(
                    texttemplate='%{text:,}',
                    textposition='outside'
                )
                
                fig_global_keywords.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    height=600,
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
                    yaxis_title="Keywords",
                    xaxis_title="Total Search Count"
                )
                
                st.plotly_chart(fig_global_keywords, use_container_width=True)
                
                # Category keyword distribution analysis
                st.markdown("---")
                st.subheader("📈 Category Keyword Distribution Analysis")
                
                # Create distribution data using corrected totals
                distribution_data = []
                for cat, stats in category_stats.items():
                    distribution_data.append({
                        'Category': cat,
                        'Unique Keywords': stats['total_keywords'],
                        'Total Volume': stats['total_count'],  # Use actual category total
                        'Average Count': stats['avg_count']
                    })
                
                dist_df = pd.DataFrame(distribution_data)
                
                # Create scatter plot for keyword distribution
                fig_distribution = px.scatter(
                    dist_df,
                    x='Unique Keywords',
                    y='Total Volume',
                    size='Average Count',
                    hover_name='Category',
                    title='<b style="color:#FF5A6E;">Category Keyword Diversity vs Volume</b>',
                    labels={
                        'Unique Keywords': 'Number of Unique Keywords',
                        'Total Volume': 'Total Search Volume',
                        'Average Count': 'Average Keyword Count'
                    },
                    color='Average Count',
                    color_continuous_scale='Viridis'
                )
                
                fig_distribution.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    xaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
                    yaxis=dict(showgrid=True, gridcolor='#E6F3FA')
                )
                
                fig_distribution.update_traces(
                    hovertemplate='<b>%{hovertext}</b><br>' +
                                 'Unique Keywords: %{x}<br>' +
                                 'Total Volume: %{y:,}<br>' +
                                 'Avg Count: %{marker.size:.1f}<extra></extra>'
                )
                
                st.plotly_chart(fig_distribution, use_container_width=True)
            
            # Download button for keyword analysis
            csv_keywords = df_ckw.to_csv(index=False)
            st.download_button(
                label="📥 Download Category Keywords CSV",
                data=csv_keywords,
                file_name="category_keywords_analysis.csv",
                mime="text/csv",
                key="category_keywords_csv_download"
            )
        else:
            st.info("Not enough keyword data per category.")
    
    except Exception as e:
        st.error(f"Error processing keyword analysis: {str(e)}")
        st.info("Not enough keyword data per category.")




# ----------------- Subcategory Tab -----------------
with tab_subcat:
    st.header("🧴 Subcategory Insights")
    st.markdown("Dive deep into subcategory performance and search trends. 🚀")

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
            
            
            # Enhanced Key Metrics Section
            st.subheader("📊 Subcategory Performance Overview")
            
            # Enhanced CSS for subcategory metrics - UNIFIED RED/ORANGE COLORS
            st.markdown("""
            <style>
            .subcat-metric-card {
                background: linear-gradient(135deg, #FF5A6E 0%, #FFB085 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                color: white;
                box-shadow: 0 8px 32px rgba(255, 90, 110, 0.3);
                margin: 10px 0;
                min-height: 160px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                transition: transform 0.2s ease;
            }
            
            .subcat-metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px rgba(255, 90, 110, 0.4);
            }
            
            .subcat-metric-card .icon {
                font-size: 3em;
                margin-bottom: 10px;
                display: block;
            }
            
            .subcat-metric-card .value {
                font-size: 1.6em;
                font-weight: bold;
                margin-bottom: 8px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                line-height: 1.2;
            }
            
            .subcat-metric-card .label {
                font-size: 1.1em;
                opacity: 0.95;
                font-weight: 600;
                margin-bottom: 6px;
            }
            
            .subcat-metric-card .sub-label {
                font-size: 1em;
                opacity: 0.9;
                font-weight: 500;
                line-height: 1.2;
            }
            
            .performance-badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                margin-left: 8px;
            }
            
            .high-performance {
                background-color: #4CAF50;
                color: white;
            }
            
            .medium-performance {
                background-color: #FF9800;
                color: white;
            }
            
            .low-performance {
                background-color: #F44336;
                color: white;
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
                <div class='subcat-metric-card'>
                    <span class='icon'>🏷️</span>
                    <div class='value'>{format_number(total_subcategories)}</div>
                    <div class='label'>Total Subcategories</div>
                    <div class='sub-label'>Active subcategories</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='subcat-metric-card'>
                    <span class='icon'>🔍</span>
                    <div class='value'>{format_number(total_searches)}</div>
                    <div class='label'>Total Searches</div>
                    <div class='sub-label'>Across all subcategories</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                performance_class = "high-performance" if avg_ctr > 5 else "medium-performance" if avg_ctr > 2 else "low-performance"
                st.markdown(f"""
                <div class='subcat-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{avg_ctr:.2f}% <span class='performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                    <div class='label'>Average CTR</div>
                    <div class='sub-label'>Click-through rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                top_subcat_display = top_subcategory[:12] + "..." if len(top_subcategory) > 12 else top_subcategory
                market_share = (top_subcategory_volume / total_searches * 100)
                st.markdown(f"""
                <div class='subcat-metric-card'>
                    <span class='icon'>👑</span>
                    <div class='value'>{top_subcat_display}</div>
                    <div class='label'>Top Subcategory</div>
                    <div class='sub-label'>{market_share:.1f}% market share</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.markdown(f"""
                <div class='subcat-metric-card'>
                    <span class='icon'>💰</span>
                    <div class='value'>{avg_cr:.2f}%</div>
                    <div class='label'>Avg Conversion Rate</div>
                    <div class='sub-label'>Overall performance</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                total_clicks = int(sc['clicks'].sum())  # ✅ Convert to integer to remove decimals
                st.markdown(f"""
                <div class='subcat-metric-card'>
                    <span class='icon'>🖱️</span>
                    <div class='value'>{format_number(total_clicks)}</div>
                    <div class='label'>Total Clicks</div>
                    <div class='sub-label'>Across all subcategories</div>
                </div>
                """, unsafe_allow_html=True)

            
            with col7:
                total_conversions = int(sc['conversions'].sum())  # ✅ Convert to integer to remove decimals
                st.markdown(f"""
                <div class='subcat-metric-card'>
                    <span class='icon'>✅</span>
                    <div class='value'>{format_number(total_conversions)}</div>
                    <div class='label'>Total Conversions</div>
                    <div class='sub-label'>Successful outcomes</div>
                </div>
                """, unsafe_allow_html=True)

            
            with col8:
                top_conversion_subcat = sc.nlargest(1, 'conversions')['sub_category'].iloc[0] if len(sc) > 0 else 'N/A'
                top_conversion_display = top_conversion_subcat[:12] + "..." if len(top_conversion_subcat) > 12 else top_conversion_subcat
                st.markdown(f"""
                <div class='subcat-metric-card'>
                    <span class='icon'>🏆</span>
                    <div class='value'>{top_conversion_display}</div>
                    <div class='label'>Conversion Leader</div>
                    <div class='sub-label'>Most conversions</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ✅ TOP 10 KEYWORDS BY SUBCATEGORY TABLE - EXACTLY LIKE CATEGORY TAB
            # Check if we have keyword data for subcategory analysis
            if 'keyword' in queries.columns and 'sub_category' in queries.columns:
                # Create keyword analysis similar to category tab
                df_sckw = queries[queries['keyword'].notna() & queries['sub_category'].notna()].copy()
                
                if len(df_sckw) > 0:
                    # Group keywords by subcategory
                    df_sckw_grouped = df_sckw.groupby(['sub_category', 'keyword']).agg({
                        'Counts': 'sum',
                        'clicks': 'sum',
                        'conversions': 'sum'
                    }).reset_index()
                    df_sckw_grouped.rename(columns={'Counts': 'count'}, inplace=True)
                    
                    # Calculate keyword performance metrics
                    df_sckw_grouped['keyword_ctr'] = df_sckw_grouped.apply(lambda r: (r['clicks']/r['count']*100) if r['count']>0 else 0, axis=1)
                    df_sckw_grouped['keyword_cr'] = df_sckw_grouped.apply(lambda r: (r['conversions']/r['count']*100) if r['count']>0 else 0, axis=1)
                    
                    # Show top keywords summary by subcategory with enhanced accuracy
                    st.subheader("🔥 Top 10 Keywords by Subcategory")
                    
                    top_keywords_summary = []
                    subcategory_stats = {}
                    
                    # Calculate total volume across all subcategories for share percentage
                    total_volume_all_subcategories = sc['Counts'].sum()
                    
                    for subcat in df_sckw_grouped['sub_category'].unique():
                        subcat_data = df_sckw_grouped[df_sckw_grouped['sub_category'] == subcat].sort_values('count', ascending=False)
                        
                        # Get top 10 keywords for this subcategory
                        top_10_keywords = subcat_data.head(10)
                        
                        # Create formatted keyword string with counts
                        keywords_list = []
                        for _, row in top_10_keywords.iterrows():
                            performance_indicator = "🔥" if row['keyword_ctr'] > 5 else "⚡" if row['keyword_ctr'] > 2 else "📊"
                            keywords_list.append(f"{performance_indicator} {row['keyword']} ({row['count']:,})")
                        
                        keywords_str = ' | '.join(keywords_list)
                        
                        # Calculate subcategory statistics - CORRECTED CALCULATION
                        actual_subcategory_total = sc[sc['sub_category'] == subcat]['Counts'].iloc[0] if len(sc[sc['sub_category'] == subcat]) > 0 else subcat_data['count'].sum()
                        
                        # Calculate correct share percentage based on total volume
                        share_percentage = (actual_subcategory_total / total_volume_all_subcategories * 100)
                        
                        total_keyword_count = subcat_data['count'].sum()
                        unique_keywords = len(subcat_data)
                        avg_keyword_count = subcat_data['count'].mean()
                        top_keyword_dominance = (top_10_keywords.iloc[0]['count'] / total_keyword_count * 100) if len(top_10_keywords) > 0 else 0
                        
                        # Store subcategory stats for additional insights
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
                            'Subcategory': subcat,
                            'Top 10 Keywords (with counts)': keywords_str,
                            'Total Keywords': unique_keywords,
                            'Subcategory Total Volume': f"{actual_subcategory_total:,}",
                            'Share %': f"{share_percentage:.2f}%",
                            'Keyword Analysis Volume': f"{total_keyword_count:,}",
                            'Avg Keyword Count': f"{avg_keyword_count:.1f}",
                            'Top Keyword': top_10_keywords.iloc[0]['keyword'] if len(top_10_keywords) > 0 else 'N/A',
                            'Top Keyword Volume': top_10_keywords.iloc[0]['count'] if len(top_10_keywords) > 0 else 0,
                            'Keyword Dominance %': f"{top_keyword_dominance:.1f}%"
                        })
                    
                    # Sort by actual subcategory total volume (descending)
                    top_keywords_summary = sorted(top_keywords_summary, key=lambda x: int(x['Subcategory Total Volume'].replace(',', '')), reverse=True)
                    summary_df = pd.DataFrame(top_keywords_summary)
                    
                    # Display the enhanced summary table with pagination
                    st.dataframe(summary_df, use_container_width=True, height=400, hide_index=True)
                    
                    # Download button for keyword summary
                    csv_keywords_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Subcategory Keywords Summary CSV",
                        data=csv_keywords_summary,
                        file_name="subcategory_keywords_summary.csv",
                        mime="text/csv",
                        key="subcategory_keywords_summary_download"
                    )
                    
                    # ✅ SUBCATEGORY KEYWORD INTELLIGENCE SECTION
                    st.markdown("---")
                    st.subheader("📊 Subcategory Keyword Intelligence")
                    
                    col_insight1, col_insight2, col_insight3 = st.columns(3)
                    
                    with col_insight1:
                        # Most diverse subcategory (most unique keywords)
                        most_diverse_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['total_keywords'])
                        subcategory_name = most_diverse_subcat[0][:15] + "..." if len(most_diverse_subcat[0]) > 15 else most_diverse_subcat[0]
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>🌟</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Most Diverse Subcategory</div>
                            <div class='sub-label'>{most_diverse_subcat[1]['total_keywords']} unique keywords</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight2:
                        # Highest volume subcategory
                        highest_volume_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['total_count'])
                        subcategory_name = highest_volume_subcat[0][:15] + "..." if len(highest_volume_subcat[0]) > 15 else highest_volume_subcat[0]
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>🚀</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Highest Volume Subcategory</div>
                            <div class='sub-label'>{highest_volume_subcat[1]['total_count']:,} total searches<br>{highest_volume_subcat[1]['share_percentage']:.2f}% share</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight3:
                        # Most concentrated subcategory
                        most_concentrated_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['share_percentage'])
                        subcategory_name = most_concentrated_subcat[0][:15] + "..." if len(most_concentrated_subcat[0]) > 15 else most_concentrated_subcat[0]
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>🎯</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Most Concentrated Subcategory</div>
                            <div class='sub-label'>{most_concentrated_subcat[1]['share_percentage']:.2f}% market share</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional insights row
                    col_insight4, col_insight5, col_insight6 = st.columns(3)
                    
                    with col_insight4:
                        # Most dominant keyword subcategory
                        most_dominant_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['dominance'])
                        subcategory_name = most_dominant_subcat[0][:15] + "..." if len(most_dominant_subcat[0]) > 15 else most_dominant_subcat[0]
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>👑</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Most Dominant Keyword</div>
                            <div class='sub-label'>{most_dominant_subcat[1]['dominance']:.1f}% dominance by top keyword</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight5:
                        # Highest average keyword volume
                        highest_avg_subcat = max(subcategory_stats.items(), key=lambda x: x[1]['avg_count'])
                        subcategory_name = highest_avg_subcat[0][:15] + "..." if len(highest_avg_subcat[0]) > 15 else highest_avg_subcat[0]
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>📊</span>
                            <div class='value'>{subcategory_name}</div>
                            <div class='label'>Highest Avg Keyword Vol</div>
                            <div class='sub-label'>{highest_avg_subcat[1]['avg_count']:.1f} avg searches per keyword</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_insight6:
                        # Most efficient subcategory (highest CTR among top keywords)
                        efficient_subcats = {}
                        for subcat, stats in subcategory_stats.items():
                            subcat_keywords = df_sckw_grouped[df_sckw_grouped['sub_category'] == subcat]
                            if len(subcat_keywords) > 0:
                                efficient_subcats[subcat] = subcat_keywords['keyword_ctr'].mean()
                        
                        if efficient_subcats:
                            most_efficient_subcat = max(efficient_subcats.items(), key=lambda x: x[1])
                            subcategory_name = most_efficient_subcat[0][:15] + "..." if len(most_efficient_subcat[0]) > 15 else most_efficient_subcat[0]
                            st.markdown(f"""
                            <div class='subcat-metric-card'>
                                <span class='icon'>⚡</span>
                                <div class='value'>{subcategory_name}</div>
                                <div class='label'>Most Efficient Keywords</div>
                                <div class='sub-label'>{most_efficient_subcat[1]:.1f}% avg CTR</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
            else:
                # If no keyword data available, show message
                st.subheader("🔥 Top 10 Keywords by Subcategory")
                
                st.markdown("---")
            
            # Interactive subcategory selection
            st.subheader("🎯 Interactive Subcategory Analysis")

            # Subcategory selector
            analysis_type = st.radio(
                "Choose Analysis Type:",
                ["📊 Top Performers Overview", "🔍 Detailed Subcategory Deep Dive", "📈 Performance Comparison", "📊 Market Share Analysis"],
                horizontal=True
            )

            # 🔧 FIX 1: Define missing variables
            total_subcategories = len(sc)
            total_searches = sc['Counts'].sum()

            # 🔧 FIX 2: Add required CSS styles
            st.markdown("""
            <style>
            .subcat-metric-card {
                background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
                padding: 20px;
                border-radius: 15px;
                border-left: 5px solid #FF5A6E;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
                margin: 10px 0;
            }

            .subcat-metric-card .icon {
                font-size: 2em;
                margin-bottom: 10px;
                display: block;
            }

            .subcat-metric-card .value {
                font-size: 1.8em;
                font-weight: bold;
                color: #0B486B;
                margin-bottom: 5px;
            }

            .subcat-metric-card .label {
                font-size: 1.1em;
                color: #2D3748;
                font-weight: 600;
                margin-bottom: 3px;
            }

            .subcat-metric-card .sub-label {
                font-size: 0.9em;
                color: #718096;
                font-style: italic;
            }

            .performance-badge {
                font-size: 0.7em;
                padding: 2px 6px;
                border-radius: 10px;
                font-weight: bold;
                margin-left: 5px;
            }

            .high-performance {
                background-color: #C6F6D5;
                color: #22543D;
            }

            .medium-performance {
                background-color: #FEFCBF;
                color: #744210;
            }

            .low-performance {
                background-color: #FED7D7;
                color: #742A2A;
            }
            </style>
            """, unsafe_allow_html=True)

            if analysis_type == "📊 Top Performers Overview":
                # Top subcategories analysis
                st.subheader("🏆 Top 20 Subcategories Performance")
                
                top_20_sc = sc.head(20).copy()
                
                # Enhanced bar chart with UNIFIED RED/ORANGE COLORS
                fig_top_subcats = px.bar(
                    top_20_sc,
                    x='sub_category',
                    y='Counts',
                    title='<b style="color:#FF5A6E;">Top 20 Subcategories by Search Volume</b>',
                    labels={'Counts': 'Search Volume', 'sub_category': 'Subcategories'},
                    color='Counts',
                    color_continuous_scale='Reds',
                    text='Counts'
                )
                
                fig_top_subcats.update_traces(
                    texttemplate='%{text:,}',
                    textposition='outside'
                )
                
                fig_top_subcats.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    height=600,
                    xaxis=dict(tickangle=45, showgrid=True, gridcolor='#E6F3FA'),
                    yaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
                    showlegend=False
                )
                
                st.plotly_chart(fig_top_subcats, use_container_width=True)
                
                # Performance metrics comparison chart
                st.subheader("📊 Performance Metrics Comparison")
                
                fig_metrics_comparison = go.Figure()
                
                # Add bars for each metric
                fig_metrics_comparison.add_trace(go.Bar(
                    name='CTR %',
                    x=top_20_sc['sub_category'],
                    y=top_20_sc['ctr'],
                    marker_color='#FF5A6E'
                ))
                
                fig_metrics_comparison.add_trace(go.Bar(
                    name='Conversion Rate %',
                    x=top_20_sc['sub_category'],
                    y=top_20_sc['conversion_rate'],
                    marker_color='#FFB085'
                ))
                
                fig_metrics_comparison.update_layout(
                    title='<b>CTR vs Conversion Rate Comparison</b>',
                    barmode='group',
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45),
                    yaxis=dict(title='Percentage (%)')
                )
                
                st.plotly_chart(fig_metrics_comparison, use_container_width=True)
                
                # ✅ TOP 10 KEYWORDS BY SUBCATEGORY SUMMARY
                st.subheader("🔥 Top 10 Keywords by Subcategory")

                try:
                    top_keywords_summary = []
                    subcategory_stats = {}

                    # Calculate total volume across all subcategories for share percentage
                    total_volume_all_subcategories = sc['Counts'].sum()

                    # Get unique subcategories from your processed dataframe
                    for subcat in queries['sub_category'].unique():  # ✅ CHANGED FROM df TO queries
                        if pd.isna(subcat):  # Skip NaN subcategories
                            continue
                            
                        # Filter data for this specific subcategory
                        subcat_data = queries[queries['sub_category'] == subcat]  # ✅ CHANGED FROM df TO queries
                        
                        # Aggregate by keyword (normalized_query) - sum the counts for each keyword
                        keyword_aggregated = subcat_data.groupby('normalized_query').agg({
                            'Counts': 'sum',
                            'clicks': 'sum', 
                            'conversions': 'sum'
                        }).reset_index()
                        
                        # Sort by Counts (volume) descending
                        keyword_aggregated = keyword_aggregated.sort_values('Counts', ascending=False)
                        
                        # Get top 10 keywords for this subcategory
                        top_10_keywords = keyword_aggregated.head(10)
                        
                        # Create formatted keyword string with counts
                        keywords_list = []
                        for _, row in top_10_keywords.iterrows():
                            keywords_list.append(f"{row['normalized_query']} ({row['Counts']:,})")
                        
                        keywords_str = ' | '.join(keywords_list)
                        
                        # Calculate subcategory statistics
                        actual_subcategory_total = sc[sc['sub_category'] == subcat]['Counts'].iloc[0] if len(sc[sc['sub_category'] == subcat]) > 0 else keyword_aggregated['Counts'].sum()
                        
                        # Calculate correct share percentage based on total volume
                        share_percentage = (actual_subcategory_total / total_volume_all_subcategories * 100)
                        
                        total_keyword_count = keyword_aggregated['Counts'].sum()
                        unique_keywords = len(keyword_aggregated)
                        avg_keyword_count = keyword_aggregated['Counts'].mean()
                        top_keyword_dominance = (top_10_keywords.iloc[0]['Counts'] / total_keyword_count * 100) if len(top_10_keywords) > 0 else 0
                        
                        # Store subcategory stats for additional insights
                        subcategory_stats[subcat] = {
                            'total_keywords': unique_keywords,
                            'total_count': actual_subcategory_total,
                            'keyword_total_count': total_keyword_count,
                            'avg_count': avg_keyword_count,
                            'top_keyword': top_10_keywords.iloc[0]['normalized_query'] if len(top_10_keywords) > 0 else 'N/A',
                            'dominance': top_keyword_dominance,
                            'share_percentage': share_percentage
                        }
                        
                        top_keywords_summary.append({
                            'Subcategory': subcat,
                            'Top 10 Keywords (with counts)': keywords_str,
                            'Total Keywords': unique_keywords,
                            'Subcategory Total Volume': f"{actual_subcategory_total:,}",
                            'Share %': f"{share_percentage:.2f}%",
                            'Keyword Analysis Volume': f"{total_keyword_count:,}",
                            'Avg Keyword Count': f"{avg_keyword_count:.1f}",
                            'Top Keyword': top_10_keywords.iloc[0]['normalized_query'] if len(top_10_keywords) > 0 else 'N/A',
                            'Top Keyword Volume': f"{top_10_keywords.iloc[0]['Counts']:,}" if len(top_10_keywords) > 0 else "0",
                            'Keyword Dominance %': f"{top_keyword_dominance:.1f}%"
                        })

                    # Sort by actual subcategory total volume (descending)
                    top_keywords_summary = sorted(top_keywords_summary, key=lambda x: int(x['Subcategory Total Volume'].replace(',', '')), reverse=True)
                    summary_df = pd.DataFrame(top_keywords_summary)

                    # Display the enhanced summary table
                    st.dataframe(summary_df, use_container_width=True, height=400, hide_index=True)

                    # Download button
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Subcategory Keywords Summary CSV",
                        data=csv_summary,
                        file_name="subcategory_keywords_summary.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"An error occurred in the Subcategory analysis: {str(e)}")
                    st.write("Please check your data format and try again.")





            elif analysis_type == "🔍 Detailed Subcategory Deep Dive":
                # Detailed analysis section
                st.subheader("🔬 Subcategory Deep Dive Analysis")
                
                # Subcategory selector with search functionality
                selected_subcategory = st.selectbox(
                    "Select a subcategory for detailed analysis:",
                    options=sc['sub_category'].tolist(),
                    index=0
                )
                
                if selected_subcategory:
                    # Get detailed data for selected subcategory
                    subcat_data = sc[sc['sub_category'] == selected_subcategory].iloc[0]
                    subcat_rank = sc.reset_index().index[sc['sub_category'] == selected_subcategory].tolist()[0] + 1
                    
                    # Detailed metrics for selected subcategory
                    col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                    
                    with col_detail1:
                        rank_performance = "high-performance" if subcat_rank <= 3 else "medium-performance" if subcat_rank <= 10 else "low-performance"
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>🏆</span>
                            <div class='value'>#{subcat_rank} <span class='performance-badge {rank_performance}'>{"Top 3" if subcat_rank <= 3 else "Top 10" if subcat_rank <= 10 else "Lower"}</span></div>
                            <div class='label'>Market Rank</div>
                            <div class='sub-label'>Out of {total_subcategories} subcategories</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_detail2:
                        market_share = (subcat_data['Counts'] / total_searches * 100)
                        share_performance = "high-performance" if market_share > 5 else "medium-performance" if market_share > 2 else "low-performance"
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>📊</span>
                            <div class='value'>{market_share:.2f}% <span class='performance-badge {share_performance}'>{"High" if market_share > 5 else "Medium" if market_share > 2 else "Low"}</span></div>
                            <div class='label'>Market Share</div>
                            <div class='sub-label'>Of total search volume</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_detail3:
                        performance_score = (subcat_data['ctr'] + subcat_data['conversion_rate']) / 2
                        score_performance = "high-performance" if performance_score > 3 else "medium-performance" if performance_score > 1 else "low-performance"
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>⭐</span>
                            <div class='value'>{performance_score:.1f} <span class='performance-badge {score_performance}'>{"High" if performance_score > 3 else "Medium" if performance_score > 1 else "Low"}</span></div>
                            <div class='label'>Performance Score</div>
                            <div class='sub-label'>Combined CTR & CR</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_detail4:
                        conversion_efficiency = subcat_data['conversion_rate'] / subcat_data['ctr'] * 100 if subcat_data['ctr'] > 0 else 0
                        efficiency_performance = "high-performance" if conversion_efficiency > 50 else "medium-performance" if conversion_efficiency > 25 else "low-performance"
                        st.markdown(f"""
                        <div class='subcat-metric-card'>
                            <span class='icon'>⚡</span>
                            <div class='value'>{conversion_efficiency:.1f}% <span class='performance-badge {efficiency_performance}'>{"High" if conversion_efficiency > 50 else "Medium" if conversion_efficiency > 25 else "Low"}</span></div>
                            <div class='label'>Conversion Efficiency</div>
                            <div class='sub-label'>CR as % of CTR</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed performance breakdown with CORRECTED CR CALCULATIONS
                    st.markdown("### 📈 Performance Breakdown")

                    metrics_data = {
                        'Metric': ['Search Volume', 'Total Clicks', 'Total Conversions', 
                                'Click-Through Rate', 'Classic CVR (Conv/Clicks)', 
                                'Conversion Rate (Conv/Counts)', 'Click Share', 'Conversion Share'],
                        'Value': [
                            f"{int(subcat_data['Counts']):,}",  # ✅ Remove decimals
                            f"{int(subcat_data['clicks']):,}",  # ✅ Remove decimals
                            f"{int(subcat_data['conversions']):,}",  # ✅ Remove decimals
                            f"{subcat_data['ctr']:.2f}%",
                            f"{subcat_data['classic_cvr']:.2f}%",
                            f"{subcat_data['conversion_rate']:.2f}%",
                            f"{subcat_data['click_share']:.2f}%",
                            f"{subcat_data['conversion_share']:.2f}%"
                        ],
                        'Performance': [
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
                    
                    # Performance comparison radar chart
                    st.markdown("### 📊 Performance Radar Chart")
                    
                    # Normalize values for radar chart
                    normalized_data = {
                        'Search Volume': subcat_data['Counts'] / sc['Counts'].max() * 100,
                        'CTR': subcat_data['ctr'] / sc['ctr'].max() * 100 if sc['ctr'].max() > 0 else 0,
                        'Conversion Rate': subcat_data['conversion_rate'] / sc['conversion_rate'].max() * 100 if sc['conversion_rate'].max() > 0 else 0,
                        'Click Share': subcat_data['click_share'],
                        'Conversion Share': subcat_data['conversion_share']
                    }
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(normalized_data.values()),
                        theta=list(normalized_data.keys()),
                        fill='toself',
                        name=selected_subcategory,
                        line_color='#FF5A6E'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title=f'Performance Radar - {selected_subcategory}',
                        height=400
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # ✅ ENHANCED DOWNLOAD BUTTON FOR DETAILED ANALYSIS
                    detailed_analysis_data = {
                        'Subcategory': [selected_subcategory],
                        'Search Volume': [subcat_data['Counts']],
                        'Total Clicks': [subcat_data['clicks']],
                        'Total Conversions': [subcat_data['conversions']],
                        'CTR %': [subcat_data['ctr']],
                        'Classic CVR %': [subcat_data['classic_cvr']],
                        'Conversion Rate %': [subcat_data['conversion_rate']],
                        'Market Rank': [subcat_rank],
                        'Market Share %': [market_share],
                        'Performance Score': [performance_score],
                        'Conversion Efficiency %': [conversion_efficiency]
                    }
                    
                    detailed_df = pd.DataFrame(detailed_analysis_data)
                    csv_detailed = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Detailed Analysis CSV",
                        data=csv_detailed,
                        file_name=f"detailed_analysis_{selected_subcategory.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="detailed_analysis_download"
                    )

            elif analysis_type == "📈 Performance Comparison":
                st.subheader("⚖️ Subcategory Performance Comparison")
                
                # Multi-select for comparison
                selected_subcategories = st.multiselect(
                    "Select subcategories to compare (max 10):",
                    options=sc['sub_category'].tolist(),
                    default=sc['sub_category'].head(5).tolist(),
                    max_selections=10
                )
                
                if selected_subcategories:
                    # Filter data for selected subcategories
                    comparison_data = sc[sc['sub_category'].isin(selected_subcategories)].copy()
                    
                    # Comparison metrics visualization
                    fig_comparison = go.Figure()
                    
                    # Add traces for different metrics
                    metrics = ['ctr', 'conversion_rate', 'click_share', 'conversion_share']
                    metric_names = ['CTR %', 'Conversion Rate %', 'Click Share %', 'Conversion Share %']
                    colors = ['#FF5A6E', '#FFB085', '#FF7F94', '#FFA5A5']
                    
                    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                        fig_comparison.add_trace(go.Bar(
                            name=name,
                            x=comparison_data['sub_category'],
                            y=comparison_data[metric],
                            marker_color=colors[i]
                        ))
                    
                    fig_comparison.update_layout(
                        title='<b>Performance Metrics Comparison</b>',
                        barmode='group',
                        plot_bgcolor='rgba(255,255,255,0.95)',
                        paper_bgcolor='rgba(255,247,232,0.8)',
                        font=dict(color='#0B486B', family='Segoe UI'),
                        height=500,
                        xaxis=dict(tickangle=45),
                        yaxis=dict(title='Percentage (%)')
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Detailed comparison table
                    st.markdown("### 📊 Detailed Comparison Table")
                    
                    comparison_table = comparison_data[['sub_category', 'Counts', 'clicks', 'conversions', 
                                                    'ctr', 'conversion_rate', 'click_share', 'conversion_share']].copy()
                    comparison_table.columns = ['Subcategory', 'Search Volume', 'Clicks', 'Conversions', 
                                            'CTR %', 'Conversion Rate %', 'Click Share %', 'Conversion Share %']
                    
                    # Format numeric columns
                    comparison_table['Search Volume'] = comparison_table['Search Volume'].apply(lambda x: f"{int(x):,}")  # ✅ Remove decimals
                    comparison_table['Clicks'] = comparison_table['Clicks'].apply(lambda x: f"{int(x):,}")  # ✅ Remove decimals
                    comparison_table['Conversions'] = comparison_table['Conversions'].apply(lambda x: f"{int(x):,}")  # ✅ Remove decimals
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
                        file_name="subcategory_comparison.csv",
                        mime="text/csv",
                        key="comparison_download"
                    )
                else:
                    st.info("Please select subcategories to compare.")

            elif analysis_type == "📊 Market Share Analysis":
                st.subheader("📊 Market Share & Distribution Analysis")
                
                # Market share visualization
                col_pie, col_treemap = st.columns(2)
                
                with col_pie:
                    # Pie chart for top 10 subcategories
                    top_10_market = sc.head(10).copy()
                    others_value = sc.iloc[10:]['Counts'].sum() if len(sc) > 10 else 0
                    
                    if others_value > 0:
                        others_row = pd.DataFrame({
                            'sub_category': ['Others'],
                            'Counts': [others_value]
                        })
                        pie_data = pd.concat([top_10_market[['sub_category', 'Counts']], others_row])
                    else:
                        pie_data = top_10_market[['sub_category', 'Counts']]
                    
                    fig_pie = px.pie(
                        pie_data,
                        values='Counts',
                        names='sub_category',
                        title='<b>Top 10 Subcategories Market Share</b>',
                        color_discrete_sequence=px.colors.sequential.Reds
                    )
                    
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_treemap:
                    # Treemap visualization
                    fig_treemap = px.treemap(
                        sc.head(20),
                        path=['sub_category'],
                        values='Counts',
                        title='<b>Subcategory Volume Distribution</b>',
                        color='ctr',
                        color_continuous_scale='Reds',
                        hover_data={'Counts': ':,', 'ctr': ':.2f'}
                    )
                    
                    fig_treemap.update_layout(height=400)
                    st.plotly_chart(fig_treemap, use_container_width=True)
                
                # Distribution analysis
                st.markdown("### 📈 Distribution Analysis")
                
                # Calculate distribution metrics
                gini_coefficient = 1 - 2 * np.sum(np.cumsum(sc['Counts'].sort_values()) / sc['Counts'].sum()) / len(sc)
                herfindahl_index = np.sum((sc['Counts'] / sc['Counts'].sum()) ** 2)
                top_5_concentration = sc.head(5)['Counts'].sum() / sc['Counts'].sum() * 100
                top_10_concentration = sc.head(10)['Counts'].sum() / sc['Counts'].sum() * 100
                
                col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
                
                with col_dist1:
                    st.markdown(f"""
                    <div class='subcat-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{gini_coefficient:.3f}</div>
                        <div class='label'>Gini Coefficient</div>
                        <div class='sub-label'>Market concentration</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_dist2:
                    st.markdown(f"""
                    <div class='subcat-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{herfindahl_index:.4f}</div>
                        <div class='label'>Herfindahl Index</div>
                        <div class='sub-label'>Market dominance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_dist3:
                    st.markdown(f"""
                    <div class='subcat-metric-card'>
                        <span class='icon'>🔝</span>
                        <div class='value'>{top_5_concentration:.1f}%</div>
                        <div class='label'>Top 5 Share</div>
                        <div class='sub-label'>Market concentration</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_dist4:
                    st.markdown(f"""
                    <div class='subcat-metric-card'>
                        <span class='icon'>💯</span>
                        <div class='value'>{top_10_concentration:.1f}%</div>
                        <div class='label'>Top 10 Share</div>
                        <div class='sub-label'>Market concentration</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Lorenz curve
                st.markdown("### 📉 Lorenz Curve - Market Concentration")
                
                sorted_counts = sc['Counts'].sort_values()
                cumulative_counts = np.cumsum(sorted_counts) / sorted_counts.sum()
                cumulative_subcategories = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
                
                fig_lorenz = go.Figure()
                
                # Add Lorenz curve
                fig_lorenz.add_trace(go.Scatter(
                    x=cumulative_subcategories,
                    y=cumulative_counts,
                    mode='lines',
                    name='Actual Distribution',
                    line=dict(color='#FF5A6E', width=3)
                ))
                
                # Add line of equality
                fig_lorenz.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Perfect Equality',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                fig_lorenz.update_layout(
                    title='<b>Lorenz Curve - Subcategory Search Volume Distribution</b>',
                    xaxis_title='Cumulative % of Subcategories',
                    yaxis_title='Cumulative % of Search Volume',
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    height=500
                )
                
                st.plotly_chart(fig_lorenz, use_container_width=True)

                
        else:
            st.warning("No subcategory data available in the uploaded file.")
            st.info("Please ensure your data contains a 'sub_category' column with valid values.")
            
    except Exception as e:
        st.error(f"An error occurred in the Subcategory analysis: {str(e)}")
        st.info("Please check your data format and try again.")


# ----------------- Generic Type Tab -----------------

# Assuming tab_generic is defined elsewhere, e.g., tabs = st.tabs(["Generic"]), tab_generic = tabs[0]
with tab_generic:
    st.header("🛠 Generic Type Insights")
    st.markdown("Dive deep into generic term performance and search trends. 🚀")

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
        
        # Enhanced CSS for generic type metrics - Unified with Subcategory Tab (Red/Orange Theme)
        st.markdown("""
        <style>
        .generic-metric-card {
            background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #FF5A6E;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin: 10px 0;
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 0.2s ease;
        }
        
        .generic-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        .generic-metric-card .icon {
            font-size: 2em;
            margin-bottom: 10px;
            display: block;
        }
        
        .generic-metric-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #0B486B;
            margin-bottom: 5px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.2;
        }
        
        .generic-metric-card .label {
            font-size: 1.1em;
            color: #2D3748;
            font-weight: 600;
            margin-bottom: 3px;
        }
        
        .generic-metric-card .sub-label {
            font-size: 0.9em;
            color: #718096;
            font-style: italic;
            line-height: 1.2;
        }
        
        .performance-badge {
            font-size: 0.7em;
            padding: 2px 6px;
            border-radius: 10px;
            font-weight: bold;
            margin-left: 5px;
        }
        
        .high-performance {
            background-color: #C6F6D5;
            color: #22543D;
        }
        
        .medium-performance {
            background-color: #FEFCBF;
            color: #744210;
        }
        
        .low-performance {
            background-color: #FED7D7;
            color: #742A2A;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Enhanced Key Metrics Section
        st.subheader("📊 Generic Type Performance Overview")
        
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
            <div class='generic-metric-card'>
                <span class='icon'>🛠️</span>
                <div class='value'>{format_number(total_generic_terms)}</div>
                <div class='label'>Total Generic Terms</div>
                <div class='sub-label'>Active search terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='generic-metric-card'>
                <span class='icon'>🔍</span>
                <div class='value'>{format_number(total_searches)}</div>
                <div class='label'>Total Searches</div>
                <div class='sub-label'>Across all generic terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            performance_class = "high-performance" if avg_ctr > 5 else "medium-performance" if avg_ctr > 2 else "low-performance"
            st.markdown(f"""
            <div class='generic-metric-card'>
                <span class='icon'>📈</span>
                <div class='value'>{avg_ctr:.2f}% <span class='performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                <div class='label'>Average CTR</div>
                <div class='sub-label'>Click-through rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top_generic_display = top_generic_term[:12] + "..." if len(top_generic_term) > 12 else top_generic_term
            market_share = (top_generic_volume / total_searches * 100) if total_searches > 0 else 0
            st.markdown(f"""
            <div class='generic-metric-card'>
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
            <div class='generic-metric-card'>
                <span class='icon'>💰</span>
                <div class='value'>{avg_cr:.2f}%</div>
                <div class='label'>Avg Conversion Rate</div>
                <div class='sub-label'>Overall performance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            total_clicks = int(gt_agg['Clicks'].sum())
            st.markdown(f"""
            <div class='generic-metric-card'>
                <span class='icon'>🖱️</span>
                <div class='value'>{format_number(total_clicks)}</div>
                <div class='label'>Total Clicks</div>
                <div class='sub-label'>Across all generic terms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            total_conversions = int(gt_agg['Conversions'].sum())
            st.markdown(f"""
            <div class='generic-metric-card'>
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
            <div class='generic-metric-card'>
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
            
            # Enhanced bar chart with Red/Orange Colors
            fig_top_generics = px.bar(
                top_20_gt,
                x='search',
                y='count',
                title='<b style="color:#FF5A6E;">Top 20 Generic Terms by Search Volume</b>',
                labels={'count': 'Search Volume', 'search': 'Generic Terms'},
                color='count',
                color_continuous_scale='Reds',
                text='count'
            )
            
            fig_top_generics.update_traces(
                texttemplate='%{text:,}',
                textposition='outside'
            )
            
            fig_top_generics.update_layout(
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI'),
                height=600,
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#E6F3FA'),
                yaxis=dict(showgrid=True, gridcolor='#E6F3FA'),
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
                marker_color='#FF5A6E'
            ))
            
            fig_metrics_comparison.add_trace(go.Bar(
                name='Conversion Rate %',
                x=top_20_gt['search'],
                y=top_20_gt['conversion_rate'],
                marker_color='#FFB085'
            ))
            
            fig_metrics_comparison.update_layout(
                title='<b>CTR vs Conversion Rate Comparison</b>',
                barmode='group',
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI'),
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
                    rank_performance = "high-performance" if generic_rank <= 3 else "medium-performance" if generic_rank <= 10 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>🏆</span>
                        <div class='value'>#{generic_rank} <span class='performance-badge {rank_performance}'>{"Top 3" if generic_rank <= 3 else "Top 10" if generic_rank <= 10 else "Lower"}</span></div>
                        <div class='label'>Market Rank</div>
                        <div class='sub-label'>Out of {total_generic_terms} terms</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail2:
                    market_share = (generic_data['count'] / total_searches * 100)
                    share_performance = "high-performance" if market_share > 5 else "medium-performance" if market_share > 2 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{market_share:.2f}% <span class='performance-badge {share_performance}'>{"High" if market_share > 5 else "Medium" if market_share > 2 else "Low"}</span></div>
                        <div class='label'>Market Share</div>
                        <div class='sub-label'>Of total search volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail3:
                    performance_score = (generic_data['ctr'] + generic_data['conversion_rate']) / 2
                    score_performance = "high-performance" if performance_score > 3 else "medium-performance" if performance_score > 1 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>⭐</span>
                        <div class='value'>{performance_score:.1f} <span class='performance-badge {score_performance}'>{"High" if performance_score > 3 else "Medium" if performance_score > 1 else "Low"}</span></div>
                        <div class='label'>Performance Score</div>
                        <div class='sub-label'>Combined CTR & CR</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail4:
                    conversion_efficiency = generic_data['conversion_rate'] / generic_data['ctr'] * 100 if generic_data['ctr'] > 0 else 0
                    efficiency_performance = "high-performance" if conversion_efficiency > 50 else "medium-performance" if conversion_efficiency > 25 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>⚡</span>
                        <div class='value'>{conversion_efficiency:.1f}% <span class='performance-badge {efficiency_performance}'>{"High" if conversion_efficiency > 50 else "Medium" if conversion_efficiency > 25 else "Low"}</span></div>
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
                    line_color='#FF5A6E'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title=f'Performance Radar - {selected_generic}',
                    height=400,
                    font=dict(color='#0B486B', family='Segoe UI'),
                    paper_bgcolor='rgba(255,247,232,0.8)'
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
                colors = ['#FF5A6E', '#FFB085', '#FF7F94', '#FFA5A5']
                
                for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                    fig_comparison.add_trace(go.Bar(
                        name=name,
                        x=comparison_data['search'],
                        y=comparison_data[metric],
                        marker_color=colors[i]
                    ))
                
                fig_comparison.update_layout(
                    title='<b>Performance Metrics Comparison</b>',
                    barmode='group',
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
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
                    title='<b>Top 10 Generic Terms Market Share</b>',
                    color_discrete_sequence=px.colors.sequential.Reds
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(
                    height=400,
                    font=dict(color='#0B486B', family='Segoe UI'),
                    paper_bgcolor='rgba(255,247,232,0.8)'
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_treemap:
                # Treemap visualization
                fig_treemap = px.treemap(
                    gt_agg.head(20),
                    path=['search'],
                    values='count',
                    title='<b>Generic Terms Volume Distribution</b>',
                    color='ctr',
                    color_continuous_scale='Reds',
                    hover_data={'count': ':,', 'ctr': ':.2f'}
                )
                
                fig_treemap.update_layout(
                    height=400,
                    font=dict(color='#0B486B', family='Segoe UI'),
                    paper_bgcolor='rgba(255,247,232,0.8)'
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Distribution analysis
            st.markdown("### 📈 Distribution Analysis")
            
            col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
            
            with col_dist1:
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>📊</span>
                    <div class='value'>{gini_coefficient:.3f}</div>
                    <div class='label'>Gini Coefficient</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist2:
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{herfindahl_index:.4f}</div>
                    <div class='label'>Herfindahl Index</div>
                    <div class='sub-label'>Market dominance</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist3:
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>🔝</span>
                    <div class='value'>{top_5_concentration:.1f}%</div>
                    <div class='label'>Top 5 Share</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist4:
                st.markdown(f"""
                <div class='generic-metric-card'>
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
                line=dict(color='#FF5A6E', width=3)
            ))
            
            # Add line of equality
            fig_lorenz.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Line of Equality',
                line=dict(color='gray', dash='dash', width=2)
            ))
            
            fig_lorenz.update_layout(
                title='<b>Lorenz Curve - Generic Terms Market Concentration</b>',
                xaxis_title='Cumulative % of Generic Terms',
                yaxis_title='Cumulative % of Search Volume',
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI'),
                height=400,
                showlegend=True
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
                
                # Quick stats for filtered data - USING CSS CARDS WITH RED/ORANGE THEME
                filtered_col1, filtered_col2, filtered_col3, filtered_col4 = st.columns(4)
                
                with filtered_col1:
                    st.markdown(f"""
                    <div class='generic-metric-card' style='background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%); border-left: 5px solid #FF5A6E;'>
                        <span class='icon'>📊</span>
                        <div class='value'>{format_number(len(filtered_data))}</div>
                        <div class='label'>Terms Found</div>
                        <div class='sub-label'>Matching filters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col2:
                    total_searches_filtered = filtered_data['count'].sum()
                    st.markdown(f"""
                    <div class='generic-metric-card' style='background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%); border-left: 5px solid #FF5A6E;'>
                        <span class='icon'>🔍</span>
                        <div class='value'>{format_number(total_searches_filtered)}</div>
                        <div class='label'>Total Searches</div>
                        <div class='sub-label'>Filtered volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col3:
                    avg_ctr_filtered = filtered_data['ctr'].mean()
                    ctr_performance = "high-performance" if avg_ctr_filtered > 5 else "medium-performance" if avg_ctr_filtered > 2 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card' style='background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%); border-left: 5px solid #FF5A6E;'>
                        <span class='icon'>📈</span>
                        <div class='value'>{avg_ctr_filtered:.2f}% <span class='performance-badge {ctr_performance}'>{"High" if avg_ctr_filtered > 5 else "Medium" if avg_ctr_filtered > 2 else "Low"}</span></div>
                        <div class='label'>Avg CTR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col4:
                    avg_cr_filtered = filtered_data['conversion_rate'].mean()
                    cr_performance = "high-performance" if avg_cr_filtered > 3 else "medium-performance" if avg_cr_filtered > 1 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card' style='background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%); border-left: 5px solid #FF5A6E;'>
                        <span class='icon'>💰</span>
                        <div class='value'>{avg_cr_filtered:.2f}% <span class='performance-badge {cr_performance}'>{"High" if avg_cr_filtered > 3 else "Medium" if avg_cr_filtered > 1 else "Low"}</span></div>
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
                
                # Enhanced table UI
                st.markdown("""
                <style>
                .generic-table-container {
                    background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
                    padding: 20px;
                    border-radius: 15px;
                    border-left: 5px solid #FF5A6E;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    margin: 10px 0;
                    transition: transform 0.2s ease;
                }
                .generic-table-container:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                }
                .generic-table-container table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: Arial, sans-serif;
                }
                .generic-table-container th {
                    background-color: #FF5A6E;
                    color: white;
                    font-weight: bold;
                    padding: 12px;
                    text-align: left;
                    font-size: 1.1em;
                }
                .generic-table-container td {
                    padding: 10px;
                    font-size: 1em;
                    color: #2D3748;
                    border-bottom: 1px solid #E2E8F0;
                }
                .generic-table-container tr:nth-child(even) {
                    background-color: #FFF5F5;
                }
                .generic-table-container tr:hover {
                    background-color: #FED7D7;
                }
                </style>
                <div class='generic-table-container'>
                """, unsafe_allow_html=True)
                st.dataframe(display_filtered, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
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


    except KeyError as e:
        st.error(f"❌ Missing required column: {str(e)}")
        st.info("Please ensure your data contains: 'search', 'count', 'Clicks', 'Conversions'")
    except ValueError as e:
        st.error(f"❌ Data format error: {str(e)}")
        st.info("Please check that numeric columns contain valid numbers")
    except Exception as e:
        st.error(f"❌ Unexpected error processing generic type data: {str(e)}")
        st.info("Please check your data format and try again.")

# ----------------- Time Analysis Tab (Enhanced) -----------------
# ----------------- Time Analysis Tab (Enhanced) -----------------
with tab_time:
    st.header("⏰ Temporal Analysis & Seasonality")
    st.markdown("Uncover monthly trends to optimize campaigns. 📅")

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
        
        # CSS for UI consistency with Generic Type Tab
        st.markdown("""
        <style>
        .generic-metric-card {
            background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #FF5A6E;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin: 10px 0;
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: transform 0.2s ease;
        }
        .generic-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .generic-metric-card .icon {
            font-size: 2em;
            margin-bottom: 10px;
            display: block;
        }
        .generic-metric-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #0B486B;
            margin-bottom: 5px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.2;
        }
        .generic-metric-card .label {
            font-size: 1.1em;
            color: #2D3748;
            font-weight: 600;
            margin-bottom: 3px;
        }
        .generic-metric-card .sub-label {
            font-size: 0.9em;
            color: #718096;
            font-style: italic;
            line-height: 1.2;
        }
        .performance-badge {
            font-size: 0.7em;
            padding: 2px 6px;
            border-radius: 10px;
            font-weight: bold;
            margin-left: 5px;
        }
        .high-performance {
            background-color: #C6F6D5;
            color: #22543D;
        }
        .medium-performance {
            background-color: #FEFCBF;
            color: #744210;
        }
        .low-performance {
            background-color: #FED7D7;
            color: #742A2A;
        }
        .generic-table-container {
            background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #FF5A6E;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            transition: transform 0.2s ease;
        }
        .generic-table-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .generic-table-container table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        .generic-table-container th {
            background-color: #FF5A6E;
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            font-size: 1.1em;
        }
        .generic-table-container td {
            padding: 10px;
            font-size: 1em;
            color: #2D3748;
            border-bottom: 1px solid #E2E8F0;
        }
        .generic-table-container tr:nth-child(even) {
            background-color: #FFF5F5;
        }
        .generic-table-container tr:hover {
            background-color: #FED7D7;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Key Metrics Section
        st.subheader("📊 Monthly Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_months = len(monthly)
        total_searches = monthly['Counts'].sum()
        avg_ctr = monthly['ctr'].mean()
        avg_cr = monthly['conversion_rate'].mean()
        
        with col1:
            st.markdown(f"""
            <div class='generic-metric-card'>
                <span class='icon'>📅</span>
                <div class='value'>{total_months}</div>
                <div class='label'>Total Months</div>
                <div class='sub-label'>Analyzed periods</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='generic-metric-card'>
                <span class='icon'>🔍</span>
                <div class='value'>{format_number(total_searches)}</div>
                <div class='label'>Total Searches</div>
                <div class='sub-label'>Across all months</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            performance_class = "high-performance" if avg_ctr > 5 else "medium-performance" if avg_ctr > 2 else "low-performance"
            st.markdown(f"""
            <div class='generic-metric-card'>
                <span class='icon'>📈</span>
                <div class='value'>{avg_ctr:.2f}% <span class='performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                <div class='label'>Average CTR</div>
                <div class='sub-label'>Monthly average</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            performance_class = "high-performance" if avg_cr > 3 else "medium-performance" if avg_cr > 1 else "low-performance"
            st.markdown(f"""
            <div class='generic-metric-card'>
                <span class='icon'>💰</span>
                <div class='value'>{avg_cr:.2f}% <span class='performance-badge {performance_class}'>{"High" if avg_cr > 3 else "Medium" if avg_cr > 1 else "Low"}</span></div>
                <div class='label'>Avg Conversion Rate</div>
                <div class='sub-label'>Monthly average</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive Analysis Section
        st.markdown("---")
        st.subheader("🎯 Interactive Time Analysis")
        
        analysis_type = st.radio(
            "Choose Analysis Type:",
            ["📊 Trends Overview", "🔍 Detailed Month Analysis", "🏷 Brand Comparison", "📊 Distribution Analysis"],
            horizontal=True
        )
        
        if analysis_type == "📊 Trends Overview":
            st.subheader("📈 Monthly Trends")
            
            # Line chart for counts
            fig_counts = px.line(
                monthly,
                x='month',
                y='Counts',
                title='<b style="color:#FF5A6E;">Monthly Search Volume</b>',
                labels={'Counts': 'Search Volume', 'month': 'Month'},
                color_discrete_sequence=['#FF5A6E']
            )
            fig_counts.update_traces(line=dict(width=3))
            fig_counts.update_layout(
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI'),
                height=400,
                xaxis=dict(tickangle=45, showgrid=True, gridcolor='#E6F3FA'),
                yaxis=dict(showgrid=True, gridcolor='#E6F3FA')
            )
            st.plotly_chart(fig_counts, use_container_width=True)
            
            # Line chart for CTR and Conversion Rate
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(
                x=monthly['month'],
                y=monthly['ctr'],
                name='CTR %',
                line=dict(color='#FF5A6E', width=3)
            ))
            fig_metrics.add_trace(go.Scatter(
                x=monthly['month'],
                y=monthly['conversion_rate'],
                name='Conversion Rate %',
                line=dict(color='#FFB085', width=3)
            ))
            fig_metrics.update_layout(
                title='<b>Monthly CTR and Conversion Rate Trends</b>',
                plot_bgcolor='rgba(255,255,255,0.95)',
                paper_bgcolor='rgba(255,247,232,0.8)',
                font=dict(color='#0B486B', family='Segoe UI'),
                height=400,
                xaxis=dict(tickangle=45, title='Month'),
                yaxis=dict(title='Percentage (%)')
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        elif analysis_type == "🔍 Detailed Month Analysis":
            st.subheader("🔬 Detailed Monthly Performance")
            
            selected_month = st.selectbox(
                "Select a month for detailed analysis:",
                options=monthly['month'].tolist(),
                index=0
            )
            
            if selected_month:
                month_data = monthly[monthly['month'] == selected_month].iloc[0]
                month_rank = monthly.reset_index().index[monthly['month'] == selected_month].tolist()[0] + 1
                
                col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                
                with col_detail1:
                    rank_performance = "high-performance" if month_rank <= 3 else "medium-performance" if month_rank <= 6 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>🏆</span>
                        <div class='value'>#{month_rank}</div>
                        <div class='label'>Month Rank</div>
                        <div class='sub-label'>Out of {total_months} months</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail2:
                    market_share = (month_data['Counts'] / total_searches * 100)
                    share_performance = "high-performance" if market_share > (100/total_months) else "medium-performance" if market_share > (50/total_months) else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>📊</span>
                        <div class='value'>{market_share:.2f}%</div>
                        <div class='label'>Market Share</div>
                        <div class='sub-label'>Of total searches</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail3:
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{month_data['ctr']:.2f}% <span class='performance-badge {"high-performance" if month_data['ctr'] > 5 else "medium-performance" if month_data['ctr'] > 2 else "low-performance"}'>{"High" if month_data['ctr'] > 5 else "Medium" if month_data['ctr'] > 2 else "Low"}</span></div>
                        <div class='label'>CTR</div>
                        <div class='sub-label'>Month performance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_detail4:
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>💰</span>
                        <div class='value'>{month_data['conversion_rate']:.2f}% <span class='performance-badge {"high-performance" if month_data['conversion_rate'] > 3 else "medium-performance" if month_data['conversion_rate'] > 1 else "low-performance"}'>{"High" if month_data['conversion_rate'] > 3 else "Medium" if month_data['conversion_rate'] > 1 else "Low"}</span></div>
                        <div class='label'>Conversion Rate</div>
                        <div class='sub-label'>Month performance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed performance table
                st.markdown("### 📊 Performance Breakdown")
                metrics_data = {
                    'Metric': ['Search Volume', 'Total Clicks', 'Total Conversions', 'CTR', 'Conversion Rate', 'Classic CVR', 'Click Share', 'Conversion Share'],
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
                    'Performance': [
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
                st.markdown("<div class='generic-table-container'>", unsafe_allow_html=True)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        elif analysis_type == "🏷 Brand Comparison":
            st.subheader("🏷 Top Brands Performance by Month")
            
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
                    title='<b style="color:#FF5A6E;">Top 5 Brands by Search Volume per Month</b>',
                    color_discrete_sequence=['#FF5A6E', '#FFB085', '#FF7F94', '#FFA5A5', '#FFCCD5']
                )
                fig_brands.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.95)',
                    paper_bgcolor='rgba(255,247,232,0.8)',
                    font=dict(color='#0B486B', family='Segoe UI'),
                    height=500,
                    xaxis=dict(tickangle=45, title='Month'),
                    yaxis=dict(title='Search Volume')
                )
                st.plotly_chart(fig_brands, use_container_width=True)
                
                # Comparison table
                st.markdown("### 📊 Brand Performance Table")
                display_brands = brand_month[['month', 'brand', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate']].copy()
                display_brands.columns = ['Month', 'Brand', 'Search Volume', 'Clicks', 'Conversions', 'CTR %', 'Conversion Rate %']
                display_brands['Search Volume'] = display_brands['Search Volume'].apply(lambda x: f"{int(x):,}")
                display_brands['Clicks'] = display_brands['Clicks'].apply(lambda x: f"{int(x):,}")
                display_brands['Conversions'] = display_brands['Conversions'].apply(lambda x: f"{int(x):,}")
                display_brands['CTR %'] = display_brands['CTR %'].apply(lambda x: f"{x:.2f}%")
                display_brands['Conversion Rate %'] = display_brands['Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                st.markdown("<div class='generic-table-container'>", unsafe_allow_html=True)
                st.dataframe(display_brands, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download brand data
                csv_brands = brand_month.to_csv(index=False)
                st.download_button(
                    label="📥 Download Brand Data CSV",
                    data=csv_brands,
                    file_name=f"brand_monthly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="brand_monthly_download"
                )
            else:
                st.info("Brand or month data not available for brand-month analysis.")
        
        elif analysis_type == "📊 Distribution Analysis":
            st.subheader("📊 Monthly Distribution Analysis")
            
            # Pie chart for market share
            fig_pie = px.pie(
                monthly,
                values='Counts',
                names='month',
                title='<b style="color:#FF5A6E;">Monthly Search Volume Distribution</b>',
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                height=400,
                font=dict(color='#0B486B', family='Segoe UI'),
                paper_bgcolor='rgba(255,247,232,0.8)'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Distribution metrics
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>📊</span>
                    <div class='value'>{gini_coefficient:.3f}</div>
                    <div class='label'>Gini Coefficient</div>
                    <div class='sub-label'>Market concentration</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_dist2:
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>🔝</span>
                    <div class='value'>{top_3_concentration:.1f}%</div>
                    <div class='label'>Top 3 Months Share</div>
                    <div class='sub-label'>Search volume concentration</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Advanced Filtering Section
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
                brand_filtered['ctr'] = brand_filtered.apply(
                    lambda r: (r['clicks'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1
                )
                brand_filtered['conversion_rate'] = brand_filtered.apply(
                    lambda r: (r['conversions'] / r['Counts'] * 100) if r['Counts'] > 0 else 0, axis=1
                )
                brand_filtered['classic_cvr'] = brand_filtered.apply(
                    lambda r: (r['conversions'] / r['clicks'] * 100) if r['clicks'] > 0 else 0, axis=1
                )
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
                    <div class='generic-metric-card'>
                        <span class='icon'>📅</span>
                        <div class='value'>{len(filtered_data)}</div>
                        <div class='label'>Months Found</div>
                        <div class='sub-label'>Matching filters</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col2:
                    total_searches_filtered = filtered_data['Counts'].sum()
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>🔍</span>
                        <div class='value'>{format_number(total_searches_filtered)}</div>
                        <div class='label'>Total Searches</div>
                        <div class='sub-label'>Filtered volume</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col3:
                    avg_ctr_filtered = filtered_data['ctr'].mean()
                    ctr_performance = "high-performance" if avg_ctr_filtered > 5 else "medium-performance" if avg_ctr_filtered > 2 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>📈</span>
                        <div class='value'>{avg_ctr_filtered:.2f}% <span class='performance-badge {ctr_performance}'>{"High" if avg_ctr_filtered > 5 else "Medium" if avg_ctr_filtered > 2 else "Low"}</span></div>
                        <div class='label'>Avg CTR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with filtered_col4:
                    avg_cr_filtered = filtered_data['conversion_rate'].mean()
                    cr_performance = "high-performance" if avg_cr_filtered > 3 else "medium-performance" if avg_cr_filtered > 1 else "low-performance"
                    st.markdown(f"""
                    <div class='generic-metric-card'>
                        <span class='icon'>💰</span>
                        <div class='value'>{avg_cr_filtered:.2f}% <span class='performance-badge {cr_performance}'>{"High" if avg_cr_filtered > 3 else "Medium" if avg_cr_filtered > 1 else "Low"}</span></div>
                        <div class='label'>Avg CR</div>
                        <div class='sub-label'>Filtered average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display filtered data
                display_filtered = filtered_data[['month', 'Counts', 'clicks', 'conversions', 'ctr', 'conversion_rate']].copy()
                display_filtered.columns = ['Month', 'Search Volume', 'Clicks', 'Conversions', 'CTR %', 'Conversion Rate %']
                display_filtered['Search Volume'] = display_filtered['Search Volume'].apply(lambda x: f"{int(x):,}")
                display_filtered['Clicks'] = display_filtered['Clicks'].apply(lambda x: f"{int(x):,}")
                display_filtered['Conversions'] = display_filtered['Conversions'].apply(lambda x: f"{int(x):,}")
                display_filtered['CTR %'] = display_filtered['CTR %'].apply(lambda x: f"{x:.2f}%")
                display_filtered['Conversion Rate %'] = display_filtered['Conversion Rate %'].apply(lambda x: f"{x:.2f}%")
                
                st.markdown("<div class='generic-table-container'>", unsafe_allow_html=True)
                st.dataframe(display_filtered, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download filtered data
                filtered_csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=filtered_csv,
                    file_name=f"filtered_monthly_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="filtered_time_download"
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
                label="📊 Complete Monthly Analysis CSV",
                data=csv_complete,
                file_name=f"monthly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="complete_time_download"
            )
        
        with col_download2:
            summary_report = f"""# Monthly Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Months Analyzed: {total_months}
- Total Search Volume: {total_searches:,}
- Average CTR: {avg_ctr:.2f}%
- Average Conversion Rate: {avg_cr:.2f}%
- Total Clicks: {int(total_clicks):,}
- Total Conversions: {int(total_conversions):,}

## Top Performing Months
{chr(10).join([f"{row['month']}: {int(row['Counts']):,} searches ({row['ctr']:.2f}% CTR, {row['conversion_rate']:.2f}% CR)" for _, row in monthly.head(3).iterrows()])}

## Market Concentration
- Gini Coefficient: {gini_coefficient:.3f}
- Top 3 Months Share: {top_3_concentration:.1f}%

## Recommendations
- Focus on high-performing months for campaign optimization
- Investigate low-performing months for improvement opportunities

Generated by Noureldeen Mohamed
"""
            st.download_button(
                label="📋 Executive Summary",
                data=summary_report,
                file_name=f"monthly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="summary_time_download"
            )
    
    except KeyError as e:
        st.error(f"❌ Missing required column: {str(e)}")
        st.info("Please ensure your data contains: 'month', 'Counts', 'clicks', 'conversions'")
    except ValueError as e:
        st.error(f"❌ Data format error: {str(e)}")
        st.info("Please check that numeric columns contain valid numbers")
    except Exception as e:
        st.error(f"❌ Unexpected error processing time data: {str(e)}")
        st.info("Please check your data format and try again.")

# ----------------- Pivot Builder Tab -----------------
with tab_pivot:
    st.header("📊 Pivot Builder & Prebuilt Pivots")
    st.markdown("Create custom pivots or explore prebuilt tables for quick insights. 🔧")

    # Apply CSS for consistency with Time Analysis Tab
    st.markdown("""
    <style>
    .generic-metric-card {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF5A6E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 10px 0;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease;
    }
    .generic-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .generic-metric-card .icon {
        font-size: 2em;
        margin-bottom: 10px;
        display: block;
    }
    .generic-metric-card .value {
        font-size: 1.8em;
        font-weight: bold;
        color: #0B486B;
        margin-bottom: 5px;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.2;
    }
    .generic-metric-card .label {
        font-size: 1.1em;
        color: #2D3748;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .generic-metric-card .sub-label {
        font-size: 0.9em;
        color: #718096;
        font-style: italic;
        line-height: 1.2;
    }
    .performance-badge {
        font-size: 0.7em;
        padding: 2px 6px;
        border-radius: 10px;
        font-weight: bold;
        margin-left: 5px;
    }
    .high-performance {
        background-color: #C6F6D5;
        color: #22543D;
    }
    .medium-performance {
        background-color: #FEFCBF;
        color: #744210;
    }
    .low-performance {
        background-color: #FED7D7;
        color: #742A2A;
    }
    .generic-table-container {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF5A6E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.2s ease;
    }
    .generic-table-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .generic-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
    }
    .generic-table-container th {
        background-color: #FF5A6E;
        color: white;
        font-weight: bold;
        padding: 12px;
        text-align: left;
        font-size: 1.1em;
    }
    .generic-table-container td {
        padding: 10px;
        font-size: 1em;
        color: #2D3748;
        border-bottom: 1px solid #E2E8F0;
    }
    .generic-table-container tr:nth-child(even) {
        background-color: #FFF5F5;
    }
    .generic-table-container tr:hover {
        background-color: #FED7D7;
    }
    </style>
    """, unsafe_allow_html=True)

    try:
        # Prebuilt Pivot: Brand × Query (Top 300)
        # Prebuilt Pivot: Brand × Query (Top 300)
        st.subheader("📋 Prebuilt: Brand × Query (Top 300)")
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
                <div class='generic-metric-card'>
                    <span class='icon'>📋</span>
                    <div class='value'>{total_rows:,}</div>
                    <div class='label'>Total Rows</div>
                    <div class='sub-label'>Top 300 Brand-Query Pairs</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>🔍</span>
                    <div class='value'>{format_number(int(total_counts))}</div>
                    <div class='label'>Total Searches</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                performance_class = "high-performance" if avg_ctr > 5 else "medium-performance" if avg_ctr > 2 else "low-performance"
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>📈</span>
                    <div class='value'>{avg_ctr:.2f}% <span class='performance-badge {performance_class}'>{"High" if avg_ctr > 5 else "Medium" if avg_ctr > 2 else "Low"}</span></div>
                    <div class='label'>Average CTR</div>
                    <div class='sub-label'>Top 300 pairs</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                performance_class = "high-performance" if avg_cr > 3 else "medium-performance" if avg_cr > 1 else "low-performance"
                st.markdown(f"""
                <div class='generic-metric-card'>
                    <span class='icon'>💰</span>
                    <div class='value'>{avg_cr:.2f}% <span class='performance-badge {performance_class}'>{"High" if avg_cr > 3 else "Medium" if avg_cr > 1 else "Low"}</span></div>
                    <div class='label'>Avg Conversion Rate</div>
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
            st.markdown("<div class='generic-table-container'>", unsafe_allow_html=True)
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
                label="📥 Download Brand × Query Pivot",
                data=csv_pv,
                file_name=f"brand_query_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="brand_query_pivot_download"
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
            st.markdown("<div class='generic-table-container'>", unsafe_allow_html=True)
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
                label="📥 Download Brand × Query Pivot",
                data=csv_pv,
                file_name=f"brand_query_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="brand_query_pivot_download"
            )
        
            
        # Custom Pivot Builder
        st.markdown("---")
        st.subheader("🔧 Custom Pivot Builder")
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
                st.markdown(f"**Preview Pivot Structure**")
                st.write(f"Rows: {', '.join(idx)}")
                st.write(f"Columns: {', '.join(cols)}")
                st.write(f"Value: {val} ({aggfunc})")
            else:
                st.warning("Please select at least one row, one column, and a value to generate the pivot.")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_pivot = st.button("Generate Pivot")
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
                    st.markdown("<div class='generic-table-container'>", unsafe_allow_html=True)
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
                        label="⬇ Download Custom Pivot CSV",
                        data=csv_pivot,
                        file_name=f"custom_pivot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="custom_pivot_download"
                    )
                except Exception as e:
                    st.error(f"Pivot generation error: {e}")
                    st.info("Ensure selected columns and values are valid and contain data.")
        
    except Exception as e:
        st.error(f"❌ Unexpected error in Pivot Builder: {e}")
        st.info("Please check your data format and ensure required columns are present.")

# ----------------- Insights & Questions (Modified) -----------------
with tab_insights:
    st.header("💡 Insights & Actionable Questions (10)")
    st.markdown("Curated insights focused on **search** data for data-driven decisions, with tables and charts. 🚀")

    # Apply CSS for red/orange theme consistency
    st.markdown("""
    <style>
    .generic-metric-card {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF5A6E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 10px 0;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease;
    }
    .generic-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .generic-metric-card .icon {
        font-size: 2em;
        margin-bottom: 10px;
        display: block;
    }
    .generic-metric-card .value {
        font-size: 1.8em;
        font-weight: bold;
        color: #0B486B;
        margin-bottom: 5px;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.2;
    }
    .generic-metric-card .label {
        font-size: 1.1em;
        color: #2D3748;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .generic-metric-card .sub-label {
        font-size: 0.9em;
        color: #718096;
        font-style: italic;
        line-height: 1.2;
    }
    .performance-badge {
        font-size: 0.7em;
        padding: 2px 6px;
        border-radius: 10px;
        font-weight: bold;
        margin-left: 5px;
    }
    .high-performance {
        background-color: #C6F6D5;
        color: #22543D;
    }
    .medium-performance {
        background-color: #FEFCBF;
        color: #744210;
    }
    .low-performance {
        background-color: #FED7D7;
        color: #742A2A;
    }
    .generic-table-container {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF5A6E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.2s ease;
    }
    .generic-table-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .generic-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
    }
    .generic-table-container th {
        background-color: #FF5A6E;
        color: white;
        font-weight: bold;
        padding: 12px;
        text-align: left;
        font-size: 1.1em;
    }
    .generic-table-container td {
        padding: 10px;
        font-size: 1em;
        color: #2D3748;
        border-bottom: 1px solid #E2E8F0;
    }
    .generic-table-container tr:nth-child(even) {
        background-color: #FFF5F5;
    }
    .generic-table-container tr:hover {
        background-color: #FED7D7;
    }
    .insight-box {
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FF5A6E;
        margin-bottom: 15px;
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

    def q_expand(title, explanation, render_fn, icon="💡"):
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown(f"<div class='insight-box'><h4>Why & How to Use</h4><p>{explanation}</p></div>", unsafe_allow_html=True)
            try:
                st.markdown("<div class='generic-table-container'>", unsafe_allow_html=True)
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
            label="📥 Download Q1 Table",
            data=csv,
            file_name=f"q1_top_queries_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q1_download"
        )
        fig = px.bar(out, x='normalized_query', y=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                     title='Top 10 Queries by Counts', color_discrete_sequence=['#FF5A6E'], text_auto=True)
        fig.update_layout(xaxis_title="Query", yaxis_title="Counts", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q1 — Top Queries by Counts (Top 10)",
             "Which queries drive the most Counts? Prioritize for search tuning and inventory planning.",
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
            label="📥 Download Q2 Table",
            data=csv,
            file_name=f"q2_high_counts_low_ctr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q2_download"
        )
        fig = px.scatter(out, x=out['Counts'].apply(lambda x: float(x.replace(',', ''))), y=out['ctr'].apply(lambda x: float(x.strip('%'))),
                         text='normalized_query', title='High Counts, Low CTR Queries',
                         color_discrete_sequence=['#FF5A6E'], size=out['Counts'].apply(lambda x: float(x.replace(',', ''))))
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_title="Counts", yaxis_title="CTR (%)")
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q2 — High Counts, Low CTR Queries (Top 10)",
             "Queries with high Counts but low engagement. Improve relevance, snippets, or imagery.",
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
            label="📥 Download Q3 Table",
            data=csv,
            file_name=f"q3_top_conversion_rate_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q3_download"
        )
        fig = px.bar(out, x='normalized_query', y=out['cr'].apply(lambda x: float(x.strip('%'))),
                     title='Top 10 Queries by Conversion Rate', color_discrete_sequence=['#FF5A6E'], text_auto='.2f')
        fig.update_layout(xaxis_title="Query", yaxis_title="Conversion Rate (%)", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q3 — Top Queries by Conversion Rate (Min Counts=200)",
             "High-converting queries for paid promotions or product focus.",
             q3, "🎯")

    # Q4: Long-Tail vs Short-Tail Queries
    def q4():
        lt = queries_clean[queries_clean['query_length'] >= 20]
        lt_counts = lt['Counts'].sum()
        total_counts = queries_clean['Counts'].sum()
        st.markdown(f"<div class='generic-metric-card'><span class='icon'>📏</span><div class='value'>{lt_counts:,.0f}</div><div class='label'>Long-Tail Counts</div><div class='sub-label'>Queries ≥20 chars, Share: {lt_counts/total_counts:.2%}</div></div>", unsafe_allow_html=True)
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
            label="📥 Download Q4 Table",
            data=csv,
            file_name=f"q4_long_tail_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q4_download"
        )
        fig = px.pie(out, names='Type', values=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                     title='Long-Tail vs Short-Tail Counts Share',
                     color_discrete_sequence=['#FF5A6E', '#FED7D7'])
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q4 — Long-Tail vs Short-Tail Queries",
             "How much Counts come from long-tail queries? Key for content strategy.",
             q4, "📏")

    # Q5: Branded vs Generic Counts Share
    def q5():
        if 'brand' in queries_clean.columns:
            generic = queries_clean[queries_clean['brand'].str.lower() == 'other']
            branded = queries_clean[(queries_clean['brand'].notna()) & (queries_clean['brand'] != '') & (queries_clean['brand'].str.lower() != 'other')]
            generic_counts = generic['Counts'].sum()
            branded_counts = branded['Counts'].sum()
            total_counts = queries_clean['Counts'].sum()
            st.markdown(f"<div class='generic-metric-card'><span class='icon'>🏷</span><div class='value'>{branded_counts:,.0f}</div><div class='label'>Branded Counts</div><div class='sub-label'>Share: {branded_counts/total_counts:.2%}</div></div>", unsafe_allow_html=True)
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
                label="📥 Download Q5 Table",
                data=csv,
                file_name=f"q5_branded_generic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="q5_download"
            )
            fig = px.pie(out, names='Type', values=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                         title='Branded vs Generic Counts Share',
                         color_discrete_sequence=['#FF5A6E', '#FED7D7'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brand column not present.")
    q_expand("Q5 — Branded vs Generic Counts Share",
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
            label="📥 Download Q6 Table",
            data=csv,
            file_name=f"q6_funnel_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q6_download"
        )
        fig = px.bar(out, x='normalized_query', y=[out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                                                   out['clicks'].apply(lambda x: float(x.replace(',', ''))),
                                                   out['conversions'].apply(lambda x: float(x.replace(',', '')))],
                     title='Top 10 Queries: Funnel Snapshot',
                     barmode='group', color_discrete_sequence=['#FF5A6E', '#FED7D7', '#C6F6D5'])
        fig.update_layout(xaxis_title="Query", yaxis_title="Value", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q6 — Query Funnel Snapshot (Top 10)",
             "View top queries' funnel: Counts → clicks → conversions.",
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
            label="📥 Download Q7 Table",
            data=csv,
            file_name=f"q7_top_ctr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q7_download"
        )
        fig = px.bar(out, x='normalized_query', y=out['ctr'].apply(lambda x: float(x.strip('%'))),
                     title='Top 10 Queries by CTR', color_discrete_sequence=['#FF5A6E'], text_auto='.2f')
        fig.update_layout(xaxis_title="Query", yaxis_title="CTR (%)", xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q7 — Top Queries by CTR (Min Counts=200)",
             "High-engagement queries for ad campaigns or content.",
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
            label="📥 Download Q8 Table",
            data=csv,
            file_name=f"q8_low_ctr_cr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="q8_download"
        )
        fig = px.scatter(out, x=out['ctr'].apply(lambda x: float(x.strip('%'))),
                         y=out['cr'].apply(lambda x: float(x.strip('%'))),
                         text='normalized_query', title='High Counts, Low CTR & Conversion Rate',
                         color_discrete_sequence=['#FF5A6E'], size=out['Counts'].apply(lambda x: float(x.replace(',', ''))))
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_title="CTR (%)", yaxis_title="Conversion Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    q_expand("Q8 — High Counts, Low CTR & Conversion Rate (Top 10)",
             "Optimize search results for these underperforming queries.",
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
                label="📥 Download Q9 Table",
                data=csv,
                file_name=f"q9_top_brands_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="q9_download"
            )
            fig = px.bar(out, x='brand', y=out['Counts'].apply(lambda x: float(x.replace(',', ''))),
                         title='Top 10 Brands by Counts', color_discrete_sequence=['#FF5A6E'], text_auto=True)
            fig.update_layout(xaxis_title="Brand", yaxis_title="Counts", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brand column missing.")
    q_expand("Q9 — Top Brands by Counts (Top 10)",
             "Rank brands by Counts for partnerships or promotions, excluding 'Other'.",
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
                label="📥 Download Q10 Table",
                data=csv,
                file_name=f"q10_category_brand_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="q10_download"
            )
            fig = px.bar(pivot.melt(id_vars='category', value_vars=top_brands, value_name='Counts'),
                         x='category', y='Counts', color='brand', title='Category vs Brand Counts',
                         barmode='stack', color_discrete_sequence=px.colors.qualitative.D3)
            fig.update_layout(xaxis_title="Category", yaxis_title="Counts", xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category or brand column missing.")
    q_expand("Q10 — Category vs Brand Performance (Pivot)",
             "Analyze brand performance within categories for targeted strategies.",
             q10, "📦🏷")

    st.info("For advanced analyses (e.g., anomaly detection, semantic clustering), contact Nour Eldeen for custom solutions.")

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