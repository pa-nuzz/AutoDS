"""AutoDS - Automated Data Science Platform"""
import os
import sys
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
import io
import gc

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent / "src"))

from autods.analysis.profiler import StatisticalProfiler
from autods.analysis.type_detector import DataTypeDetector
from autods.preprocessing.orchestrator import PreprocessingOrchestrator
from autods.modeling.orchestrator import ModelOrchestrator
from autods.modeling.recommender import ModelRecommender
from autods.modeling.task_detector import TaskDetector
from autods.reports.pipeline import run_complete_analysis
from autods.utils.demo_data import load_demo_dataset

# Page config
st.set_page_config(
    page_title="AutoDS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS - professional style
st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu, footer, header, .stDeployButton {display: none !important;}

/* Clean typography */
.main {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;}

/* Step navigation */
.step-container {
    display: flex;
    gap: 8px;
    margin-bottom: 30px;
    padding: 15px 0;
    border-bottom: 1px solid #e5e5e5;
}
.step {
    padding: 6px 16px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 500;
}
.step-current {background: #0066cc; color: white;}
.step-done {background: #28a745; color: white;}
.step-todo {background: #f0f0f0; color: #666;}

/* Cards */
.card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 15px;
}

/* Tables */
.data-table {font-size: 13px;}
.data-table th {background: #f5f5f5; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
_INITIAL_STATE = {
    'df': None, 'processed_df': None, 'profile': None,
    'target_column': None, 'mode': None, 'current_step': 1,
    'analysis_done': False, 'preprocessing_done': False,
    'preprocessing_needs': None, 'models_trained': False,
    'training_in_progress': False,
    'model_results': None, 'model_recommendations': None,
    'report_generated': False, 'report_files': None,
    'guide_steps_completed': {},
}

for key, val in _INITIAL_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Utilities
def _get_memory_usage(df):
    bytes_used = df.memory_usage(deep=True).sum()
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024:
            return f"{bytes_used:.1f} {unit}"
        bytes_used /= 1024
    return f"{bytes_used:.1f} TB"


def _get_df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


@st.cache_data
def _run_profiler(df_hash, df_json):
    df = pd.read_json(df_json)
    return StatisticalProfiler(df).get_profile()


@st.cache_data
def _get_type_detector(df_hash, df_json):
    return DataTypeDetector(pd.read_json(df_json))


# Step indicator
def render_steps():
    steps = ["Upload", "Analyze", "Preprocess", "Model", "Export"]
    current = st.session_state.current_step
    
    html = '<div class="step-container">'
    for i, step in enumerate(steps, 1):
        if i < current:
            html += f'<div class="step step-done">{step}</div>'
        elif i == current:
            html += f'<div class="step step-current">{step}</div>'
        else:
            html += f'<div class="step step-todo">{step}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# Step 1: Upload
def render_step_1():
    st.markdown("## AutoDS")
    st.markdown("Automated data analysis and model training.")
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.markdown("### Upload Data")
        st.caption("Supports files up to 8GB (CSV, Excel, JSON, Parquet)")
        
        uploaded = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Large files (5GB+) are supported with chunked loading"
        )
        
        if uploaded:
            try:
                file_size = len(uploaded.getvalue())
                file_size_mb = file_size / (1024 * 1024)
                file_size_gb = file_size_mb / 1024
                
                st.info(f"File size: {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
                
                # Progress bar for large files
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Loading..."):
                    if uploaded.name.endswith('.csv'):
                        status_text.text("Reading CSV...")
                        
                        # For large CSV files, use chunked reading
                        if file_size_gb > 0.5:  # > 500MB
                            chunk_size = min(100000, max(10000, int(50000000 / file_size_gb)))  # Adaptive chunk size
                            
                            # First, read header to get columns
                            uploaded.seek(0)
                            header_df = pd.read_csv(uploaded, nrows=0)
                            columns = header_df.columns.tolist()
                            
                            # Read in chunks with progress
                            chunks = []
                            total_rows = 0
                            uploaded.seek(0)
                            
                            for i, chunk in enumerate(pd.read_csv(uploaded, chunksize=chunk_size, low_memory=False)):
                                chunks.append(chunk)
                                total_rows += len(chunk)
                                
                                # Update progress (estimate based on file size and rows read)
                                progress = min(0.9, (i * chunk_size * file_size) / (file_size * 10))  # Rough estimate
                                progress_bar.progress(min(0.9, progress))
                                status_text.text(f"Loaded {total_rows:,} rows...")
                                
                                # Force garbage collection
                                if i % 5 == 0:
                                    gc.collect()
                            
                            status_text.text("Combining chunks...")
                            df = pd.concat(chunks, ignore_index=True)
                            del chunks
                            gc.collect()
                        else:
                            df = pd.read_csv(uploaded, low_memory=False)
                            progress_bar.progress(0.5)
                            
                    elif uploaded.name.endswith(('.xlsx', '.xls')):
                        status_text.text("Reading Excel...")
                        df = pd.read_excel(uploaded)
                        progress_bar.progress(0.5)
                        
                    elif uploaded.name.endswith('.json'):
                        status_text.text("Reading JSON...")
                        df = pd.read_json(uploaded)
                        progress_bar.progress(0.5)
                        
                    elif uploaded.name.endswith('.parquet'):
                        status_text.text("Reading Parquet...")
                        df = pd.read_parquet(uploaded)
                        progress_bar.progress(0.5)
                    
                    progress_bar.progress(0.7)
                    status_text.text("Processing...")
                    
                    # Memory management: Sample if too large
                    original_rows = len(df)
                    MAX_ROWS = 200000  # Increased from 100k
                    MAX_MEMORY_GB = 4  # Keep under 4GB memory usage
                    
                    current_memory_gb = df.memory_usage(deep=True).sum() / (1024**3)
                    
                    if original_rows > MAX_ROWS or current_memory_gb > MAX_MEMORY_GB:
                        if original_rows > MAX_ROWS:
                            df = df.sample(n=MAX_ROWS, random_state=42)
                            st.warning(f"Dataset sampled from {original_rows:,} to {MAX_ROWS:,} rows for performance")
                        if current_memory_gb > MAX_MEMORY_GB:
                            # Further reduce if still too large
                            target_rows = int(MAX_ROWS * (MAX_MEMORY_GB / current_memory_gb))
                            df = df.sample(n=min(target_rows, len(df)), random_state=42)
                            st.warning(f"Dataset further sampled to fit memory constraints")
                        gc.collect()
                    
                    progress_bar.progress(0.9)
                    status_text.text("Finalizing...")
                    
                    st.session_state.df = df
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    st.success(f"Loaded: {len(df):,} rows x {len(df.columns)} columns ({_get_memory_usage(df)})")
                    
                    st.dataframe(df.head(), use_container_width=True, height=200)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rows", f"{len(df):,}")
                    c2.metric("Columns", len(df.columns))
                    c3.metric("Memory", _get_memory_usage(df))
                    c4.metric("Missing", f"{df.isnull().sum().sum():,}")
                    
                    if st.button("Continue to Analysis", type="primary"):
                        st.session_state.current_step = 2
                        st.rerun()
                        
            except MemoryError:
                st.error("File too large for available memory. Try a smaller file or increase system RAM.")
            except Exception as e:
                st.error(f"Failed to load: {str(e)}")
        
        st.markdown("---")
        url = st.text_input("Or enter a URL", placeholder="https://...")
        if url and st.button("Load from URL"):
            try:
                with st.spinner("Loading from URL..."):
                    df = pd.read_csv(url)
                    if len(df) > 200000:
                        df = df.sample(n=200000, random_state=42)
                        st.warning("Dataset sampled to 200,000 rows for performance")
                    st.session_state.df = df
                    st.session_state.current_step = 2
                    st.rerun()
            except Exception as e:
                st.error(f"Failed: {str(e)}")
    
    with col2:
        st.markdown("### Demo Datasets")
        
        demos = [
            ('customer_churn', 'Customer Churn', 'Classification', '1,000'),
            ('house_prices', 'House Prices', 'Regression', '1,000'),
            ('iris_extended', 'Iris Extended', 'Classification', '300'),
            ('employee_attrition', 'Employee Attrition', 'Classification', '800'),
        ]
        
        for key, name, task, rows in demos:
            with st.container():
                st.markdown(f"**{name}** — {task} | {rows} rows")
                if st.button(f"Load {name}", key=f"demo_{key}", use_container_width=True):
                    with st.spinner("Loading..."):
                        df = load_demo_dataset(key)
                        st.session_state.df = df
                        st.session_state.target_column = {
                            'customer_churn': 'churn',
                            'house_prices': 'price',
                            'iris_extended': 'species',
                            'employee_attrition': 'attrition',
                        }.get(key)
                        st.session_state.current_step = 2
                        st.rerun()


# Step 2: Analyze
def render_step_2():
    df = st.session_state.df
    if df is None:
        st.session_state.current_step = 1
        st.rerun()
        return
    
    col_back, col_title = st.columns([0.1, 0.9])
    with col_back:
        if st.button("Back", key="back_2"):
            st.session_state.current_step = 1
            st.session_state.analysis_done = False
            st.session_state.profile = None
            st.rerun()
    with col_title:
        st.markdown("## Dataset Analysis")
    
    # Run analysis
    if not st.session_state.analysis_done:
        with st.spinner("Analyzing..."):
            try:
                df_json = df.head(1000).to_json()
                profile = _run_profiler(_get_df_hash(df), df_json)
                st.session_state.profile = profile
                st.session_state.analysis_done = True
                
                if not st.session_state.target_column:
                    suggestion = profile.get('target_suggestion', {})
                    if suggestion.get('has_good_candidate'):
                        st.session_state.target_column = suggestion.get('suggested_column')
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return
    
    profile = st.session_state.profile
    overview = profile.get('overview', {})
    quality = profile.get('quality_score', {})
    
    # Overview metrics
    st.markdown("### Overview")
    cols = st.columns(5)
    metrics = [
        ("Rows", f"{overview.get('n_rows', 0):,}"),
        ("Columns", str(overview.get('n_columns', 0))),
        ("Quality", f"{quality.get('overall', 0):.0f}/100"),
        ("Missing", f"{overview.get('missing_pct', 0):.1f}%"),
        ("Duplicates", f"{overview.get('duplicate_rows', 0):,}"),
    ]
    for i, (label, value) in enumerate(metrics):
        cols[i].metric(label, value)
    
    # Column types
    st.markdown("---")
    st.markdown("### Column Types")
    
    type_detector = _get_type_detector(_get_df_hash(df), df.head(1000).to_json())
    type_counts = {}
    for col in df.columns:
        info = type_detector.columns_info.get(col, {})
        t = info.get('type', 'unknown').replace('_', ' ').title()
        type_counts[t] = type_counts.get(t, 0) + 1
    
    type_cols = st.columns(len(type_counts) if type_counts else 1)
    for i, (t, count) in enumerate(type_counts.items()):
        type_cols[i].metric(t, count)
    
    # Visualizations
    st.markdown("---")
    st.markdown("### Visualizations")
    
    tabs = st.tabs(["Distributions", "Correlations", "Missing Values"])
    
    with tabs[0]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) > 0:
            fig = make_subplots(rows=2, cols=3, subplot_titles=list(numeric_cols))
            for i, col in enumerate(numeric_cols):
                row, col_idx = i // 3 + 1, i % 3 + 1
                fig.add_trace(go.Histogram(x=df[col], name=col), row=row, col=col_idx)
            fig.update_layout(height=500, showlegend=False,
                paper_bgcolor='white', plot_bgcolor='#fafafa',
                font=dict(size=11, color='#000'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns")
    
    with tabs[1]:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu',
                           aspect='auto', zmin=-1, zmax=1,
                           text_auto='.2f')
            fig.update_layout(height=500, paper_bgcolor='white', plot_bgcolor='#fafafa',
                            font=dict(color='#000'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 2+ numeric columns")
    
    with tabs[2]:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            fig = px.bar(x=missing.index, y=missing.values,
                        labels={'x': 'Column', 'y': 'Count'},
                        color=missing.values,
                        color_continuous_scale=['#28a745', '#ffc107', '#dc3545'])
            fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='#fafafa',
                            font=dict(color='#000'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values")
    
    # AI Suggestions
    st.markdown("---")
    st.markdown("### Suggestions")
    
    suggestions = []
    if overview.get('missing_pct', 0) > 5:
        suggestions.append(f"{overview.get('missing_pct', 0):.1f}% missing values - consider imputation")
    
    high_corr = profile.get('correlations', {}).get('high_correlations', [])
    if high_corr:
        top = high_corr[0]
        suggestions.append(f"'{top['col1']}' and '{top['col2']}' highly correlated ({top['correlation']:.2f})")
    
    for msg in suggestions[:3]:
        st.info(msg)
    
    # Target selection
    st.markdown("---")
    st.markdown("### Target Column (Optional)")
    
    target_options = ["None (Exploratory)"] + list(df.columns)
    default = 0
    if st.session_state.target_column and st.session_state.target_column in target_options:
        default = target_options.index(st.session_state.target_column)
    
    selected = st.selectbox("Select target", target_options, index=default)
    target = None if selected == "None (Exploratory)" else selected
    st.session_state.target_column = target
    
    if target:
        try:
            task = TaskDetector(df, target).get_task_info()
            st.caption(f"Detected: {task.get('type', 'unknown').replace('_', ' ').title()}")
        except:
            pass
    
    # Mode selection
    st.markdown("---")
    st.markdown("### Choose Mode")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Auto Mode** — Let AI handle everything automatically.")
        if st.button("Start Auto", type="primary", use_container_width=True):
            st.session_state.mode = 'auto'
            st.session_state.current_step = 3
            st.rerun()
    
    with col2:
        st.markdown("**Guide Mode** — Step-by-step with full control.")
        if st.button("Start Guided", use_container_width=True):
            st.session_state.mode = 'guide'
            st.session_state.current_step = 3
            st.rerun()


# Step 3: Preprocess
def render_step_3():
    df = st.session_state.df
    if df is None:
        st.session_state.current_step = 1
        st.rerun()
        return
    
    col_back, col_title = st.columns([0.1, 0.9])
    with col_back:
        if st.button("Back", key="back_3"):
            st.session_state.current_step = 2
            st.session_state.preprocessing_done = False
            st.session_state.preprocessing_needs = None
            st.rerun()
    with col_title:
        st.markdown("## Preprocessing")
    
    # Quality overview
    st.markdown("### Data Quality")
    cols = st.columns(4)
    cols[0].metric("Missing", f"{df.isnull().sum().sum():,}")
    cols[1].metric("Duplicates", f"{df.duplicated().sum():,}")
    cols[2].metric("Numeric", len(df.select_dtypes(include=[np.number]).columns))
    cols[3].metric("Categorical", len(df.select_dtypes(include=['object']).columns))
    
    # Get needs
    if st.session_state.preprocessing_needs is None:
        with st.spinner("Analyzing..."):
            try:
                prep = PreprocessingOrchestrator(df, target_column=st.session_state.target_column)
                st.session_state.preprocessing_needs = prep.get_needs_summary()
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {str(e)}")
                return
    
    needs = st.session_state.preprocessing_needs
    
    if st.session_state.mode == 'auto':
        render_auto_preprocessing(df, needs)
    else:
        render_guide_preprocessing(df, needs)


def render_auto_preprocessing(df, needs):
    details = needs.get('details', [])
    required = needs.get('required', 0)
    recommended = needs.get('recommended', 0)
    
    if not st.session_state.preprocessing_done:
        st.markdown("---")
        st.markdown("### Detected Issues")
        
        if required == 0 and recommended == 0:
            st.success("Data is clean - no preprocessing needed.")
        else:
            required_items = [d for d in details if d.get('priority') == 'required']
            recommended_items = [d for d in details if d.get('priority') == 'recommended']
            
            if required_items:
                st.markdown("**Required:**")
                for item in required_items[:5]:
                    st.markdown(f"- {item.get('type', 'Unknown')}: {item.get('description', '')}")
            
            if recommended_items:
                st.markdown("**Recommended:**")
                for item in recommended_items[:5]:
                    st.markdown(f"- {item.get('type', 'Unknown')}: {item.get('description', '')}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Skip (Use Original)", use_container_width=True):
                st.session_state.processed_df = df
                st.session_state.preprocessing_done = True
                st.rerun()
        with col2:
            if st.button("Run Preprocessing", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        prep = PreprocessingOrchestrator(df, target_column=st.session_state.target_column)
                        result = prep.run_auto(mode='balanced', save=False)
                        st.session_state.processed_df = result['data']
                        st.session_state.preprocessing_done = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")
    else:
        st.success("Preprocessing complete")
        
        orig = df.shape
        proc = st.session_state.processed_df.shape
        
        st.markdown("### Results")
        comp_cols = st.columns(3)
        comp_cols[0].metric("Rows", f"{orig[0]:,}", f"{proc[0]:,}")
        comp_cols[1].metric("Columns", orig[1], proc[1])
        comp_cols[2].metric("Missing", f"{df.isnull().sum().sum():,}", "0")
        
        if st.button("Continue to Modeling", type="primary", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()


def render_guide_preprocessing(df, needs):
    details = needs.get('details', [])
    total = len(details)
    completed = sum(1 for step, done in st.session_state.guide_steps_completed.items() if done)
    
    st.markdown("---")
    st.markdown("### Preprocessing Steps")
    
    if total > 0:
        st.progress(completed / total)
        st.caption(f"{completed}/{total} completed")
        
        for need in details[:5]:
            need_id = f"{need.get('type', '')}_{need.get('column', '')}"
            is_done = st.session_state.guide_steps_completed.get(need_id, False)
            priority = need.get('priority', 'optional')
            
            with st.expander(f"[{priority.upper()}] {need.get('type', 'Unknown')}: {need.get('description', '')}"):
                checked = st.checkbox("Done", value=is_done, key=f"check_{need_id}")
                if checked != is_done:
                    st.session_state.guide_steps_completed[need_id] = checked
                    st.rerun()
                st.code(need.get('code_snippet', '# No code'), language='python')
    else:
        st.success("No preprocessing required")
    
    required_count = sum(1 for n in details if n.get('priority') == 'required')
    required_done = sum(1 for n in details if n.get('priority') == 'required' 
                       and st.session_state.guide_steps_completed.get(f"{n.get('type', '')}_{n.get('column', '')}", False))
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Auto Apply & Continue", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    prep = PreprocessingOrchestrator(df, target_column=st.session_state.target_column)
                    result = prep.run_auto(mode='balanced', save=False)
                    st.session_state.processed_df = result['data']
                    st.session_state.preprocessing_done = True
                    st.session_state.current_step = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {str(e)}")
    
    with col2:
        if total == 0 or required_done >= required_count:
            if st.button("Continue to Modeling", type="primary", use_container_width=True):
                st.session_state.processed_df = st.session_state.processed_df or df
                st.session_state.preprocessing_done = True
                st.session_state.current_step = 4
                st.rerun()
        else:
            st.caption(f"{required_count - required_done} required steps remaining")


# Step 4: Model
def render_step_4():
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    if df is None:
        st.session_state.current_step = 1
        st.rerun()
        return
    
    col_back, col_title = st.columns([0.1, 0.9])
    with col_back:
        if st.button("Back", key="back_4"):
            st.session_state.current_step = 3
            st.session_state.models_trained = False
            st.session_state.model_results = None
            st.session_state.model_recommendations = None
            st.rerun()
    with col_title:
        st.markdown("## Model Training")
    
    # Target
    col1, col2 = st.columns([1, 3])
    col1.markdown("**Target**")
    
    target_options = ["None"] + [c for c in df.columns if df[c].dtype != 'object'][:20]
    default = 0
    if st.session_state.target_column:
        if st.session_state.target_column in target_options:
            default = target_options.index(st.session_state.target_column)
        else:
            target_options.insert(1, st.session_state.target_column)
            default = 1
    
    selected = col2.selectbox("Select", target_options, index=default, label_visibility="collapsed", key="target_4")
    target = None if selected == "None" else selected
    st.session_state.target_column = target
    
    if not target:
        st.info("No target selected. Skip to Export to download data.")
        if st.button("Skip to Export"):
            st.session_state.current_step = 5
            st.rerun()
        return
    
    # Task detection
    try:
        task_detector = TaskDetector(df, target)
        task_info = task_detector.get_task_info()
        task_type = task_info.get('type', 'unknown')
        st.caption(f"Task: {task_type.replace('_', ' ').title()}")
    except Exception as e:
        st.error(f"Detection failed: {str(e)}")
        return
    
    # Get recommendations
    if st.session_state.model_recommendations is None:
        with st.spinner("Getting recommendations..."):
            try:
                dataset_info = {'n_samples': len(df), 'has_missing_values': df.isnull().any().any()}
                recommender = ModelRecommender(task_detector, dataset_info)
                st.session_state.model_recommendations = recommender.get_recommendations(top_k=3)
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {str(e)}")
                return
    
    recommendations = st.session_state.model_recommendations
    
    # Recommendations
    st.markdown("---")
    st.markdown("### Recommended Models")
    
    rec_cols = st.columns(3)
    for i, rec in enumerate(recommendations[:3]):
        with rec_cols[i]:
            st.markdown(f"**#{i+1}: {rec.get('model_name', 'Unknown')}**")
            st.caption(f"Match: {rec.get('confidence_score', 0):.0f}%")
            st.markdown(f"<small>{rec.get('reasoning', '')[:80]}...</small>", unsafe_allow_html=True)
    
    # Training
    st.markdown("---")
    
    if not st.session_state.models_trained and not st.session_state.training_in_progress:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Only", use_container_width=True):
                pass
        with col2:
            if st.button("Train All", type="primary", use_container_width=True):
                st.session_state.training_in_progress = True
                st.rerun()
    
    elif st.session_state.training_in_progress and not st.session_state.models_trained:
        error_container = st.empty()
        with st.spinner("Training models..."):
            try:
                model_orch = ModelOrchestrator(df, target_column=target)
                results = model_orch.train_baseline(save=False)
                
                if results.get('error'):
                    error_container.error(f"Training error: {results['error']}")
                    st.session_state.training_in_progress = False
                elif results.get('models_trained', 0) == 0:
                    error_container.error("No models trained. Check if numeric features exist after preprocessing.")
                    st.session_state.training_in_progress = False
                else:
                    st.session_state.model_results = results
                    st.session_state.models_trained = True
                    st.session_state.training_in_progress = False
                    st.rerun()
            except Exception as e:
                st.session_state.training_in_progress = False
                error_container.error(f"Training failed: {str(e)}")
    
    elif st.session_state.models_trained and st.session_state.model_results:
        results = st.session_state.model_results
        st.success(f"{results.get('models_trained', 0)} models trained")
        
        if 'results' in results and results['results']:
            results_df = pd.DataFrame([
                {'Model': r['model_name'],
                 'Score': r['main_metric_value'],
                 'CV': r.get('cv_mean', '-'),
                 'Time': round(r['training_time'], 2)}
                for r in results['results']
            ]).sort_values('Score', ascending=False)
            
            fig = px.bar(results_df, x='Model', y='Score',
                        text=results_df['Score'].apply(lambda x: f'{x:.3f}'),
                        color_discrete_sequence=['#0066cc'])
            fig.update_traces(textposition='outside', textfont=dict(color='#000', size=12))
            fig.update_layout(height=400, paper_bgcolor='white', plot_bgcolor='#fafafa',
                            font=dict(color='#000'), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Best model
        if results.get('best_model'):
            bm = results['best_model']
            st.markdown("---")
            st.markdown("### Best Model")
            
            bm_cols = st.columns(3)
            bm_cols[0].metric("Model", bm['model_name'])
            bm_cols[1].metric("Metric", bm['main_metric'])
            bm_cols[2].metric("Score", f"{bm['main_metric_value']:.4f}")
            
            if bm.get('feature_importance'):
                st.markdown("**Feature Importance**")
                fi_df = pd.DataFrame(bm['feature_importance']).head(10)
                fig = px.bar(fi_df, x='importance', y='feature', orientation='h',
                           color_discrete_sequence=['#0066cc'],
                           text=fi_df['importance'].apply(lambda x: f'{x:.3f}'))
                fig.update_traces(textposition='outside', textfont=dict(color='#000', size=11))
                fig.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                font=dict(color='#000'), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Download Model**")
            model_bytes = pickle.dumps(bm.get('model'))
            st.download_button(f"Download {bm['model_name']}.pkl", model_bytes,
                           file_name=f"model_{bm['model_name']}.pkl",
                           mime="application/octet-stream")
        
        if st.button("Continue to Export", type="primary", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()


# Step 5: Export
def render_step_5():
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    if df is None:
        st.session_state.current_step = 1
        st.rerun()
        return
    
    col_back, col_title = st.columns([0.1, 0.9])
    with col_back:
        if st.button("Back", key="back_5"):
            st.session_state.current_step = 4
            st.rerun()
    with col_title:
        st.markdown("## Export")
    
    # Progress
    st.markdown("### Summary")
    prog_cols = st.columns(5)
    steps = [
        ("Data", st.session_state.df is not None, f"{len(st.session_state.df)} rows" if st.session_state.df is not None else ""),
        ("Analysis", st.session_state.analysis_done, "Done" if st.session_state.analysis_done else ""),
        ("Preprocess", st.session_state.preprocessing_done, "Done" if st.session_state.preprocessing_done else ""),
        ("Models", st.session_state.models_trained, f"{st.session_state.model_results.get('models_trained', 0)} trained" if st.session_state.model_results else ""),
        ("Report", st.session_state.report_generated, "Ready" if st.session_state.report_generated else ""),
    ]
    for i, (label, done, detail) in enumerate(steps):
        prog_cols[i].metric(label, "Complete" if done else "Pending", detail)
    
    # Generate report
    st.markdown("---")
    st.markdown("### Generate Report")
    
    if not st.session_state.report_generated:
        if st.button("Generate", type="primary", use_container_width=True):
            with st.spinner("Generating..."):
                try:
                    results = run_complete_analysis(
                        df,
                        target=st.session_state.target_column,
                        output_dir="reports",
                        use_llm=False
                    )
                    st.session_state.report_generated = True
                    st.session_state.report_files = results.get('exported_files', {})
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {str(e)}")
    else:
        st.success("Report generated")
    
    # Downloads
    st.markdown("---")
    st.markdown("### Downloads")
    
    downloads = [
        ("processed_data.csv", "Clean dataset (CSV)", lambda: df.to_csv(index=False).encode('utf-8'), "text/csv"),
        ("data.json", "Data (JSON)", lambda: df.to_json(orient='records').encode('utf-8'), "application/json"),
    ]
    
    if st.session_state.report_generated and st.session_state.report_files:
        if st.session_state.report_files.get('html'):
            downloads.append(("report.html", "Analysis Report (HTML)",
                            lambda: st.session_state.report_files['html'], "text/html"))
        if st.session_state.report_files.get('notebook'):
            downloads.append(("notebook.ipynb", "Jupyter Notebook",
                            lambda: st.session_state.report_files['notebook'], "application/json"))
    
    if st.session_state.model_results and st.session_state.model_results.get('best_model'):
        bm = st.session_state.model_results['best_model']
        model_bytes = pickle.dumps(bm.get('model'))
        downloads.append((f"model_{bm['model_name']}.pkl", "Trained Model",
                        lambda: model_bytes, "application/octet-stream"))
    
    dl_cols = st.columns(3)
    for i, (filename, desc, get_data, mime) in enumerate(downloads):
        with dl_cols[i % 3]:
            try:
                data = get_data()
                st.download_button(f"{filename}", data, file_name=filename, mime=mime,
                                 key=f"dl_{i}", use_container_width=True)
                st.caption(desc)
            except Exception as e:
                st.button(f"{filename}", disabled=True, key=f"dl_disabled_{i}", use_container_width=True)
    
    # Reset
    st.markdown("---")
    if st.button("Start New Analysis", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# Main
def main():
    if st.session_state.current_step > 1 and st.session_state.df is None:
        st.session_state.current_step = 1
        st.rerun()
        return
    
    render_steps()
    st.divider()
    
    if st.session_state.current_step == 1:
        render_step_1()
    elif st.session_state.current_step == 2:
        render_step_2()
    elif st.session_state.current_step == 3:
        render_step_3()
    elif st.session_state.current_step == 4:
        render_step_4()
    elif st.session_state.current_step == 5:
        render_step_5()


if __name__ == "__main__":
    main()
