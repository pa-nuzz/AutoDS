"""AutoDS Streamlit UI - All Phases Complete (1-8)."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autods.data.ingestion import DataIngestion
from autods.analysis.profiler import StatisticalProfiler
from autods.preprocessing.orchestrator import PreprocessingOrchestrator
from autods.modeling.orchestrator import ModelOrchestrator
from autods.reports.pipeline import run_complete_analysis

# Phase 7 & 8 imports
from autods.utils.error_handler import ErrorHandler
from autods.utils.session_manager import SessionManager, auto_save_session
from autods.utils.checklist import AnalysisChecklist, get_checklist
from autods.utils.export_generator import ExportGenerator, generate_analysis_exports
from autods.utils.demo_data import DemoDatasetGenerator

# Page config
st.set_page_config(
    page_title="AutoData Analyst | AI-Powered Data Science",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Custom CSS - Light theme with blues/greys
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main theme */
    .main {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%);
        min-height: 100vh;
    }
    
    /* Header gradient banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        opacity: 0.3;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255,255,255,0.9);
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Panel Cards - Matching the 6-panel concept */
    .panel-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .panel-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .panel-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
    }
    
    .panel-card:hover::before {
        transform: scaleX(1);
    }
    
    .panel-icon {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        font-size: 2.5rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    .panel-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }
    
    .panel-desc {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Upload Zone */
    .upload-zone {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 3px dashed #cbd5e1;
        border-radius: 24px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .upload-zone:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        transform: scale(1.01);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .upload-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .upload-hint {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    /* Progress Cards */
    .progress-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .progress-icon {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
        border-color: #dbeafe;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3b82f6;
        line-height: 1;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 0.75rem;
        border-radius: 16px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        padding: 0 28px;
        border-radius: 12px;
        font-weight: 500;
        color: #64748b;
        transition: all 0.3s ease;
        background: transparent;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Status Boxes */
    .status-success {
        background: linear-gradient(135deg, #d1fae5 0%, #ecfdf5 100%);
        border-left: 4px solid #10b981;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .status-info {
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 3rem 2rem;
        margin-top: 4rem;
        background: linear-gradient(180deg, transparent 0%, #f8fafc 100%);
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* Robot Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .robot-float {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'auto_mode_results' not in st.session_state:
    st.session_state.auto_mode_results = None


def render_hero():
    """Render hero banner with 6-panel feature showcase."""
    st.markdown("""
    <div class="hero-banner animate-in">
        <div class="robot-float" style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <div class="hero-title">Welcome to AutoData Analyst</div>
        <div class="hero-subtitle">Upload your data, and let AI do the magic!</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 6 Feature Panels Row
    cols = st.columns(3)
    panels = [
        ("📁", "Upload Data", "CSV • Excel • JSON • Parquet"),
        ("🖼️", "Images", "JPG • PNG • Processed automatically"),
        ("🔗", "URL Import", "Kaggle • GitHub • Drive • Direct"),
        ("📊", "Smart Analysis", "Auto-detect patterns & insights"),
        ("🔧", "Preprocessing", "Clean • Transform • Prepare"),
        ("🤖", "AI Models", "Train • Evaluate • Deploy"),
    ]
    
    for i, (icon, title, desc) in enumerate(panels):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="panel-card animate-in" style="animation-delay: {i * 0.1}s;">
                <div class="panel-icon">{icon}</div>
                <div class="panel-title">{title}</div>
                <div class="panel-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def render_header():
    """Legacy - replaced by render_hero."""
    render_hero()


def render_upload_tab():
    """Render upload tab - Panel 1: Welcome concept."""
    st.markdown('<div class="section-header">📤 Upload Your Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">📁</div>
            <div class="upload-title">Drop files here</div>
            <div class="upload-hint">or click to browse</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("", 
                                   type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                                   label_visibility="collapsed")
        
        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                elif uploaded.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded)
                elif uploaded.name.endswith('.json'):
                    df = pd.read_json(uploaded)
                elif uploaded.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded)
                
                st.session_state.df = df
                st.markdown(f"""
                <div class="status-success">
                    ✅ <strong>Loaded successfully!</strong><br>
                    {df.shape[0]:,} rows × {df.shape[1]} columns
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("👁️ Preview Data"):
                    st.dataframe(df.head(8), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">🔗</div>
            <div class="upload-title">Import from URL</div>
            <div class="upload-hint">Kaggle • GitHub • Google Drive</div>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input("", placeholder="https://...", label_visibility="collapsed")
        
        if url and st.button("📥 Download Data", use_container_width=True):
            with st.spinner("Downloading..."):
                try:
                    ingestion = DataIngestion()
                    result = ingestion.from_url(url)
                    if result['success']:
                        st.session_state.df = result['data']
                        st.markdown(f"""
                        <div class="status-success">
                            ✅ <strong>Downloaded!</strong> {result['data'].shape[0]:,} rows loaded
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Download failed: {str(e)}")


def render_explore_tab():
    """Render data exploration tab with interactive visualizations."""
    st.markdown("<h2 style='color: #1e293b; margin-bottom: 1.5rem;'>🔍 Explore Your Data</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("👆 Please upload data first in the Upload tab")
        return
    
    df = st.session_state.df
    
    # Quick profile
    with st.spinner("Analyzing data..."):
        profiler = StatisticalProfiler(df)
        profile = profiler.get_profile()
        st.session_state.profile = profile
    
    # Metrics row
    overview = profile.get('overview', {})
    quality = profile.get('quality_score', {})
    
    cols = st.columns(4)
    metrics = [
        ("Rows", f"{overview.get('n_rows', 0):,}"),
        ("Columns", overview.get('n_columns', 0)),
        ("Missing %", f"{overview.get('missing_pct', 0):.1f}%"),
        ("Quality Score", f"{quality.get('overall', 0)}/100"),
    ]
    
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive visualizations
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data preview
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df, use_container_width=True)
    
    # Visualizations
    viz_tabs = st.tabs(["📊 Distributions", "📈 Correlations", "📉 Missing Values", "🔢 Data Types"])
    
    with viz_tabs[0]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        if numeric_cols:
            selected_col = st.selectbox("Select column", numeric_cols)
            fig = px.histogram(df, x=selected_col, 
                              title=f"Distribution of {selected_col}",
                              template="plotly_white",
                              color_discrete_sequence=['#3b82f6'])
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, 
                           text_auto=True, 
                           aspect="auto",
                           color_continuous_scale='Blues',
                           title="Correlation Matrix")
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    with viz_tabs[2]:
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing': missing.values,
                'Percentage': (missing.values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=True)
            
            fig = px.bar(missing_df, 
                        x='Missing', 
                        y='Column',
                        orientation='h',
                        color='Percentage',
                        color_continuous_scale='Blues',
                        title="Missing Values by Column")
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    with viz_tabs[3]:
        type_counts = df.dtypes.value_counts()
        fig = px.pie(values=type_counts.values, 
                    names=[str(t) for t in type_counts.index],
                    title="Data Type Distribution",
                    color_discrete_sequence=['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)


def render_preprocess_tab():
    """Render preprocessing tab with Auto/DIY toggle."""
    st.markdown("<h2 style='color: #1e293b; margin-bottom: 1.5rem;'>🔧 Preprocessing</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("👆 Please upload data first")
        return
    
    df = st.session_state.df
    
    # Mode selection
    st.markdown("<h3 style='color: #64748b; margin-bottom: 1rem;'>Choose Your Approach</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card" style="text-align: center; cursor: pointer;">
            <h2 style='color: #2563eb;'>⚡</h2>
            <h4 style='color: #1e293b;'>Do It For Me</h4>
            <p style='color: #64748b; font-size: 0.875rem;'>Auto-preprocess with intelligent defaults</p>
        </div>
        """, unsafe_allow_html=True)
        auto_mode = st.button("Select Auto Mode", key="auto_preprocess_btn", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center; cursor: pointer;">
            <h2 style='color: #2563eb;'>🛠️</h2>
            <h4 style='color: #1e293b;'>I Want To Do It</h4>
            <p style='color: #64748b; font-size: 0.875rem;'>Step-by-step guide with full control</p>
        </div>
        """, unsafe_allow_html=True)
        diy_mode = st.button("Select DIY Mode", key="diy_preprocess_btn", use_container_width=True)
    
    # Target column selection
    st.markdown("<br>", unsafe_allow_html=True)
    target_col = st.selectbox("🎯 Select Target Column (Optional)", 
                              ["None"] + df.columns.tolist())
    target = None if target_col == "None" else target_col
    
    # Auto Mode
    if auto_mode:
        with st.spinner("🔧 Running auto-preprocessing..."):
            try:
                orchestrator = PreprocessingOrchestrator(df, target)
                
                # Show needs summary
                needs = orchestrator.get_needs_summary()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Needs", needs.get('total_needs', 0))
                col2.metric("Required", needs.get('required', 0), delta_color="inverse")
                col3.metric("Recommended", needs.get('recommended', 0))
                
                # Run preprocessing
                result = orchestrator.run_auto(mode='balanced', save=False)
                st.session_state.processed_df = result['data']
                
                st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>Preprocessing Complete!</strong><br>
                    Applied {len(result['summary']['steps_applied'])} steps | 
                    New shape: {result['summary']['processed_shape']}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Steps Applied"):
                    for step in result['summary']['steps_applied']:
                        st.write(f"✓ {step}")
                
                with st.expander("Preview Processed Data"):
                    st.dataframe(result['data'].head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
    
    # DIY Mode
    if diy_mode:
        with st.spinner("📖 Generating DIY guide..."):
            try:
                orchestrator = PreprocessingOrchestrator(df, target)
                result = orchestrator.run_diy(save=False)
                
                st.markdown(result['markdown_guide'])
                
                if result.get('notebook_path'):
                    with open(result['notebook_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download Jupyter Notebook",
                            f.read(),
                            file_name="preprocessing_guide.ipynb",
                            mime="application/json"
                        )
                        
            except Exception as e:
                st.error(f"Error generating guide: {str(e)}")


def render_models_tab():
    """Render model recommendation and training tab."""
    st.markdown("<h2 style='color: #1e293b; margin-bottom: 1.5rem;'>🤖 Models & Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("👆 Please upload data first")
        return
    
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    # Target selection
    target = st.selectbox("🎯 Select Target Column", 
                         [c for c in df.columns if df[c].dtype != 'object'][:10])
    
    if not target:
        st.info("Please select a target column to proceed")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔍 Get Recommendations", use_container_width=True):
            with st.spinner("Analyzing task type and recommending models..."):
                try:
                    orchestrator = ModelOrchestrator(df, target)
                    recommendations = orchestrator.get_recommendations()
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>Task Type:</strong> {recommendations['task_type']}<br>
                        <strong>Description:</strong> {recommendations['task_info'].get('description', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("Top Recommendations")
                    for rec in recommendations['recommendations'][:3]:
                        with st.container():
                            st.markdown(f"""
                            <div class="card">
                                <h4>#{rec['rank']} {rec['model_name']}</h4>
                                <p style='color: #64748b; font-size: 0.875rem;'>{rec['reasoning']}</p>
                                <p><strong>Expected Performance:</strong> {rec['estimated_performance']['expected']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🚀 Train Baseline Models", use_container_width=True):
            with st.spinner("Training models (this may take a few minutes)..."):
                try:
                    orchestrator = ModelOrchestrator(df, target)
                    results = orchestrator.train_baseline()
                    st.session_state.model_results = results
                    
                    if 'error' in results:
                        st.error(results['error'])
                        return
                    
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>Training Complete!</strong> Trained {results['models_trained']} models
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results table
                    results_df = pd.DataFrame([
                        {
                            'Model': r['model_name'],
                            'Score': round(r['main_metric_value'], 4),
                            'CV Mean': round(r.get('cv_mean'), 4) if r.get('cv_mean') else '-',
                            'Time (s)': round(r['training_time'], 2)
                        }
                        for r in results['results']
                    ])
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best model highlight
                    best = results['best_model']
                    st.markdown(f"""
                    <div class="card" style="background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);">
                        <h4>🏆 Best Model: {best['model_name']}</h4>
                        <p><strong>{best['main_metric']}:</strong> {best['main_metric_value']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")


def render_reports_tab():
    """Render reports and export tab."""
    st.markdown("<h2 style='color: #1e293b; margin-bottom: 1.5rem;'>📊 Reports & Exports</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("👆 Please upload data first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style='color: #2563eb;'>📄 Generate Full Report</h3>
            <p style='color: #64748b;'>Comprehensive analysis with all phases</p>
        </div>
        """, unsafe_allow_html=True)
        
        target = st.selectbox("Target (optional)", 
                            ["None"] + st.session_state.df.columns.tolist(),
                            key="report_target")
        target_val = None if target == "None" else target
        
        formats = st.multiselect("Export Formats", 
                                ["HTML", "Jupyter Notebook", "Markdown", "JSON", "CSV"],
                                default=["HTML", "CSV"])
        
        if st.button("📊 Generate Report", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                try:
                    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
                    
                    results = run_complete_analysis(
                        df, 
                        target=target_val,
                        output_dir="reports",
                        use_llm=False,
                        preprocessing_mode='balanced',
                        train_models=bool(target_val)
                    )
                    
                    st.markdown(f"""
                    <div class="success-box">
                        ✅ <strong>Reports Generated!</strong><br>
                        Location: {results['output_directory']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show file locations
                    st.subheader("Generated Files")
                    for fmt, path in results['exported_files'].items():
                        st.text(f"{fmt.upper()}: {path}")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style='color: #2563eb;'>💾 Quick Export</h3>
            <p style='color: #64748b;'>Export current data</p>
        </div>
        """, unsafe_allow_html=True)
        
        export_df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        json_data = export_df.to_json(orient='records')
        st.download_button(
            "📥 Download JSON",
            json_data,
            file_name="data.json",
            mime="application/json",
            use_container_width=True
        )


def render_auto_mode_tab():
    """Render full auto mode - one click complete analysis."""
    st.markdown("<h2 style='color: #1e293b; margin-bottom: 1.5rem;'>⚡ Auto Mode - Full Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("👆 Please upload data first in the Upload tab")
        return
    
    df = st.session_state.df
    
    st.markdown("""
    <div class="info-box">
        <strong>🤖 Leave it to the AI Expert</strong><br>
        This mode will automatically: Profile your data → Preprocess → Recommend models → Train baselines → Generate reports
    </div>
    """, unsafe_allow_html=True)
    
    target = st.selectbox("🎯 Select Target Column (Optional - leave blank for exploratory analysis)", 
                         ["None (Exploratory)"] + df.columns.tolist())
    
    target_val = None if target == "None (Exploratory)" else target
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 START COMPLETE ANALYSIS", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Phase 1: Profiling (20%)
                status_text.text("📊 Phase 1/4: Profiling data...")
                profiler = StatisticalProfiler(df)
                profile = profiler.get_profile()
                progress_bar.progress(25)
                
                # Phase 2: Preprocessing (40%)
                status_text.text("🔧 Phase 2/4: Preprocessing...")
                prep_orchestrator = PreprocessingOrchestrator(df, target_val)
                prep_result = prep_orchestrator.run_auto(mode='balanced', save=False)
                processed_df = prep_result['data']
                progress_bar.progress(50)
                
                # Phase 3: Modeling (75%)
                if target_val:
                    status_text.text("🤖 Phase 3/4: Training models...")
                    model_orchestrator = ModelOrchestrator(processed_df, target_val)
                    model_results = model_orchestrator.train_baseline()
                else:
                    model_results = None
                progress_bar.progress(75)
                
                # Phase 4: Reports (100%)
                status_text.text("📄 Phase 4/4: Generating reports...")
                report_results = run_complete_analysis(
                    processed_df,
                    target=target_val,
                    output_dir="reports/auto_mode",
                    use_llm=False,
                    preprocessing_mode='balanced',
                    train_models=bool(target_val)
                )
                progress_bar.progress(100)
                
                status_text.text("✅ Analysis Complete!")
                
                # Display summary
                st.markdown("""
                <div class="success-box" style="padding: 2rem;">
                    <h3>✨ Analysis Complete!</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📊 Rows", f"{df.shape[0]:,}")
                col2.metric("📁 Columns", df.shape[1])
                col3.metric("🔧 Steps Applied", len(prep_result['summary']['steps_applied']))
                
                if model_results and model_results.get('best_model'):
                    col4.metric("🏆 Best Score", f"{model_results['best_model']['main_metric_value']:.3f}")
                else:
                    col4.metric("✅ Status", "Complete")
                
                # Download links
                st.subheader("📥 Download Your Reports")
                
                for fmt, path in report_results['exported_files'].items():
                    if Path(path).exists():
                        with open(path, 'rb') as f:
                            st.download_button(
                                f"📥 Download {fmt.upper()}",
                                f.read(),
                                file_name=Path(path).name,
                                use_container_width=True
                            )
                
                st.session_state.auto_mode_results = {
                    'profile': profile,
                    'preprocessing': prep_result,
                    'modeling': model_results,
                    'reports': report_results
                }
                
            except Exception as e:
                st.error(f"Error in auto mode: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def main():
    """Main app entry point."""
    render_header()
    
    # Create tabs
    tabs = st.tabs([
        "📤 Upload",
        "🔍 Explore", 
        "🔧 Preprocess",
        "🤖 Models",
        "📊 Reports",
        "⚡ Auto Mode"
    ])
    
    with tabs[0]:
        render_upload_tab()
    
    with tabs[1]:
        render_explore_tab()
    
    with tabs[2]:
        render_preprocess_tab()
    
    with tabs[3]:
        render_models_tab()
    
    with tabs[4]:
        render_reports_tab()
    
    with tabs[5]:
        render_auto_mode_tab()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🤖 AutoDS AI Assistant v1.0 | Your Intelligent Data Science Companion</p>
        <p style="font-size: 0.8rem;">Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
