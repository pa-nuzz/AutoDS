"""AutoDS Streamlit UI - 5-Step Flow."""
import os
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from autods.analysis.profiler import StatisticalProfiler
from autods.analysis.insights import InsightEngine
from autods.preprocessing.orchestrator import PreprocessingOrchestrator
from autods.modeling.orchestrator import ModelOrchestrator
from autods.modeling.recommender import ModelRecommender
from autods.reports.pipeline import run_complete_analysis
from autods.utils.demo_data import DemoDatasetGenerator
from autods.utils.error_handler import ErrorHandler

st.set_page_config(page_title="AutoDS", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main, .stApp { background-color: #FAF9F6 !important; }
h1, h2, h3, h4, p, div { color: #1C1C1E !important; }
.stButton > button[kind="primary"] { background-color: #1C1C1E !important; color: #FFFFFF !important; border-radius: 12px !important; }
.stButton > button[kind="secondary"] { background-color: #F0EFEC !important; color: #1C1C1E !important; border-radius: 12px !important; border: 1px solid #E8E6E1 !important; }
.card { background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 16px; padding: 24px; margin: 16px 0; }
.metric-card { background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 12px; padding: 20px; text-align: center; }
.metric-value { font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.875rem; color: #6B6B6B; }
.upload-zone { background: #FFFFFF; border: 2px dashed #E8E6E1; border-radius: 16px; padding: 40px; text-align: center; }
.demo-card { background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 12px; }
.suggestion-card { background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 12px; padding: 20px; margin: 12px 0; }
.success-banner { background: #FFFFFF; border: 1px solid #10B981; border-radius: 12px; padding: 16px; display: flex; align-items: center; gap: 12px; margin: 16px 0; }
.stTabs [data-baseweb="tab-list"] { background-color: #F0EFEC; border-radius: 12px; padding: 6px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #6B6B6B; border-radius: 8px; padding: 10px 20px; font-weight: 500; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #1C1C1E; color: #FFFFFF; }
.progress-bar-container { background: #F0EFEC; border-radius: 12px; padding: 16px; margin-bottom: 20px; }
.progress-step { display: inline-flex; align-items: center; margin-right: 8px; font-size: 0.85rem; }
.progress-step.completed { color: #10B981; }
.progress-step.active { color: #1C1C1E; font-weight: 600; }
.progress-step.locked { color: #9CA3AF; }
.need-tag { display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 500; margin: 4px; }
.need-tag.required { background: #FEE2E2; color: #DC2626; }
.need-tag.recommended { background: #FEF3C7; color: #D97706; }
.need-tag.optional { background: #D1FAE5; color: #059669; }
.model-card { background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 16px; padding: 20px; margin: 12px 0; }
.model-card.recommended { border-color: #10B981; border-left: 4px solid #10B981; }
.checklist-item { display: flex; align-items: center; padding: 8px 0; }
.checklist-item input[type="checkbox"] { margin-right: 12px; }
.export-option { display: flex; justify-content: space-between; align-items: center; padding: 12px; background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 8px; margin: 8px 0; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

_INITIAL_STATE = {'df': None, 'processed_df': None, 'profile': None, 'insights': None,
    'model_results': None, 'target_column': None, 'mode': None, 'data_loaded': False,
    'analysis_done': False, 'preprocessing_done': False, 'models_trained': False,
    'current_step': 1, 'analysis_plan': None, 'preprocessing_needs': None,
    'guide_steps_completed': {}, 'model_recommendations': None, 'report_generated': False,
    'report_files': None}
for _k, _v in _INITIAL_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Debug mode
AUTODS_DEBUG = os.environ.get('AUTODS_DEBUG', '0') == '1'

def _show_error(e, context=None):
    """Display error using ErrorHandler with optional debug details."""
    st.markdown(ErrorHandler.format_for_streamlit(e, context), unsafe_allow_html=True)
    if AUTODS_DEBUG:
        import traceback
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

def _get_memory_usage(df):
    bytes_used = df.memory_usage(deep=True).sum()
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024:
            return f"{bytes_used:.1f} {unit}"
        bytes_used /= 1024
    return f"{bytes_used:.1f} TB"

def _generate_plan(df):
    plans = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category', 'bool'] or df[col].nunique() <= 10:
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 10:
                plans.append({'task_name': f"Predict {col}", 'task_type': 'classification', 'target': col,
                    'reason': f"Column '{col}' has {unique_vals} unique values", 'difficulty': 'Easy' if unique_vals == 2 else 'Medium',
                    'recommended_models': ['Random Forest', 'XGBoost', 'Logistic Regression'],
                    'preprocessing': ['Handle missing values', 'Encode categoricals', 'Scale features']})
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() > 20:
            plans.append({'task_name': f"Predict {col}", 'task_type': 'regression', 'target': col,
                'reason': f"Column '{col}' is numeric with {df[col].nunique()} unique values", 'difficulty': 'Medium',
                'recommended_models': ['Random Forest', 'XGBoost', 'Linear Regression'],
                'preprocessing': ['Handle missing values', 'Scale features', 'Handle outliers']})
    plans.insert(0, {'task_name': "Exploratory Analysis", 'task_type': 'exploratory', 'target': None,
        'reason': "No target specified - discover patterns", 'difficulty': 'Easy',
        'recommended_models': [], 'preprocessing': ['Clean data', 'Handle missing values']})
    return plans[:5]



def render_step_1():
    st.markdown("<h1 style='font-size: 2rem; font-weight: 700;'>Upload Your Data</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6B6B6B; margin-bottom: 32px;'>Start by uploading a dataset or try one of our demos</p>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='upload-zone'>", unsafe_allow_html=True)
        st.markdown("<h3>📁 Upload File</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6B6B6B; font-size: 0.9rem;'>CSV, Excel, JSON, or Parquet</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'parquet'], label_visibility="collapsed")
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
                if len(df) > 100000:
                    st.warning(f"Large dataset ({len(df):,} rows). Sampling to 100,000 rows.")
                    df = df.sample(n=100000, random_state=42)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.current_step = 2
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load file: {str(e)}")
    with col2:
        st.markdown("<h3 style='margin-bottom: 16px;'>Try a Demo</h3>", unsafe_allow_html=True)
        demos = [
            {'name': 'customer_churn', 'icon': '👥', 'task': 'Classification', 'rows': '1,000'},
            {'name': 'house_prices', 'icon': '🏠', 'task': 'Regression', 'rows': '1,000'},
            {'name': 'iris_extended', 'icon': '🌸', 'task': 'Classification', 'rows': '150'},
        ]
        for demo in demos:
            st.markdown(f"<div class='demo-card'><div style='font-size: 2rem;'>{demo['icon']}</div><div style='font-weight: 600;'>{demo['name'].replace('_', ' ').title()}</div><div style='color: #6B6B6B; font-size: 0.8rem;'>{demo['task']} • {demo['rows']} rows</div></div>", unsafe_allow_html=True)
            if st.button(f"Load {demo['name'].replace('_', ' ').title()}", key=f"demo_{demo['name']}", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        df = DemoDatasetGenerator.load_demo_dataset(demo['name'])
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.current_step = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")


def render_step_2():
    df = st.session_state.df
    if not st.session_state.analysis_done:
        with st.spinner("🔍 Analyzing your dataset..."):
            try:
                profiler = StatisticalProfiler(df)
                profile = profiler.get_profile()
                insight_engine = InsightEngine(profiler)
                st.session_state.profile = profile
                st.session_state.insights = insight_engine
                st.session_state.analysis_done = True
                st.session_state.analysis_plan = _generate_plan(df)
                for plan in st.session_state.analysis_plan:
                    if plan['task_type'] == 'classification' and plan['target']:
                        st.session_state.target_column = plan['target']
                        break
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                return
    profile = st.session_state.profile
    overview = profile.get('overview', {})
    quality = profile.get('quality_score', {})
    st.markdown("<h1 style='font-size: 2rem; font-weight: 700;'>Dataset Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='success-banner'><span style='font-size: 1.5rem;'>✅</span><div><div style='font-weight: 600;'>Analysis Complete</div><div style='color: #6B6B6B; font-size: 0.9rem;'>{overview.get('n_rows', 0):,} rows × {overview.get('n_columns', 0)} columns • {_get_memory_usage(df)}</div></div></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 32px;'>Dataset Overview</h3>", unsafe_allow_html=True)
    cols = st.columns(5)
    metrics = [("Rows", f"{overview.get('n_rows', 0):,}"), ("Columns", str(overview.get('n_columns', 0))),
        ("Quality", f"{quality.get('overall', 0)}/100"), ("Missing", f"{overview.get('missing_pct', 0):.1f}%"),
        ("Duplicates", str(overview.get('duplicate_rows', 0)))]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(f"<div class='metric-card'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)
    with st.expander("📋 View First 5 Rows"):
        st.dataframe(df.head(), use_container_width=True)
    st.markdown("<h3 style='margin-top: 32px;'>Suggested Approaches</h3>", unsafe_allow_html=True)
    if st.session_state.analysis_plan:
        for plan in st.session_state.analysis_plan[:3]:
            diff_color = {'Easy': '#10B981', 'Medium': '#F59E0B', 'Hard': '#EF4444'}.get(plan['difficulty'], '#6B6B6B')
            models = ', '.join(plan['recommended_models']) if plan['recommended_models'] else 'N/A'
            st.markdown(f"""
            <div class='suggestion-card'>
                <h4>{plan['task_name']}</h4>
                <p style='color: #6B6B6B; font-size: 0.9rem;'>{plan['reason']}</p>
                <div style='display: flex; gap: 16px; margin: 12px 0;'>
                    <span style='background: {diff_color}20; color: {diff_color}; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 500;'>{plan['difficulty']}</span>
                    <span style='background: #F0EFEC; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem;'>{plan['task_type'].title()}</span>
                </div>
                <p style='font-size: 0.85rem;'><strong>Models:</strong> {models}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 32px;'>Select Target Column (Optional)</h3>", unsafe_allow_html=True)
    target_options = ["None (Exploratory)"] + list(df.columns)
    default_index = 0
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        default_index = target_options.index(st.session_state.target_column)
    selected_target = st.selectbox("Choose a column to predict", target_options, index=default_index, label_visibility="collapsed")
    st.session_state.target_column = None if selected_target == "None (Exploratory)" else selected_target
    st.markdown("<h3 style='margin-top: 32px;'>Choose Your Mode</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 16px; padding: 20px; margin-bottom: 12px;">
            <h4>⚡ Auto Mode</h4>
            <ul style="color: #6B6B6B; font-size: 0.9rem; margin: 12px 0; padding-left: 20px;">
                <li>Automatically detects data issues</li>
                <li>Applies best-practice fixes instantly</li>
                <li>Handles missing values & outliers</li>
                <li>Encodes categoricals & scales features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Auto Mode →", use_container_width=True, type="primary"):
            st.session_state.mode = 'auto'
            st.session_state.current_step = 3
            st.rerun()
    with col2:
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 16px; padding: 20px; margin-bottom: 12px;">
            <h4>🗺 Guide Mode</h4>
            <ul style="color: #6B6B6B; font-size: 0.9rem; margin: 12px 0; padding-left: 20px;">
                <li>Step-by-step preprocessing guide</li>
                <li>Learn what each transformation does</li>
                <li>Full control over each step</li>
                <li>Get copy-pasteable code snippets</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Guide Mode →", use_container_width=True):
            st.session_state.mode = 'guide'
            st.session_state.current_step = 3
            st.rerun()


def render_step_3():
    st.markdown("<h1 style='font-size: 2rem; font-weight: 700;'>Preprocessing</h1>", unsafe_allow_html=True)
    
    if st.session_state.mode == 'auto':
        # Show needs detection first
        if st.session_state.preprocessing_needs is None:
            with st.spinner("🔍 Analyzing preprocessing needs..."):
                try:
                    prep = PreprocessingOrchestrator(st.session_state.df, target_column=st.session_state.target_column)
                    needs = prep.get_needs_summary()
                    st.session_state.preprocessing_needs = needs
                    st.rerun()
                except Exception as e:
                    _show_error(e, "preprocessing needs analysis")
                    return
        
        needs = st.session_state.preprocessing_needs
        
        # Display needs as colored tags
        if not st.session_state.preprocessing_done:
            st.markdown("<h3 style='margin-top: 24px;'>Detected Preprocessing Needs</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div style='text-align: center;'><div style='font-size: 1.5rem; color: #DC2626;'>🔴 {needs.get('required', 0)}</div><div style='font-size: 0.8rem; color: #6B6B6B;'>Required</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='text-align: center;'><div style='font-size: 1.5rem; color: #D97706;'>🟡 {needs.get('recommended', 0)}</div><div style='font-size: 0.8rem; color: #6B6B6B;'>Recommended</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div style='text-align: center;'><div style='font-size: 1.5rem; color: #059669;'>🟢 {needs.get('optional', 0)}</div><div style='font-size: 0.8rem; color: #6B6B6B;'>Optional</div></div>", unsafe_allow_html=True)
            
            # Show specific needs as tags
            st.markdown("<div style='margin-top: 16px;'>", unsafe_allow_html=True)
            for need in needs.get('details', [])[:6]:
                priority = need.get('priority', 'optional')
                tag_class = 'required' if priority == 'required' else ('recommended' if priority == 'recommended' else 'optional')
                icon = '🔴' if priority == 'required' else ('🟡' if priority == 'recommended' else '🟢')
                st.markdown(f"<span class='need-tag {tag_class}'>{icon} {need.get('type', 'Unknown')}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("🔧 Run Auto Preprocessing →", use_container_width=True, type="primary"):
                with st.spinner("🔧 Cleaning and preparing your data..."):
                    try:
                        prep = PreprocessingOrchestrator(st.session_state.df, target_column=st.session_state.target_column)
                        result = prep.run_auto(mode='balanced', save=False)
                        st.session_state.processed_df = result['data']
                        st.session_state.preprocessing_done = True
                        st.rerun()
                    except Exception as e:
                        _show_error(e, "auto preprocessing")
        else:
            # Show results after preprocessing
            st.markdown("<div class='success-banner'><span style='font-size: 1.5rem;'>✅</span><div><div style='font-weight: 600;'>Preprocessing Complete</div><div style='color: #6B6B6B; font-size: 0.9rem;'>Data is ready for modeling</div></div></div>", unsafe_allow_html=True)
            
            # Show results summary
            original_shape = st.session_state.df.shape
            processed_shape = st.session_state.processed_df.shape if st.session_state.processed_df is not None else original_shape
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{original_shape[0]:,}</div><div class='metric-label'>Original Rows</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{processed_shape[1]}</div><div class='metric-label'>Features After</div></div>", unsafe_allow_html=True)
            
            # Auto-advance after 2 seconds or manual button
            st.markdown("<p style='color: #6B6B6B; margin-top: 20px;'>✅ Data Ready — Moving to modeling...</p>", unsafe_allow_html=True)
            if st.button("Continue to Modeling →", use_container_width=True, type="primary"):
                st.session_state.current_step = 4
                st.rerun()
    
    elif st.session_state.mode == 'guide':
        st.markdown("<h3 style='margin-top: 24px;'>🗺 Guide Mode</h3>", unsafe_allow_html=True)
        st.info("Guide mode provides step-by-step preprocessing with full control.")
        
        # Get needs for guide
        if st.session_state.preprocessing_needs is None:
            with st.spinner("🔍 Analyzing preprocessing needs..."):
                try:
                    prep = PreprocessingOrchestrator(st.session_state.df, target_column=st.session_state.target_column)
                    needs = prep.get_needs_summary()
                    st.session_state.preprocessing_needs = needs
                    st.rerun()
                except Exception as e:
                    _show_error(e, "preprocessing needs analysis")
                    return
        
        needs = st.session_state.preprocessing_needs
        details = needs.get('details', [])
        
        # Progress bar
        completed_steps = sum(1 for step, done in st.session_state.guide_steps_completed.items() if done)
        total_steps = len(details)
        progress = completed_steps / total_steps if total_steps > 0 else 0
        st.progress(progress, text=f"Progress: {completed_steps}/{total_steps} steps completed")
        
        # Show prioritized sections
        priority_order = ['required', 'recommended', 'optional']
        priority_titles = {'required': '🔴 Required Steps', 'recommended': '🟡 Recommended Steps', 'optional': '🟢 Optional Steps'}
        
        for priority in priority_order:
            priority_needs = [n for n in details if n.get('priority') == priority]
            if priority_needs:
                st.markdown(f"<h4 style='margin-top: 20px;'>{priority_titles[priority]}</h4>", unsafe_allow_html=True)
                for need in priority_needs:
                    need_id = f"{need.get('type', '')}_{need.get('column', '')}"
                    is_done = st.session_state.guide_steps_completed.get(need_id, False)
                    
                    with st.container():
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            if st.checkbox("", value=is_done, key=f"check_{need_id}"):
                                st.session_state.guide_steps_completed[need_id] = True
                                st.rerun()
                        with col2:
                            st.markdown(f"<strong>{need.get('type', 'Unknown')}</strong>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #6B6B6B; font-size: 0.85rem;'>{need.get('description', 'No description')}</p>", unsafe_allow_html=True)
                            with st.expander("📋 View Code Snippet"):
                                code = need.get('code_snippet', '# No code snippet available')
                                st.code(code, language='python')
        
        # Show continue button when required steps are done
        required_count = sum(1 for n in details if n.get('priority') == 'required')
        required_done = sum(1 for n in details if n.get('priority') == 'required' and st.session_state.guide_steps_completed.get(f"{n.get('type', '')}_{n.get('column', '')}", False))
        
        if required_done >= required_count and required_count > 0:
            if st.button("Continue to Model Selection →", use_container_width=True, type="primary"):
                st.session_state.current_step = 4
                st.rerun()
        else:
            st.markdown(f"<p style='color: #6B6B6B;'>Complete {required_count - required_done} more required step(s) to continue.</p>", unsafe_allow_html=True)


def render_step_4():
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    st.markdown("<h1 style='font-size: 2rem; font-weight: 700;'>Model Training</h1>", unsafe_allow_html=True)
    
    # Target column selector (pre-filled from step 2)
    target_options = ["None (Exploratory)"] + [c for c in df.columns if df[c].dtype != 'object'][:20]
    default_index = 0
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        try:
            default_index = target_options.index(st.session_state.target_column)
        except ValueError:
            default_index = 0
    
    selected_target = st.selectbox("🎯 Target Column", target_options, index=default_index)
    target = None if selected_target == "None (Exploratory)" else selected_target
    st.session_state.target_column = target
    
    if not target:
        st.info("No target column selected. Skip to Reports to export your cleaned data.")
        if st.button("Skip to Reports →", use_container_width=True, type="primary"):
            st.session_state.current_step = 5
            st.rerun()
        return
    
    # Task type detection
    recommender = ModelRecommender(df, target)
    task_info = recommender.detect_task_type()
    task_type = task_info.get('type', 'unknown')
    task_desc = {'binary_classification': 'Binary Classification', 'multi_class_classification': 'Multi-Class Classification', 'regression': 'Regression'}.get(task_type, 'Unknown')
    task_emoji = {'binary_classification': '🔴', 'multi_class_classification': '🔵', 'regression': '📊'}.get(task_type, '❓')
    
    st.markdown(f"<div class='card'><div style='font-size: 1.1rem;'><strong>{task_emoji} This looks like {task_desc}</strong></div><div style='color: #6B6B6B; font-size: 0.9rem;'>Target has {df[target].nunique()} unique values • {task_info.get('reasoning', 'N/A')}</div></div>", unsafe_allow_html=True)
    
    # Get or show recommendations
    if st.session_state.model_recommendations is None:
        with st.spinner("🔍 Analyzing and recommending models..."):
            try:
                recs = recommender.recommend()
                st.session_state.model_recommendations = recs
                st.rerun()
            except Exception as e:
                _show_error(e, "model recommendation")
                return
    
    recommendations = st.session_state.model_recommendations
    top_3 = recommendations[:3] if recommendations else []
    
    # Show top 3 model recommendation cards
    st.markdown("<h3 style='margin-top: 24px;'>🤖 Top Model Recommendations</h3>", unsafe_allow_html=True)
    
    for i, rec in enumerate(top_3):
        confidence = rec.get('confidence', 0)
        is_top = i == 0
        card_class = "model-card recommended" if is_top else "model-card"
        
        with st.container():
            st.markdown(f"""
            <div class='{card_class}'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='font-size: 1.1rem; font-weight: 600;'>#{i+1} {rec.get('model', 'Unknown')}</div>
                    <div style='background: {"#10B98120" if is_top else "#F0EFEC"}; color: {"#059669" if is_top else "#6B6B6B"}; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem;'>{confidence:.0%} Match</div>
                </div>
                <p style='color: #6B6B6B; font-size: 0.9rem; margin: 8px 0;'>{rec.get('reasoning', 'No reasoning provided')}</p>
                <div style='display: flex; gap: 16px; font-size: 0.85rem; color: #6B6B6B;'>
                    <span>⏱️ ~{rec.get('estimated_time', 'N/A')}s training</span>
                    <span>📈 Est. {rec.get('expected_performance', 'N/A')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("ℹ️ Learn More"):
                st.markdown(f"<strong>Pros:</strong> {', '.join(rec.get('pros', ['N/A']))}", unsafe_allow_html=True)
                st.markdown(f"<strong>Cons:</strong> {', '.join(rec.get('cons', ['N/A']))}", unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<h3 style='margin-top: 24px;'>🚀 Next Steps</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 See Recommendations Only", use_container_width=True):
            st.info("Showing recommendations above. Ready to train when you are!")
    
    with col2:
        if not st.session_state.models_trained:
            if st.button("🚀 Train Baseline Models", use_container_width=True, type="primary"):
                with st.spinner("🤖 Training baseline models..."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        model_orch = ModelOrchestrator(df, target_column=target)
                        
                        # Simulate progress for UX
                        status_text.text("Initializing models...")
                        progress_bar.progress(10)
                        
                        results = model_orch.train_baseline(save=False)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training complete!")
                        
                        st.session_state.model_results = results
                        st.session_state.models_trained = True
                        st.rerun()
                    except Exception as e:
                        _show_error(e, "model training")
    
    # Show training results
    if st.session_state.models_trained and st.session_state.model_results:
        results = st.session_state.model_results
        st.markdown("<div class='success-banner'><span style='font-size: 1.5rem;'>✅</span><div><div style='font-weight: 600;'>Training Complete</div><div style='color: #6B6B6B; font-size: 0.9rem;'>{0} models trained</div></div></div>".format(results.get('models_trained', 0)), unsafe_allow_html=True)
        
        if 'results' in results and results['results']:
            st.markdown("<h3 style='margin-top: 24px;'>📊 Model Results</h3>", unsafe_allow_html=True)
            
            # Sortable results table
            results_df = pd.DataFrame([
                {'Model': r['model_name'], 'Score': r['main_metric_value'],
                 'CV Mean': r.get('cv_mean', '-'), 'Time (s)': r['training_time']}
                for r in results['results']
            ])
            results_df = results_df.sort_values('Score', ascending=False)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Best model highlight
            if results.get('best_model'):
                bm = results['best_model']
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%); border: 2px solid #10B981; border-radius: 16px; padding: 24px; margin-top: 24px;'>
                    <div style='font-size: 1.5rem; margin-bottom: 8px;'>🏆</div>
                    <div style='font-weight: 600; font-size: 1.1rem;'>Best Model: {bm['model_name']}</div>
                    <div style='color: #6B6B6B;'>{bm['main_metric']}: <strong style='color: #059669; font-size: 1.25rem;'>{bm['main_metric_value']:.4f}</strong></div>
                    <div style='color: #9CA3AF; font-size: 0.85rem; margin-top: 8px;'>Trained in {bm.get('training_time', 0):.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature importance chart (if available)
                if 'feature_importance' in bm and bm['feature_importance']:
                    st.markdown("<h4 style='margin-top: 20px;'>📈 Feature Importance (Top 10)</h4>", unsafe_allow_html=True)
                    import plotly.express as px
                    fi_df = pd.DataFrame(bm['feature_importance'])
                    fi_df = fi_df.head(10)
                    fig = px.bar(fi_df, x='importance', y='feature', orientation='h',
                                title="Feature Importance", color_discrete_sequence=['#10B981'])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Continue to Reports →", use_container_width=True, type="primary"):
            st.session_state.current_step = 5
            st.rerun()


def render_step_5():
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    st.markdown("<h1 style='font-size: 2rem; font-weight: 700;'>Reports & Export</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6B6B6B;'>Download your analysis results and generate comprehensive reports</p>", unsafe_allow_html=True)
    
    # Checklist of completed steps
    st.markdown("<h3 style='margin-top: 24px;'>✅ Analysis Checklist</h3>", unsafe_allow_html=True)
    checklist_items = [
        ("Data Uploaded", st.session_state.df is not None),
        ("Analysis Complete", st.session_state.analysis_done),
        ("Preprocessing Done", st.session_state.preprocessing_done),
        ("Models Trained", st.session_state.models_trained),
        ("Report Generated", st.session_state.report_generated)
    ]
    
    cols = st.columns(5)
    for col, (label, done) in zip(cols, checklist_items):
        icon = "✅" if done else "⬜"
        color = "#10B981" if done else "#9CA3AF"
        col.markdown(f"<div style='text-align: center;'><div style='font-size: 1.5rem;'>{icon}</div><div style='font-size: 0.75rem; color: {color};'>{label}</div></div>", unsafe_allow_html=True)
    
    # Generate Full Report Section
    st.markdown("<h3 style='margin-top: 32px;'>📄 Generate Full Report</h3>", unsafe_allow_html=True)
    
    if not st.session_state.report_generated:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6B6B6B;'>Generate a comprehensive report including all analysis steps, visualizations, and model results.</p>", unsafe_allow_html=True)
        
        if st.button("📊 Generate Full Report", use_container_width=True, type="primary"):
            with st.spinner("Generating comprehensive report... This may take a minute."):
                try:
                    results = run_complete_analysis(
                        df, 
                        target=st.session_state.target_column, 
                        output_dir="reports", 
                        use_llm=False
                    )
                    st.session_state.report_generated = True
                    st.session_state.report_files = results.get('exported_files', {})
                    st.success(f"✅ Report generated successfully in {results['output_directory']}")
                    st.rerun()
                except Exception as e:
                    _show_error(e, "report generation")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='success-banner'><span style='font-size: 1.5rem;'>✅</span><div><div style='font-weight: 600;'>Report Generated</div><div style='color: #6B6B6B; font-size: 0.9rem;'>All export formats are ready for download</div></div></div>", unsafe_allow_html=True)
    
    # Export Options
    st.markdown("<h3 style='margin-top: 32px;'>💾 Export Options</h3>", unsafe_allow_html=True)
    
    # Helper to get file size
    def get_file_size(data, format_type):
        if format_type == 'csv':
            size = len(data.to_csv(index=False).encode('utf-8'))
        elif format_type == 'json':
            size = len(data.to_json(orient='records').encode('utf-8'))
        else:
            size = 0
        
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"
    
    # CSV Export
    csv_data = df.to_csv(index=False).encode('utf-8')
    csv_size = get_file_size(df, 'csv')
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<div class='export-option'><div>📊 <strong>CSV</strong> — Processed data in CSV format</div><div style='color: #9CA3AF; font-size: 0.85rem;'>{csv_size}</div></div>", unsafe_allow_html=True)
    with col2:
        st.download_button("Download", csv_data, file_name="processed_data.csv", mime="text/csv", use_container_width=True)
    
    # JSON Export
    json_data = df.to_json(orient='records')
    json_size = get_file_size(df, 'json')
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<div class='export-option'><div>🗂 <strong>JSON</strong> — Data as JSON records</div><div style='color: #9CA3AF; font-size: 0.85rem;'>{json_size}</div></div>", unsafe_allow_html=True)
    with col2:
        st.download_button("Download", json_data, file_name="data.json", mime="application/json", use_container_width=True)
    
    # Generated reports (if available)
    if st.session_state.report_generated and st.session_state.report_files:
        st.markdown("<h4 style='margin-top: 20px;'>📑 Generated Reports</h4>", unsafe_allow_html=True)
        
        for fmt, path in st.session_state.report_files.items():
            if Path(path).exists():
                size = Path(path).stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    icon = {"html": "📄", "notebook": "📓", "markdown": "📝", "json": "🗂"}.get(fmt.lower(), "📄")
                    st.markdown(f"<div class='export-option'><div>{icon} <strong>{fmt.upper()}</strong> — Full analysis report</div><div style='color: #9CA3AF; font-size: 0.85rem;'>{size_str}</div></div>", unsafe_allow_html=True)
                with col2:
                    with open(path, 'rb') as f:
                        mime_types = {"html": "text/html", "notebook": "application/json", "markdown": "text/markdown", "json": "application/json"}
                        st.download_button("Download", f.read(), file_name=Path(path).name, mime=mime_types.get(fmt.lower(), "application/octet-stream"), use_container_width=True)
    
    # Start New Analysis button
    st.markdown("<hr style='margin: 32px 0; border: none; border-top: 1px solid #E8E6E1;'>", unsafe_allow_html=True)
    if st.button("🔄 Start New Analysis", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main():
    st.markdown("<div style='text-align: center; padding: 20px 0 30px 0;'><h1 style='font-size: 2.5rem; font-weight: 700;'>⚡ AutoDS</h1><p style='color: #6B6B6B; font-size: 1.1rem;'>Automated Data Science Platform</p></div>", unsafe_allow_html=True)
    
    # Define steps
    steps = [
        {"label": "1. Upload", "step_num": 1, "icon": "📤"},
        {"label": "2. Analyze", "step_num": 2, "icon": "🔍"},
        {"label": "3. Preprocess", "step_num": 3, "icon": "⚙️"},
        {"label": "4. Model", "step_num": 4, "icon": "🤖"},
        {"label": "5. Reports", "step_num": 5, "icon": "📊"}
    ]
    
    current = st.session_state.current_step
    
    # Build HTML progress bar
    progress_html = "<div class='progress-bar-container'>"
    progress_html += "<div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;'>"
    
    for i, step in enumerate(steps):
        step_num = step["step_num"]
        label = step["label"]
        icon = step["icon"]
        
        if step_num < current:
            # Completed step
            status_class = "completed"
            indicator = "✅"
            arrow = "→" if i < len(steps) - 1 else ""
        elif step_num == current:
            # Active step
            status_class = "active"
            indicator = icon
            arrow = "→" if i < len(steps) - 1 else ""
        else:
            # Locked step
            status_class = "locked"
            indicator = "🔒"
            arrow = "→" if i < len(steps) - 1 else ""
        
        progress_html += f"<div class='progress-step {status_class}'>{indicator} {label}</div>"
        if arrow:
            progress_html += f"<div style='color: #D1D5DB;'>→</div>"
    
    progress_html += "</div></div>"
    st.markdown(progress_html, unsafe_allow_html=True)
    
    # Create tabs with labels
    tab_labels = [f"{s['icon']} {s['label']}" for s in steps]
    tabs = st.tabs(tab_labels)
    
    # Render appropriate content based on current step
    # Locked tabs show a message to complete previous steps
    with tabs[0]:
        render_step_1()
    
    with tabs[1]:
        if current >= 2:
            render_step_2()
        else:
            st.markdown("<div style='text-align: center; padding: 40px;'><div style='font-size: 3rem; margin-bottom: 16px;'>🔒</div><h3>Step Locked</h3><p style='color: #6B6B6B;'>Complete Step 1 (Upload) to unlock this step.</p></div>", unsafe_allow_html=True)
            if st.button("Go to Upload →", key="goto_upload"):
                st.session_state.current_step = 1
                st.rerun()
    
    with tabs[2]:
        if current >= 3:
            render_step_3()
        else:
            st.markdown("<div style='text-align: center; padding: 40px;'><div style='font-size: 3rem; margin-bottom: 16px;'>🔒</div><h3>Step Locked</h3><p style='color: #6B6B6B;'>Complete Step 2 (Analyze) to unlock this step.</p></div>", unsafe_allow_html=True)
            if st.button("Go to Analyze →", key="goto_analyze"):
                st.session_state.current_step = 2
                st.rerun()
    
    with tabs[3]:
        if current >= 4:
            render_step_4()
        else:
            st.markdown("<div style='text-align: center; padding: 40px;'><div style='font-size: 3rem; margin-bottom: 16px;'>🔒</div><h3>Step Locked</h3><p style='color: #6B6B6B;'>Complete Step 3 (Preprocess) to unlock this step.</p></div>", unsafe_allow_html=True)
            if st.button("Go to Preprocess →", key="goto_preprocess"):
                st.session_state.current_step = 3
                st.rerun()
    
    with tabs[4]:
        if current >= 5:
            render_step_5()
        else:
            st.markdown("<div style='text-align: center; padding: 40px;'><div style='font-size: 3rem; margin-bottom: 16px;'>🔒</div><h3>Step Locked</h3><p style='color: #6B6B6B;'>Complete Step 4 (Model) to unlock this step.</p></div>", unsafe_allow_html=True)
            if st.button("Go to Model →", key="goto_model"):
                st.session_state.current_step = 4
                st.rerun()

if __name__ == "__main__":
    main()
