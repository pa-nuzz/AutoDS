"""AutoDS Streamlit UI - Clean Minimal Design."""
import os
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent / "src"))

from autods.analysis.profiler import StatisticalProfiler
from autods.analysis.insights import InsightEngine
from autods.preprocessing.orchestrator import PreprocessingOrchestrator
from autods.modeling.orchestrator import ModelOrchestrator
from autods.modeling.recommender import ModelRecommender
from autods.modeling.task_detector import TaskDetector
from autods.reports.pipeline import run_complete_analysis
from autods.utils.demo_data import DemoDatasetGenerator
from autods.utils.error_handler import ErrorHandler

st.set_page_config(page_title="AutoDS", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* Clean dark theme */
.stApp {
    background-color: #0a0a0a !important;
}

/* White text */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #ffffff !important;
}

/* Simple inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea, 
.stSelectbox > div > div, .stFileUploader > div > div {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

/* Simple buttons */
.stButton > button {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

.stButton > button[kind="primary"] {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
}

/* Dataframes */
.stDataFrame, .stDataFrame * {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"] {
    color: #888888 !important;
}

[data-testid="stMetricValue"] {
    color: #3b82f6 !important;
}

/* Hide branding */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

_INITIAL_STATE = {
    'df': None, 'processed_df': None, 'profile': None, 'insights': None,
    'model_results': None, 'target_column': None, 'mode': None, 'data_loaded': False,
    'analysis_done': False, 'preprocessing_done': False, 'models_trained': False,
    'current_step': 1, 'analysis_plan': None, 'preprocessing_needs': None,
    'guide_steps_completed': {}, 'model_recommendations': None, 
    'report_generated': False, 'report_files': None, 'training_in_progress': False
}
for _k, _v in _INITIAL_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def _show_error(e, context=None):
    st.error(f"Error in {context}: {str(e)}")

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
                plans.append({
                    'task_name': f"Predict {col}", 'task_type': 'classification', 'target': col,
                    'reason': f"Column '{col}' has {unique_vals} unique values",
                    'difficulty': 'Easy' if unique_vals == 2 else 'Medium',
                    'recommended_models': ['Random Forest', 'XGBoost', 'Logistic Regression']
                })
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() > 20:
            plans.append({
                'task_name': f"Predict {col}", 'task_type': 'regression', 'target': col,
                'reason': f"Column '{col}' is numeric with {df[col].nunique()} unique values",
                'difficulty': 'Medium',
                'recommended_models': ['Random Forest', 'XGBoost', 'Linear Regression']
            })
    plans.insert(0, {
        'task_name': "Exploratory Analysis", 'task_type': 'exploratory', 'target': None,
        'reason': "No target specified - discover patterns", 'difficulty': 'Easy',
        'recommended_models': []
    })
    return plans[:5]

def render_step_1():
    st.header("Upload Your Data")
    st.caption("CSV, Excel, JSON, or Parquet — up to 100MB")
    
    uploaded = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'parquet'])
    
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
            
            st.success(f"Loaded {uploaded.name}: {len(df):,} rows × {len(df.columns)} columns")
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.current_step = 2
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load file: {str(e)}")
    
    st.divider()
    st.subheader("Try a Demo")
    
    demos = [
        ('Customer Churn', 'customer_churn', 'Classification', '1k rows'),
        ('House Prices', 'house_prices', 'Regression', '1k rows'),
        ('Iris Extended', 'iris_extended', 'Multi-class', '150 rows'),
    ]
    
    for name, key, task, rows in demos:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{name}**  \n{task} • {rows}")
        with col2:
            if st.button("Load", key=f"demo_{key}"):
                with st.spinner("Loading..."):
                    try:
                        df = DemoDatasetGenerator.load_demo_dataset(key)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.current_step = 2
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {str(e)}")

def render_step_2():
    df = st.session_state.df
    
    if not st.session_state.analysis_done:
        with st.spinner("Analyzing your dataset..."):
            try:
                profiler = StatisticalProfiler(df)
                profile = profiler.get_profile()
                st.session_state.profile = profile
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
    quality_score = quality.get('overall', 0)
    
    st.header("Dataset Overview")
    st.caption(f"{overview.get('n_rows', 0):,} rows · {overview.get('n_columns', 0)} columns · {_get_memory_usage(df)} · Quality {quality_score}/100")
    
    cols = st.columns(5)
    metrics = [
        ("Rows", f"{overview.get('n_rows', 0):,}"),
        ("Columns", str(overview.get('n_columns', 0))),
        ("Quality", f"{quality_score}"),
        ("Missing %", f"{overview.get('missing_pct', 0):.1f}%"),
        ("Duplicates", str(overview.get('duplicate_rows', 0))),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    
    with st.expander("View first 5 rows"):
        st.dataframe(df.head(), use_container_width=True)
    
    # Data Visualizations
    st.subheader("Data Exploration")
    viz_tabs = st.tabs(["Distributions", "Correlations", "Target Analysis"])
    
    with viz_tabs[0]:
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # First 4 numeric
        if len(numeric_cols) > 0:
            fig = make_subplots(rows=2, cols=2, subplot_titles=numeric_cols[:4])
            for i, col in enumerate(numeric_cols[:4]):
                row = i // 2 + 1
                col_pos = i % 2 + 1
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, marker_color='#00d4ff', opacity=0.8),
                    row=row, col=col_pos
                )
            fig.update_layout(
                showlegend=False, 
                height=500,
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#141414',
                font_color='#e8e8e8'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns for distribution plots")
    
    with viz_tabs[1]:
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr, 
                color_continuous_scale='teal',
                aspect='auto',
                title="Feature Correlations"
            )
            fig.update_layout(
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#141414',
                font_color='#e8e8e8',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation")
    
    with viz_tabs[2]:
        # Target analysis if target selected
        if st.session_state.target_column and st.session_state.target_column in df.columns:
            target_col = st.session_state.target_column
            if df[target_col].dtype == 'object' or df[target_col].nunique() <= 10:
                # Categorical target
                value_counts = df[target_col].value_counts()
                fig = px.pie(
                    values=value_counts.values, 
                    names=value_counts.index,
                    title=f"Target Distribution: {target_col}",
                    color_discrete_sequence=['#00d4ff', '#0099cc', '#006699', '#003366']
                )
            else:
                # Numeric target
                fig = px.histogram(
                    df, x=target_col, 
                    title=f"Target Distribution: {target_col}",
                    color_discrete_sequence=['#00d4ff']
                )
            fig.update_layout(
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#141414',
                font_color='#e8e8e8',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a target column to see target analysis")
    
    st.subheader("Suggested Approaches")
    if st.session_state.analysis_plan:
        for plan in st.session_state.analysis_plan[:3]:
            with st.container():
                st.markdown(f"**{plan['task_name']}** — {plan['task_type'].title()} — {plan['difficulty']}")
                st.caption(plan['reason'])
    
    st.subheader("Select Target Column (optional)")
    target_options = ["None — Explore only"] + list(df.columns)
    default = 0
    if st.session_state.target_column and st.session_state.target_column in df.columns:
        default = target_options.index(st.session_state.target_column)
    selected = st.selectbox("Target", target_options, index=default)
    st.session_state.target_column = None if selected == "None — Explore only" else selected
    
    st.divider()
    st.subheader("Choose Mode")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Auto Mode")
        st.caption("Automatically detects and fixes data issues")
        if st.button("Continue with Auto", type="primary", use_container_width=True):
            st.session_state.mode = 'auto'
            st.session_state.current_step = 3
            st.rerun()
    with col2:
        st.markdown("### Guide Mode")
        st.caption("Step-by-step preprocessing with explanations")
        if st.button("Continue with Guide", use_container_width=True):
            st.session_state.mode = 'guide'
            st.session_state.current_step = 3
            st.rerun()

def render_step_3():
    st.header("Preprocessing")
    
    # Show data quality stats even when clean
    df = st.session_state.df
    st.subheader("Data Quality Overview")
    
    quality_cols = st.columns(4)
    with quality_cols[0]:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with quality_cols[1]:
        st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
    with quality_cols[2]:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    with quality_cols[3]:
        cat_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Categorical", cat_cols)
    
    if st.session_state.mode == 'auto':
        if st.session_state.preprocessing_needs is None:
            with st.spinner("Analyzing preprocessing needs..."):
                try:
                    prep = PreprocessingOrchestrator(st.session_state.df, target_column=st.session_state.target_column)
                    needs = prep.get_needs_summary()
                    st.session_state.preprocessing_needs = needs
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {str(e)}")
                    return
        
        needs = st.session_state.preprocessing_needs
        
        if not st.session_state.preprocessing_done:
            st.subheader("Detected Issues")
            
            # Skeleton loading state
            if needs.get('required', 0) == 0 and needs.get('recommended', 0) == 0 and needs.get('optional', 0) == 0:
                st.info("✓ Data looks clean! No preprocessing issues detected.")
                st.markdown("""
                **Detailed Analysis:**
                - ✓ No missing values found
                - ✓ No duplicate rows detected  
                - ✓ No outliers detected
                - ✓ Data types are consistent
                """)
            
            cols = st.columns(3)
            issues = [
                ("Required", needs.get('required', 0)),
                ("Recommended", needs.get('recommended', 0)),
                ("Optional", needs.get('optional', 0)),
            ]
            for col, (label, count) in zip(cols, issues):
                col.metric(label, count)
            
            st.caption("Issues found:")
            for need in needs.get('details', [])[:6]:
                st.markdown(f"- {need.get('type', 'Unknown')}")
            
            # User controls
            control_cols = st.columns([2, 1, 1])
            with control_cols[1]:
                if st.button("Skip Preprocessing", use_container_width=True):
                    st.session_state.processed_df = st.session_state.df
                    st.session_state.preprocessing_done = True
                    st.rerun()
            with control_cols[2]:
                if st.button("Run Auto Preprocessing", type="primary", use_container_width=True):
                    with st.spinner("Cleaning and preparing your data..."):
                        try:
                            prep = PreprocessingOrchestrator(st.session_state.df, target_column=st.session_state.target_column)
                            result = prep.run_auto(mode='balanced', save=False)
                            st.session_state.processed_df = result['data']
                            st.session_state.preprocessing_done = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Preprocessing failed: {str(e)}")
                            st.button("🔄 Retry", key="retry_prep", on_click=lambda: None)
        else:
            st.success("Preprocessing Complete!")
            orig = st.session_state.df.shape
            proc = st.session_state.processed_df.shape
            st.caption(f"{orig[0]:,} rows → {proc[0]:,} rows · {proc[1]} features")
            
            col1, col2 = st.columns(2)
            col1.metric("Original Rows", orig[0])
            col2.metric("Features", proc[1])
            
            if st.button("Continue to Modeling", type="primary"):
                st.session_state.current_step = 4
                st.rerun()
    else:
        st.subheader("Preprocessing Guide")
        
        if st.session_state.preprocessing_needs is None:
            with st.spinner("Analyzing preprocessing needs..."):
                try:
                    prep = PreprocessingOrchestrator(st.session_state.df, target_column=st.session_state.target_column)
                    needs = prep.get_needs_summary()
                    st.session_state.preprocessing_needs = needs
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {str(e)}")
                    return
        
        needs = st.session_state.preprocessing_needs
        details = needs.get('details', [])
        
        completed = sum(1 for step, done in st.session_state.guide_steps_completed.items() if done)
        total = len(details)
        st.progress(completed / total if total > 0 else 0)
        st.caption(f"{completed}/{total} steps completed")
        
        for need in details[:5]:
            need_id = f"{need.get('type', '')}_{need.get('column', '')}"
            is_done = st.session_state.guide_steps_completed.get(need_id, False)
            
            checked = st.checkbox(f"{need.get('type', 'Unknown')}: {need.get('description', 'No description')}", value=is_done, key=f"check_{need_id}")
            if checked != is_done:
                st.session_state.guide_steps_completed[need_id] = checked
                st.rerun()
            
            with st.expander("View code"):
                st.code(need.get('code_snippet', '# No code available'), language='python')
        
        required_count = sum(1 for n in details if n.get('priority') == 'required')
        required_done = sum(1 for n in details if n.get('priority') == 'required' and st.session_state.guide_steps_completed.get(f"{n.get('type', '')}_{n.get('column', '')}", False))
        
        if required_done >= required_count and required_count > 0:
            if st.button("Continue to Modeling", type="primary"):
                with st.spinner("Applying preprocessing..."):
                    try:
                        prep = PreprocessingOrchestrator(st.session_state.df, target_column=st.session_state.target_column)
                        result = prep.run_auto(mode='balanced', save=False)
                        st.session_state.processed_df = result['data']
                        st.session_state.preprocessing_done = True
                        st.session_state.current_step = 4
                        st.rerun()
                    except Exception as e:
                        st.error(f"Preprocessing failed: {str(e)}")
        else:
            st.caption(f"{required_count - required_done} required steps remaining")

def render_step_4():
    st.header("Model Training")
    
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    col1, col2 = st.columns([1, 3])
    col1.markdown("**Target column**")
    target_options = ["None (Exploratory)"] + [c for c in df.columns if df[c].dtype != 'object'][:20]
    default = 0
    if st.session_state.target_column:
        if st.session_state.target_column in target_options:
            default = target_options.index(st.session_state.target_column)
        else:
            # Target was categorical - add it to options
            target_options.insert(1, st.session_state.target_column)
            default = 1
    
    selected = col2.selectbox("Target", target_options, index=default, label_visibility="collapsed")
    target = None if selected == "None (Exploratory)" else selected
    st.session_state.target_column = target
    
    if not target:
        st.info("No target selected. Skip to Reports to export your data.")
        if st.button("Skip to Reports"):
            st.session_state.current_step = 5
            st.rerun()
        return
    
    task_detector = TaskDetector(df, target)
    task_info = task_detector.get_task_info()
    task_type = task_info.get('type', 'unknown')
    task_desc = {'binary_classification': 'Binary', 'multiclass_classification': 'Multi-Class', 'regression': 'Regression'}.get(task_type, 'Unknown')
    
    st.caption(f"Detected: **{task_desc}** — {task_info.get('reasoning', 'N/A')}")
    
    if st.session_state.model_recommendations is None:
        with st.spinner("Analyzing and recommending models..."):
            try:
                dataset_info = {'n_samples': len(df), 'has_missing_values': df.isnull().any().any()}
                recommender = ModelRecommender(task_detector, dataset_info)
                recs = recommender.get_recommendations(top_k=3)
                st.session_state.model_recommendations = recs
                st.rerun()
            except Exception as e:
                st.error(f"Recommendation failed: {str(e)}")
                if st.button("🔄 Retry Analysis"):
                    st.rerun()
                return
    
    recommendations = st.session_state.model_recommendations
    
    st.subheader("Recommended Models")
    for i, rec in enumerate(recommendations[:3]):
        with st.container():
            cols = st.columns([1, 3, 1])
            cols[0].markdown(f"**#{i+1}**")
            cols[1].markdown(f"**{rec.get('model_name', 'Unknown')}**")
            confidence = rec.get('confidence_score', 0)
            cols[2].markdown(f"{confidence:.0f}% match")
            st.caption(rec.get('reasoning', 'No reasoning'))
            with st.expander("Details"):
                model_info = rec.get('model_info', {})
                pros = model_info.get('pros', [])
                cons = model_info.get('cons', [])
                if pros:
                    st.markdown(f"**Pros:** {', '.join(pros)}")
                if cons:
                    st.markdown(f"**Cons:** {', '.join(cons)}")
    
    if not st.session_state.models_trained and not st.session_state.training_in_progress:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Recommendations Only"):
                st.info("Showing recommendations above.")
        with col2:
            if st.button("Train Models", type="primary"):
                st.session_state.training_in_progress = True
                st.rerun()
    
    if st.session_state.training_in_progress and not st.session_state.models_trained:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            model_orch = ModelOrchestrator(df, target_column=target)
            
            messages = [
                "Initializing models...",
                "Preparing data splits...",
                "Training Random Forest...",
                "Training XGBoost...",
                "Cross-validating...",
                "Finalizing results..."
            ]
            
            for i, msg in enumerate(messages):
                progress_bar.progress((i + 1) / len(messages))
                status_text.text(msg)
                import time
                time.sleep(0.3)
            
            results = model_orch.train_baseline(save=False)
            
            progress_bar.progress(1.0)
            status_text.text("Training complete!")
            
            st.session_state.model_results = results
            st.session_state.models_trained = True
            st.session_state.training_in_progress = False
            st.rerun()
        except Exception as e:
            st.session_state.training_in_progress = False
            st.error(f"Training failed: {str(e)}")
            
            # Error recovery options
            error_cols = st.columns(3)
            with error_cols[0]:
                if st.button("🔄 Retry Training", key="retry_training"):
                    st.session_state.training_in_progress = True
                    st.rerun()
            with error_cols[1]:
                if st.button("⏭️ Skip to Reports", key="skip_to_reports"):
                    st.session_state.current_step = 5
                    st.rerun()
            with error_cols[2]:
                if st.button("🔧 Change Target", key="change_target"):
                    st.session_state.models_trained = False
                    st.session_state.model_results = None
                    st.rerun()
    
    if st.session_state.models_trained and st.session_state.model_results:
        results = st.session_state.model_results
        st.success(f"Training complete! {results.get('models_trained', 0)} models trained.")
        
        if 'results' in results and results['results']:
            results_df = pd.DataFrame([
                {'Model': r['model_name'], 'Score': r['main_metric_value'],
                 'CV Mean': r.get('cv_mean', '-'), 'Time (s)': r['training_time']}
                for r in results['results']
            ])
            results_df = results_df.sort_values('Score', ascending=False)
            
            # Model comparison chart
            fig = px.bar(
                results_df, 
                x='Model', 
                y='Score',
                title='Model Performance Comparison',
                color='Score',
                color_continuous_scale='teal',
                text='Score'
            )
            fig.update_layout(
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#141414',
                font_color='#e8e8e8',
                height=400,
                xaxis_title="",
                yaxis_title="Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        if results.get('best_model'):
            bm = results['best_model']
            st.subheader("Best Model")
            cols = st.columns(3)
            cols[0].metric("Model", bm['model_name'])
            cols[1].metric("Metric", bm['main_metric'])
            cols[2].metric("Score", f"{bm['main_metric_value']:.4f}")
            st.caption(f"Trained in {bm.get('training_time', 0):.2f}s")
            
            # Feature importance if available
            if 'feature_importance' in bm and bm['feature_importance']:
                st.subheader("Feature Importance")
                fi_df = pd.DataFrame(bm['feature_importance']).head(10)
                fig = px.bar(
                    fi_df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title='Top 10 Important Features',
                    color_discrete_sequence=['#00d4ff']
                )
                fig.update_layout(
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#141414',
                    font_color='#e8e8e8',
                    height=400,
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Continue to Reports", type="primary"):
            st.session_state.current_step = 5
            st.rerun()

def render_step_5():
    st.header("Export & Reports")
    
    df = st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.df
    
    st.subheader("Progress")
    cols = st.columns(5)
    items = [
        ("Upload", st.session_state.df is not None),
        ("Analysis", st.session_state.analysis_done),
        ("Preprocess", st.session_state.preprocessing_done),
        ("Models", st.session_state.models_trained),
        ("Report", st.session_state.report_generated),
    ]
    for col, (label, done) in zip(cols, items):
        col.caption(label)
        if done:
            col.markdown(":heavy_check_mark:")
        else:
            col.markdown(":heavy_multiplication_x:")
    
    st.divider()
    
    if not st.session_state.report_generated:
        st.info("Generate a comprehensive HTML report with all analysis steps and results.")
        if st.button("Generate Full Report", type="primary"):
            with st.spinner("Building your report..."):
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
        st.success("Report generated! All export formats ready.")
    
    st.subheader("Download Files")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("**CSV** — Processed data")
    with col2:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download", csv_data, file_name="processed_data.csv", mime="text/csv", key="dl_base_csv")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("**JSON** — Data as JSON records")
    with col2:
        json_data = df.to_json(orient='records')
        st.download_button("Download", json_data, file_name="data.json", mime="application/json", key="dl_base_json")
    
    if st.session_state.report_generated and st.session_state.report_files:
        for fmt, path in st.session_state.report_files.items():
            if Path(path).exists():
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{fmt.upper()}** — Report")
                with col2:
                    with open(path, 'rb') as f:
                        mime = {"html": "text/html", "notebook": "application/json", "markdown": "text/markdown", "json": "application/json"}.get(fmt.lower(), "application/octet-stream")
                        st.download_button("Download", f.read(), file_name=Path(path).name, mime=mime, key=f"dl_{fmt}")
    
    st.divider()
    if st.button("Start New Analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def main():
    st.title("AutoDS")
    st.caption("Automated Data Science Platform")
    
    current = st.session_state.current_step
    df = st.session_state.df
    mode = st.session_state.mode
    
    if current > 1 and df is None:
        st.warning("Session expired. Restarting from Step 1.")
        st.session_state.current_step = 1
        st.rerun()
        return
    
    if current > 3 and mode is None:
        st.warning("Mode selection missing. Returning to Step 2.")
        st.session_state.current_step = 2
        st.rerun()
        return
    
    steps = ["Upload", "Analyze", "Preprocess", "Model", "Reports"]
    current_step_idx = current - 1
    
    progress_cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(progress_cols, steps)):
        if i < current_step_idx:
            col.markdown(f"**{step}**")
        elif i == current_step_idx:
            col.markdown(f"**→ {step}**")
        else:
            col.caption(step)
    
    st.divider()
    
    if current == 1:
        render_step_1()
    elif current == 2:
        render_step_2()
    elif current == 3:
        render_step_3()
    elif current == 4:
        render_step_4()
    elif current == 5:
        render_step_5()

if __name__ == "__main__":
    main()
