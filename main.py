"""AutoDS AI Assistant - Main entry point."""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autods.data.ingestion import DataIngestion


def main():
    """Main entry point for AutoDS."""
    print("=" * 60)
    print("AutoDS AI Assistant - All 6 Phases Complete")
    print("=" * 60)
    print()
    
    # Initialize
    ingestion = DataIngestion()
    
    print("Directory structure created:")
    print(f"  - raw_data: {ingestion.raw_data_dir.absolute()}")
    print(f"  - processed_data: {ingestion.processed_data_dir.absolute()}")
    print()
    
    print("PHASE 1: Input Handling & Data Fetching")
    print("-" * 40)
    print("  Formats: CSV, Excel, JSON, Parquet, Images, Audio, Archives")
    print("  URLs: Kaggle, GitHub, Google Drive, Direct links")
    print()
    
    print("PHASE 2: Data Profiling & Exploration")
    print("-" * 40)
    print("  Auto-detect types, statistical profiling, visualizations")
    print("  AI insights (Gemini, OpenRouter, Groq, DeepSeek)")
    print()
    
    print("PHASE 3: Preprocessing Suggestions & Automation")
    print("-" * 40)
    print("  'Do it for me' / 'DIY' modes, imputation, encoding, scaling")
    print()
    
    print("PHASE 4: Model Recommendation & Baseline Training")
    print("-" * 40)
    print("  Task detection, model recommendations, auto-training")
    print("  XGBoost, LightGBM, Random Forest, Linear models")
    print()
    
    print("PHASE 5: Reporting & Education")
    print("-" * 40)
    print("  Comprehensive reports with all phases")
    print("  Export: HTML, Jupyter Notebook, Markdown, JSON, CSV")
    print("  Plain-English AI insights and step-by-step guides")
    print()
    
    print("PHASE 6: UI/UX Design")
    print("-" * 40)
    print("  Streamlit web interface")
    print("  Premium design: Light theme, blues/greys, modern layout")
    print("  Tabs: Upload, Explore, Preprocess, Models, Reports, Auto Mode")
    print("  Interactive plots, drag-drop upload, Auto/DIY toggle")
    print()
    
    print("PHASE 7: Security & Best Practices (COMPLETE)")
    print("-" * 40)
    print("  API key manager with rotation/fallback")
    print("  File safety: executable detection, size limits, ZIP bomb detection")
    print("  User-friendly error handling with actionable messages")
    print()
    
    print("PHASE 8: Optional Extras (COMPLETE)")
    print("-" * 40)
    print("  Session persistence for saving/reusing runs")
    print("  Interactive checklist for next steps")
    print("  Export ready-to-run notebooks/scripts")
    print("  Demo datasets: Customer Churn, House Prices, Sales Forecast, Iris Extended, Employee Attrition")
    print()
    
    print("Launch the Web App:")
    print("  streamlit run app.py")
    print()
    
    print("Demo Datasets Available:")
    print("  Customer Churn, House Prices, Sales Forecast, Iris Extended, Employee Attrition")
    print()
    
    print("Quick Start - Complete Pipeline (Code):")
    print("  from autods.reports.pipeline import run_complete_analysis")
    print('  results = run_complete_analysis(df, target="target_column")')
    print()


if __name__ == "__main__":
    main()
