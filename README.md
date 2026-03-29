# AutoDS AI Assistant - Complete Data Science Platform

A full-stack automated data science and machine learning platform with 6 complete phases.

## 🚀 Quick Start

### Launch the Web App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📋 All 6 Phases Complete

### Phase 1: Input Handling & Data Fetching
- **Files**: CSV, Excel, JSON, Parquet, HDF5, Images, Audio, Archives
- **URLs**: Kaggle, GitHub, Google Drive, Direct links
- **Security**: File validation, sanitization, malware detection

### Phase 2: Data Profiling & Exploration
- Auto-detect data types (numeric, categorical, text, datetime, image, audio)
- Statistical profiling (missing values, correlations, outliers)
- Interactive visualizations (Plotly charts)
- AI insights via LLM APIs (Gemini, OpenRouter, Groq, DeepSeek)

### Phase 3: Preprocessing Suggestions & Automation
- Need detection (missing, encoding, scaling, outliers, text, duplicates)
- **"Do It For Me"** mode: Auto preprocessing pipeline
- **"I Want To Do It"** mode: Step-by-step DIY guides
- Strategies: Imputation, encoding, scaling, text vectorization

### Phase 4: Model Recommendation & Baseline Training
- Task detection (regression, classification, clustering, NLP, image)
- Model recommendations with reasoning
- Auto-training: XGBoost, LightGBM, Random Forest, Linear models
- Performance metrics & cross-validation

### Phase 5: Reporting & Education
- Comprehensive reports combining all phases
- Export formats: HTML, Jupyter Notebook, Markdown, JSON, CSV
- Plain-English AI insights
- Step-by-step guides

### Phase 6: UI/UX Design
- **Streamlit web interface** with premium design
- Light theme with blues/greys
- Modern Inter font
- 6 tabs matching visual concept panels:
  - 📤 Upload (drag-drop + URL)
  - 🔍 Explore (interactive plots)
  - 🔧 Preprocess (Auto/DIY toggle)
  - 🤖 Models (recommendations + training)
  - 📊 Reports (exports)
  - ⚡ Auto Mode (one-click analysis)

## 🎯 Usage Examples

### Web Interface
```bash
# Launch the app
streamlit run app.py

# Access at http://localhost:8501
```

### Python API
```python
from autods.reports.pipeline import run_complete_analysis
import pandas as pd

# Load your data
df = pd.read_csv('data.csv')

# Run complete analysis
results = run_complete_analysis(
    df, 
    target='target_column',
    use_llm=True
)

# Access generated reports
html_report = results['exported_files']['html']
notebook = results['exported_files']['notebook']
csv_data = results['exported_files']['csv']
```

### Quick Preprocessing
```python
from autods.preprocessing.orchestrator import auto_preprocess

# One-line preprocessing
df_processed = auto_preprocess(df, target='target', mode='balanced')
```

### Model Training
```python
from autods.modeling.orchestrator import full_modeling_pipeline

# Get recommendations and train
results = full_modeling_pipeline(df, target='target')
```

## 📁 Project Structure

```
AutoDS/
├── app.py                          # Streamlit UI (Phase 6)
├── main.py                         # CLI entry point
├── requirements.txt                # Dependencies
├── .env                            # API keys (configured)
├── src/autods/
│   ├── data/
│   │   └── ingestion.py            # Phase 1
│   ├── input/
│   │   ├── file_handler.py         # Phase 1
│   │   └── url_handlers.py         # Phase 1
│   ├── security/
│   │   └── validator.py            # Phase 1
│   ├── analysis/
│   │   ├── type_detector.py        # Phase 2
│   │   ├── profiler.py             # Phase 2
│   │   ├── insights.py             # Phase 2
│   │   ├── visualizations.py       # Phase 2
│   │   └── ai_enhancement.py       # Phase 2
│   ├── preprocessing/
│   │   ├── detector.py             # Phase 3
│   │   ├── auto_processor.py       # Phase 3
│   │   ├── diy_guide.py            # Phase 3
│   │   └── orchestrator.py         # Phase 3
│   ├── modeling/
│   │   ├── task_detector.py        # Phase 4
│   │   ├── recommender.py          # Phase 4
│   │   ├── tabular_trainer.py      # Phase 4
│   │   └── orchestrator.py         # Phase 4
│   └── reports/
│       ├── generator.py            # Phase 2
│       ├── comprehensive_report.py # Phase 5
│       ├── html_generator.py       # Phase 5
│       ├── notebook_generator.py   # Phase 5
│       ├── export_manager.py       # Phase 5
│       └── pipeline.py             # Phase 5
├── raw_data/                       # Upload directory
├── processed_data/                 # Output directory
└── reports/                        # Generated reports
```

## 🔑 API Keys (Already Configured)

Your `.env` file contains:
- OpenRouter API key
- Gemini API key
- Groq API key
- DeepSeek API key

These power the AI insights features in Phases 2, 4, and 5.

## 🎨 UI Design Features

### Visual Concept Panels (from your uploaded image)
1. **Welcome Panel**: Hero banner with animated robot
2. **Upload Panel**: Drag-drop zones for files and URLs
3. **Explore Panel**: Interactive analysis with real-time charts
4. **Preprocess Panel**: Auto/DIY mode toggle cards
5. **Models Panel**: Recommendation cards with scores
6. **Auto Mode Panel**: One-click complete analysis

### Design System
- **Colors**: Primary blue (#3b82f6), Light backgrounds (#f8fafc)
- **Typography**: Inter font family
- **Effects**: Floating animations, hover transitions, gradient accents
- **Components**: Panel cards, upload zones, metric cards, progress indicators

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web app
streamlit run app.py
```

## 📊 Supported Data Types

| Type | Formats |
|------|---------|
| Tabular | CSV, Excel, JSON, Parquet, HDF5 |
| Text | TXT, MD, LOG |
| Images | JPG, PNG, GIF, BMP, TIFF |
| Audio | WAV, MP3, FLAC, AAC |
| Archives | ZIP, TAR, GZ |

## 🤖 Supported ML Tasks

- **Regression**: Linear, Ridge, Random Forest, XGBoost, LightGBM
- **Classification**: Logistic, Random Forest, XGBoost, LightGBM
- **Clustering**: K-Means, DBSCAN
- **NLP**: TF-IDF + Logistic, BERT (recommendations)
- **Images**: CNN, ResNet, EfficientNet (recommendations)

## 📝 License

MIT License - Built with ❤️ using Python, Streamlit, and scikit-learn
