"""Setup for AutoDS package."""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="autods",
    version="1.0.0",
    description="Automated Data Science Platform — profile, preprocess, train, report in one line.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AutoDS",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "scipy>=1.11.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "validators>=0.22.0",
        "pathvalidate>=3.2.0",
        "pyarrow>=12.0.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "ml": ["xgboost>=2.0.0", "lightgbm>=4.0.0"],
        "vision": ["Pillow>=10.0.0"],
        "audio": ["librosa>=0.10.0"],
        "kaggle": ["kaggle>=1.5.0"],
        "drive": ["gdown>=4.7.0"],
        "nlp": ["nltk>=3.8.0"],
        "app": ["streamlit>=1.28.0", "seaborn>=0.12.0", "matplotlib>=3.7.0"],
        "full": [
            "xgboost>=2.0.0", "lightgbm>=4.0.0",
            "Pillow>=10.0.0", "librosa>=0.10.0",
            "kaggle>=1.5.0", "gdown>=4.7.0",
            "nltk>=3.8.0", "streamlit>=1.28.0",
            "seaborn>=0.12.0", "matplotlib>=3.7.0",
        ],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0", "mypy>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "autods=autods.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
