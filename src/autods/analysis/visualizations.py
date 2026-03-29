"""Visualization generator for data profiling."""
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


class VisualizationGenerator:
    """Generate visualizations for data profiling."""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "reports/visualizations"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set styles
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
        self.generated_files: List[str] = []
    
    def generate_all_visualizations(self) -> Dict[str, List[str]]:
        """Generate all standard visualizations."""
        results = {
            'overview': self.generate_overview_plots(),
            'distributions': self.generate_distribution_plots(),
            'correlations': self.generate_correlation_plots(),
            'missing_values': self.generate_missing_value_plots(),
            'categorical': self.generate_categorical_plots(),
        }
        return results
    
    def generate_overview_plots(self) -> List[str]:
        """Generate overview plots."""
        files = []
        
        # Data types pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        dtype_counts = self.df.dtypes.apply(str).value_counts()
        ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Data Types Distribution')
        plt.tight_layout()
        
        filepath = self._save_plot(fig, 'overview_data_types.png')
        files.append(filepath)
        plt.close(fig)
        
        return files
    
    def generate_distribution_plots(self, max_cols: int = 10) -> List[str]:
        """Generate distribution plots for numeric columns."""
        files = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:max_cols]
        
        if len(numeric_cols) == 0:
            return files
        
        # Histograms
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            data = self.df[col].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_title(f'{col}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = self._save_plot(fig, 'distributions_histograms.png')
        files.append(filepath)
        plt.close(fig)
        
        # Box plots
        fig, ax = plt.subplots(figsize=(12, 6))
        data_to_plot = [self.df[col].dropna() for col in numeric_cols]
        bp = ax.boxplot(data_to_plot, labels=numeric_cols, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_title('Box Plots - Numeric Columns')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = self._save_plot(fig, 'distributions_boxplots.png')
        files.append(filepath)
        plt.close(fig)
        
        return files
    
    def generate_correlation_plots(self) -> List[str]:
        """Generate correlation heatmaps."""
        files = []
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return files
        
        # Correlation matrix heatmap
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, linewidths=0.5)
        ax.set_title('Correlation Matrix (Lower Triangle)')
        plt.tight_layout()
        
        filepath = self._save_plot(fig, 'correlation_matrix.png')
        files.append(filepath)
        plt.close(fig)
        
        # Full correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, ax=ax,
                   square=True, linewidths=0.5)
        ax.set_title('Full Correlation Matrix')
        plt.tight_layout()
        
        filepath = self._save_plot(fig, 'correlation_full.png')
        files.append(filepath)
        plt.close(fig)
        
        return files
    
    def generate_missing_value_plots(self) -> List[str]:
        """Generate missing value visualizations."""
        files = []
        
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) == 0:
            return files
        
        # Bar chart of missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_pct = (missing / len(self.df) * 100).sort_values(ascending=True)
        
        colors = ['red' if x > 50 else 'orange' if x > 20 else 'yellow' if x > 5 else 'green' 
                  for x in missing_pct.values]
        
        ax.barh(range(len(missing_pct)), missing_pct.values, color=colors)
        ax.set_yticks(range(len(missing_pct)))
        ax.set_yticklabels(missing_pct.index)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Column')
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='5% threshold')
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.7, label='20% threshold')
        ax.legend()
        plt.tight_layout()
        
        filepath = self._save_plot(fig, 'missing_values_bar.png')
        files.append(filepath)
        plt.close(fig)
        
        # Missing value heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, 
                   cmap='viridis', ax=ax)
        ax.set_title('Missing Value Pattern (Yellow = Missing)')
        plt.tight_layout()
        
        filepath = self._save_plot(fig, 'missing_values_heatmap.png')
        files.append(filepath)
        plt.close(fig)
        
        return files
    
    def generate_categorical_plots(self, max_cols: int = 6) -> List[str]:
        """Generate plots for categorical columns."""
        files = []
        
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cat_cols = [col for col in cat_cols if self.df[col].nunique() <= 30][:max_cols]
        
        if len(cat_cols) == 0:
            return files
        
        # Value counts bar charts
        n_cols = 2
        n_rows = (len(cat_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, col in enumerate(cat_cols):
            ax = axes[idx]
            value_counts = self.df[col].value_counts().head(15)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            ax.bar(range(len(value_counts)), value_counts.values, color=colors)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_title(f'{col} - Value Counts')
            ax.set_ylabel('Count')
        
        # Hide empty subplots
        for idx in range(len(cat_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        filepath = self._save_plot(fig, 'categorical_value_counts.png')
        files.append(filepath)
        plt.close(fig)
        
        return files
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save plot to file."""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        self.generated_files.append(str(filepath))
        return str(filepath)
    
    def get_generated_files(self) -> List[str]:
        """Get list of all generated visualization files."""
        return self.generated_files


class InteractiveVisualizationGenerator:
    """Generate interactive Plotly visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def create_interactive_correlation(self) -> go.Figure:
        """Create interactive correlation heatmap."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return go.Figure()
        
        corr_matrix = numeric_df.corr()
        
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.columns),
            annotation_text=corr_matrix.round(2).values,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            showscale=True
        )
        fig.update_layout(title='Interactive Correlation Matrix')
        
        return fig
    
    def create_interactive_histogram(self, column: str) -> go.Figure:
        """Create interactive histogram."""
        fig = px.histogram(self.df, x=column, nbins=50, 
                          title=f'Distribution of {column}',
                          marginal='box')
        return fig
    
    def create_interactive_scatter(self, x_col: str, y_col: str, 
                                    color_col: Optional[str] = None) -> go.Figure:
        """Create interactive scatter plot."""
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col,
                        title=f'{y_col} vs {x_col}',
                        opacity=0.6)
        return fig
    
    def create_interactive_missing_heatmap(self) -> go.Figure:
        """Create interactive missing value heatmap."""
        missing_df = self.df.isnull().astype(int)
        
        fig = px.imshow(missing_df.T,
                       labels=dict(x="Row Index", y="Column", color="Missing"),
                       title="Missing Values Heatmap",
                       color_continuous_scale=['lightblue', 'red'])
        
        return fig
    
    def save_plotly_html(self, fig: go.Figure, filename: str, 
                        output_dir: str = "reports/visualizations") -> str:
        """Save plotly figure as HTML."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        fig.write_html(str(filepath))
        return str(filepath)
