"""DIY Guide Generator - Step-by-step preprocessing instructions."""
from typing import Dict, Any, List, Optional
import pandas as pd

from .detector import PreprocessingDetector, PreprocessingNeed, NeedSeverity


class DIYGuide:
    """Generate step-by-step preprocessing instructions for DIY mode."""
    
    def __init__(self, detector: PreprocessingDetector):
        self.detector = detector
        self.guide_steps = []
        self._generate_guide()
    
    def _generate_guide(self):
        """Generate complete preprocessing guide."""
        needs = self.detector.get_needs()
        
        # Sort by severity: required first, then recommended, then optional
        severity_order = {NeedSeverity.REQUIRED: 0, NeedSeverity.RECOMMENDED: 1, NeedSeverity.OPTIONAL: 2}
        needs.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        # Group by type
        by_type = {}
        for need in needs:
            t = need['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(need)
        
        # Generate steps in logical order
        step_num = 1
        
        # 1. Data cleaning (duplicates, missing values)
        if PreprocessingNeed.DUPLICATE_REMOVAL in by_type:
            self._add_duplicate_step(step_num, by_type[PreprocessingNeed.DUPLICATE_REMOVAL])
            step_num += 1
        
        if PreprocessingNeed.MISSING_VALUES in by_type:
            self._add_missing_values_step(step_num, by_type[PreprocessingNeed.MISSING_VALUES])
            step_num += 1
        
        # 2. Feature engineering (datetime)
        self._add_datetime_step(step_num)
        step_num += 1
        
        # 3. Outlier treatment
        if PreprocessingNeed.OUTLIER_TREATMENT in by_type:
            self._add_outlier_step(step_num, by_type[PreprocessingNeed.OUTLIER_TREATMENT])
            step_num += 1
        
        # 4. Encoding
        if PreprocessingNeed.ENCODING in by_type:
            self._add_encoding_step(step_num, by_type[PreprocessingNeed.ENCODING])
            step_num += 1
        
        # 5. Text processing
        if PreprocessingNeed.TEXT_VECTORIZATION in by_type:
            self._add_text_step(step_num, by_type[PreprocessingNeed.TEXT_VECTORIZATION])
            step_num += 1
        
        # 6. Feature selection / cardinality reduction
        if PreprocessingNeed.FEATURE_SELECTION in by_type:
            self._add_cardinality_step(step_num, by_type[PreprocessingNeed.FEATURE_SELECTION])
            step_num += 1
        
        # 7. Scaling (always last)
        if PreprocessingNeed.SCALING in by_type:
            self._add_scaling_step(step_num, by_type[PreprocessingNeed.SCALING])
            step_num += 1
        
        # 8. Class imbalance (if target is present)
        imbalance_needs = [n for n in needs if n['type'] == 'class_imbalance']
        if imbalance_needs:
            self._add_imbalance_step(step_num, imbalance_needs)
            step_num += 1
    
    def _add_step(self, number: int, title: str, description: str, 
                  code: str, explanation: str, severity: str):
        """Add a guide step."""
        self.guide_steps.append({
            'step_number': number,
            'title': title,
            'description': description,
            'code': code,
            'explanation': explanation,
            'severity': severity
        })
    
    def _add_duplicate_step(self, step_num: int, needs: List[Dict]):
        """Add duplicate removal step."""
        need = needs[0]  # Usually only one
        count = need['details'].get('duplicate_count', 0)
        
        self._add_step(
            step_num,
            "Remove Duplicate Rows",
            f"Found {count} duplicate rows that should be removed",
            """# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Found {duplicate_count} duplicate rows")

# Remove duplicates
df = df.drop_duplicates()

# Alternative: Keep last occurrence instead of first
df = df.drop_duplicates(keep='last')""",
            "Duplicate rows can bias your model and lead to data leakage between train/test sets. Always remove them early in the pipeline.",
            need['severity']
        )
    
    def _add_missing_values_step(self, step_num: int, needs: List[Dict]):
        """Add missing value handling step."""
        # Group by strategy
        drop_cols = []
        advanced_cols = []
        simple_cols = []
        
        for need in needs:
            strategy = need['details'].get('strategy', '')
            cols = need['columns']
            
            if strategy == 'drop_column':
                drop_cols.extend(cols)
            elif strategy == 'advanced_imputation':
                advanced_cols.extend(cols)
            else:
                simple_cols.extend(cols)
        
        code_parts = []
        explanations = []
        
        if drop_cols:
            code_parts.append(f"""# Drop columns with excessive missing values (>50%)
cols_to_drop = {drop_cols}
df = df.drop(columns=cols_to_drop)""")
            explanations.append(f"Drop {len(drop_cols)} columns with >50% missing values")
        
        if simple_cols:
            code_parts.append(f"""# Simple imputation for columns with moderate missing values
from sklearn.impute import SimpleImputer

# For numeric columns - use median
numeric_cols_with_missing = [col for col in {simple_cols} if df[col].dtype in ['int64', 'float64']]
imputer = SimpleImputer(strategy='median')
df[numeric_cols_with_missing] = imputer.fit_transform(df[numeric_cols_with_missing])

# For categorical columns - use most frequent
categorical_cols_with_missing = [col for col in {simple_cols} if col not in numeric_cols_with_missing]
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols_with_missing] = imputer.fit_transform(df[categorical_cols_with_missing])""")
            explanations.append(f"Use simple imputation (median/mode) for {len(simple_cols)} columns with 5-20% missing")
        
        if advanced_cols:
            code_parts.append(f"""# Advanced imputation for columns with high missing values
from sklearn.impute import KNNImputer

# KNN imputation uses similar rows to estimate missing values
imputer = KNNImputer(n_neighbors=5)
cols_to_impute = {advanced_cols}
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])""")
            explanations.append(f"Use KNN imputation for {len(advanced_cols)} columns with 20-50% missing")
        
        self._add_step(
            step_num,
            "Handle Missing Values",
            f"Missing values detected in {len(simple_cols) + len(advanced_cols) + len(drop_cols)} columns",
            "\n\n".join(code_parts),
            "\n".join(explanations) + "\n\nMissing values can cause errors in most ML algorithms. Choose strategy based on missing percentage and column importance.",
            NeedSeverity.REQUIRED if drop_cols else NeedSeverity.RECOMMENDED
        )
    
    def _add_datetime_step(self, step_num: int):
        """Add datetime feature extraction step."""
        datetime_cols = [col for col, info in self.detector.type_detector.columns_info.items()
                        if info.get('type') == 'datetime']
        
        if not datetime_cols:
            return
        
        code = """# Extract datetime features
for col in datetime_columns:
    # Convert to datetime if not already
    df[col] = pd.to_datetime(df[col])
    
    # Extract components
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df[f'{col}_dayofweek'] = df[col].dt.dayofweek  # 0=Monday, 6=Sunday
    df[f'{col}_quarter'] = df[col].dt.quarter
    df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Drop original datetime column
    df = df.drop(columns=[col])"""
        
        self._add_step(
            step_num,
            "Extract Datetime Features",
            f"Found {len(datetime_cols)} datetime column(s): {datetime_cols}",
            code,
            "ML models can't directly use datetime objects. Extract meaningful numeric features like year, month, day of week.",
            NeedSeverity.RECOMMENDED
        )
    
    def _add_outlier_step(self, step_num: int, needs: List[Dict]):
        """Add outlier treatment step."""
        cols = []
        for need in needs:
            cols.extend(need['columns'])
        cols = list(set(cols))
        
        self._add_step(
            step_num,
            "Handle Outliers",
            f"Found {len(cols)} numeric columns with significant outliers",
            f"""# Method 1: Capping (Winsorization)
from scipy.stats import mstats

for col in {cols[:3]}:  # Apply to outlier columns
    # Cap at 1st and 99th percentile
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# Method 2: Log transformation (for right-skewed data)
# df['log_col'] = np.log1p(df[col])  # Use log1p to handle zeros

# Method 3: IQR-based removal
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]""",
            "Outliers can skew model training. Choose between capping (keeps data) vs removal (aggressive). Use RobustScaler if keeping outliers.",
            NeedSeverity.RECOMMENDED
        )
    
    def _add_encoding_step(self, step_num: int, needs: List[Dict]):
        """Add categorical encoding step."""
        nominal_cols = []
        ordinal_cols = []
        
        for need in needs:
            encoding_type = need['details'].get('encoding_type')
            if encoding_type == 'nominal':
                nominal_cols.extend(need['columns'])
            elif encoding_type == 'ordinal':
                ordinal_cols.extend(need['columns'])
        
        code_parts = []
        
        if nominal_cols:
            # Split by cardinality
            low_card = [c for c in nominal_cols if self.detector.df[c].nunique() <= 10]
            high_card = [c for c in nominal_cols if self.detector.df[c].nunique() > 10]
            
            if low_card:
                code_parts.append(f"""# One-hot encoding for low cardinality categorical columns ({len(low_card)} columns)
from sklearn.preprocessing import OneHotEncoder

# For columns with ≤10 categories
low_cardinality_cols = {low_card}
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded = encoder.fit_transform(df[low_cardinality_cols])

# Create DataFrame with encoded features
encoded_cols = encoder.get_feature_names_out(low_cardinality_cols)
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

# Combine and drop original columns
df = pd.concat([df.drop(columns=low_cardinality_cols), encoded_df], axis=1)""")
            
            if high_card:
                code_parts.append(f"""# Target encoding for high cardinality columns ({len(high_card)} columns)
# Alternative to one-hot when there are many categories
from category_encoders import TargetEncoder

high_cardinality_cols = {high_card}
encoder = TargetEncoder(cols=high_cardinality_cols)
df[high_cardinality_cols] = encoder.fit_transform(df[high_cardinality_cols], y)""")
        
        if ordinal_cols:
            code_parts.append(f"""# Ordinal encoding for ordered categorical columns ({len(ordinal_cols)} columns)
from sklearn.preprocessing import OrdinalEncoder

ordinal_cols = {ordinal_cols}
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])""")
        
        self._add_step(
            step_num,
            "Encode Categorical Variables",
            f"Found {len(nominal_cols)} nominal and {len(ordinal_cols)} ordinal categorical columns",
            "\n\n".join(code_parts),
            "ML models require numeric input. Use one-hot for nominal (no order), ordinal encoder when order matters. For high cardinality, try target encoding.",
            NeedSeverity.REQUIRED
        )
    
    def _add_text_step(self, step_num: int, needs: List[Dict]):
        """Add text preprocessing step."""
        text_cols = []
        for need in needs:
            text_cols.extend(need['columns'])
        
        self._add_step(
            step_num,
            "Vectorize Text Features",
            f"Found {len(text_cols)} text column(s): {text_cols}",
            """# Text vectorization using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# For a single text column
text_col = 'text_column_name'
vectorizer = TfidfVectorizer(
    max_features=100,      # Limit vocabulary size
    stop_words='english', # Remove common words
    ngram_range=(1, 2),   # Use unigrams and bigrams
    min_df=2              # Ignore rare terms
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(df[text_col])

# Convert to DataFrame (optional)
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
    index=df.index
)

# Combine with main dataframe
df = pd.concat([df.drop(columns=[text_col]), tfidf_df], axis=1)

# Alternative: Use pre-trained embeddings
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(df[text_col].tolist())""",
            "Text must be converted to numeric form. TF-IDF is simple and effective. For better semantic meaning, use embeddings (BERT, Word2Vec).",
            NeedSeverity.RECOMMENDED
        )
    
    def _add_cardinality_step(self, step_num: int, needs: List[Dict]):
        """Add cardinality reduction step."""
        cols = []
        for need in needs:
            cols.extend(need['columns'])
        
        self._add_step(
            step_num,
            "Reduce High Cardinality",
            f"Found {len(cols)} columns with >50 unique categories",
            f"""# Reduce cardinality by grouping rare categories
for col in {cols}:
    # Calculate value counts
    value_counts = df[col].value_counts(normalize=True)
    
    # Keep categories that appear in >1% of rows, group rest as 'Other'
    threshold = 0.01
    rare_categories = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(rare_categories, 'Other')
    
    print(f"{{col}}: Reduced from {{len(value_counts)}} to {{df[col].nunique()}} categories")

# Alternative: Use frequency encoding
# df[col + '_freq'] = df[col].map(df[col].value_counts())""",
            "High cardinality leads to too many features (curse of dimensionality) or sparse data. Group rare categories or use target/frequency encoding.",
            NeedSeverity.OPTIONAL
        )
    
    def _add_scaling_step(self, step_num: int, needs: List[Dict]):
        """Add scaling step."""
        need = needs[0]
        numeric_cols = need['details'].get('all_numeric', [])
        scaler_type = need['details'].get('scaler', 'standard')
        
        scaler_code = {
            'standard': "StandardScaler()  # Zero mean, unit variance - good for normally distributed data",
            'minmax': "MinMaxScaler()  # Scale to [0, 1] - preserves zero values, good for neural networks",
            'robust': "RobustScaler()  # Uses median/IQR - good when outliers are present",
        }
        
        self._add_step(
            step_num,
            "Scale Numeric Features",
            f"Scale {len(numeric_cols)} numeric columns",
            f"""# Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Choose scaler based on your data:
scaler = {scaler_code.get(scaler_type, scaler_code['standard'])}

# Fit and transform
numeric_cols = {numeric_cols}
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save scaler for later use on test data
import joblib
joblib.dump(scaler, 'scaler.pkl')""",
            "Scaling ensures all features contribute equally. Required for algorithms sensitive to magnitude (SVM, KNN, neural networks, regularized linear models). Tree-based models don't require scaling.",
            need['severity']
        )
    
    def _add_imbalance_step(self, step_num: int, needs: List[Dict]):
        """Add class imbalance handling step."""
        need = needs[0]
        ratio = need['details'].get('imbalance_ratio', 1)
        
        self._add_step(
            step_num,
            "Address Class Imbalance",
            f"Target has class imbalance (ratio {ratio:.1f}:1)",
            """# Handle class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Method 1: Use class weights in model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')

# Method 2: SMOTE - Synthetic oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Method 3: Random undersampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Method 4: Combine over and undersampling
from imblearn.pipeline import Pipeline
pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.5)),  # Increase minority to 50% of majority
    ('under', RandomUnderSampler(sampling_strategy=0.8))  # Reduce majority
])""",
            "Class imbalance causes models to bias toward majority class. Use class weights (simple), SMOTE (generates synthetic samples), or resampling techniques.",
            NeedSeverity.RECOMMENDED
        )
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """Get all guide steps."""
        return self.guide_steps
    
    def get_markdown_guide(self) -> str:
        """Generate markdown formatted guide."""
        lines = ["# Data Preprocessing Guide (DIY Mode)", ""]
        
        # Summary
        needs_summary = self.detector.get_summary()
        lines.extend([
            "## Summary",
            f"- Total preprocessing needs detected: {needs_summary['total_needs']}",
            f"- Required steps: {needs_summary['required']}",
            f"- Recommended steps: {needs_summary['recommended']}",
            f"- Optional steps: {needs_summary['optional']}",
            "",
        ])
        
        # Steps
        for step in self.guide_steps:
            severity_emoji = {"required": "🔴", "recommended": "🟡", "optional": "🟢"}
            emoji = severity_emoji.get(step['severity'], "⚪")
            
            lines.extend([
                f"## Step {step['step_number']}: {emoji} {step['title']}",
                "",
                f"**Priority:** {step['severity'].upper()}",
                "",
                f"{step['description']}",
                "",
                "### Code:",
                "```python",
                step['code'],
                "```",
                "",
                "### Why this matters:",
                step['explanation'],
                "",
                "---",
                "",
            ])
        
        # Footer
        lines.extend([
            "## Next Steps",
            "",
            "After completing preprocessing:",
            "1. Split data into train/validation/test sets",
            "2. Consider feature selection/engineering",
            "3. Choose appropriate ML model for your task",
            "",
            "---",
            "*Generated by AutoDS AI Assistant*",
        ])
        
        return "\n".join(lines)
    
    def save_guide(self, filepath: str):
        """Save guide to markdown file."""
        with open(filepath, 'w') as f:
            f.write(self.get_markdown_guide())
