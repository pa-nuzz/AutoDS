"""AI enhancement layer for LLM-powered data analysis."""
import os
import json
from typing import Dict, Any, List, Optional, Union
import requests
from dotenv import load_dotenv


load_dotenv()


class LLMClient:
    """Client for LLM API calls with support for multiple providers."""
    
    PROVIDER_GEMINI = "gemini"
    PROVIDER_OPENROUTER = "openrouter"
    PROVIDER_GROQ = "groq"
    PROVIDER_DEEPSEEK = "deepseek"
    
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openrouter")).lower()
        self.api_key = api_key or self._get_api_key()
        self.base_url = self._get_base_url()
        self.model = self._get_model()
        
        if not self.api_key:
            raise ValueError(f"API key not found for provider: {self.provider}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment based on provider."""
        key_mapping = {
            self.PROVIDER_GEMINI: "GEMINI_API_KEY",
            self.PROVIDER_OPENROUTER: "OPENROUTER_API_KEY",
            self.PROVIDER_GROQ: "GROQ_API_KEY",
            self.PROVIDER_DEEPSEEK: "DEEPSEEK_API_KEY",
        }
        env_var = key_mapping.get(self.provider)
        return os.getenv(env_var) if env_var else None
    
    def _get_base_url(self) -> str:
        """Get API base URL for provider."""
        urls = {
            self.PROVIDER_GEMINI: "https://generativelanguage.googleapis.com/v1beta",
            self.PROVIDER_OPENROUTER: "https://openrouter.ai/api/v1",
            self.PROVIDER_GROQ: "https://api.groq.com/openai/v1",
            self.PROVIDER_DEEPSEEK: "https://api.deepseek.com/v1",
        }
        return urls.get(self.provider, urls[self.PROVIDER_OPENROUTER])
    
    def _get_model(self) -> str:
        """Get default model for provider."""
        models = {
            self.PROVIDER_GEMINI: "gemini-1.5-flash",
            self.PROVIDER_OPENROUTER: "anthropic/claude-3.5-sonnet",
            self.PROVIDER_GROQ: "llama-3.3-70b-versatile",
            self.PROVIDER_DEEPSEEK: "deepseek-chat",
        }
        return os.getenv("LLM_MODEL") or models.get(self.provider, models[self.PROVIDER_OPENROUTER])
    
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
        """Generate text from LLM."""
        if self.provider == self.PROVIDER_GEMINI:
            return self._call_gemini(prompt, max_tokens, temperature)
        else:
            return self._call_openai_compatible(prompt, max_tokens, temperature)
    
    def _call_gemini(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Gemini API."""
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return "No response generated"
    
    def _call_openai_compatible(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call OpenAI-compatible API (OpenRouter, Groq, DeepSeek)."""
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add specific headers for OpenRouter
        if self.provider == self.PROVIDER_OPENROUTER:
            headers["HTTP-Referer"] = "https://autods.ai"
            headers["X-Title"] = "AutoDS AI Assistant"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        return "No response generated"


class AIDataSummarizer:
    """Generate AI-powered summaries of data analysis."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, use_llm: bool = True):
        self.use_llm = use_llm and llm_client is not None
        self.llm = llm_client if use_llm else None
    
    def summarize_dataset(self, profile: Dict[str, Any]) -> str:
        """Generate human-readable dataset summary."""
        if not self.use_llm or not self.llm:
            return self._template_summary(profile)
        
        try:
            prompt = self._build_dataset_prompt(profile)
            return self.llm.generate(prompt)
        except Exception as e:
            # Fallback to template on API failure
            return self._template_summary(profile) + f"\n\n(LLM Error: {str(e)})"
    
    def summarize_insights(self, insights: List[Dict[str, Any]], 
                          suggestions: List[Dict[str, Any]]) -> str:
        """Generate summary of insights and recommendations."""
        if not self.use_llm or not self.llm:
            return self._template_insights(insights, suggestions)
        
        try:
            prompt = self._build_insights_prompt(insights, suggestions)
            return self.llm.generate(prompt)
        except Exception as e:
            return self._template_insights(insights, suggestions) + f"\n\n(LLM Error: {str(e)})"
    
    def generate_recommendations(self, profile: Dict[str, Any]) -> str:
        """Generate preprocessing and modeling recommendations."""
        if not self.use_llm or not self.llm:
            return self._template_recommendations(profile)
        
        try:
            prompt = self._build_recommendations_prompt(profile)
            return self.llm.generate(prompt)
        except Exception as e:
            return self._template_recommendations(profile) + f"\n\n(LLM Error: {str(e)})"
    
    def _build_dataset_prompt(self, profile: Dict[str, Any]) -> str:
        """Build prompt for dataset summary."""
        overview = profile.get('overview', {})
        quality = profile.get('quality_score', {})
        target = profile.get('target_suggestion', 'Unknown')
        
        columns_info = []
        for col, info in profile.get('columns', {}).items():
            type_info = info.get('type_info', {})
            stats = info.get('statistics', {})
            columns_info.append({
                'name': col,
                'type': type_info.get('type', 'unknown'),
                'missing_pct': stats.get('missing_pct', 0),
                'unique': stats.get('unique', 0),
            })
        
        prompt = f"""You are a data science expert. Provide a clear, concise summary of this dataset for a data analyst.

DATASET OVERVIEW:
- Rows: {overview.get('n_rows', 'N/A')}
- Columns: {overview.get('n_columns', 'N/A')}
- Data Quality Score: {quality.get('overall', 'N/A')}/100
- Suggested Target: {target}

COLUMN SUMMARY:
{json.dumps(columns_info[:10], indent=2)}

Please provide:
1. A 2-3 sentence overview of the dataset
2. Key characteristics (size, quality, complexity)
3. The likely ML task type (regression, classification, clustering)
4. Main data quality concerns

Be concise and actionable."""
        
        return prompt
    
    def _build_insights_prompt(self, insights: List[Dict], suggestions: List[Dict]) -> str:
        """Build prompt for insights summary."""
        critical = [i for i in insights if i['severity'] == 'critical']
        high = [i for i in insights if i['severity'] == 'high']
        
        prompt = f"""You are a data science expert. Summarize these data quality findings and recommendations.

CRITICAL ISSUES ({len(critical)}):
{json.dumps(critical[:5], indent=2)}

HIGH PRIORITY ISSUES ({len(high)}):
{json.dumps(high[:5], indent=2)}

TOP RECOMMENDATIONS ({len(suggestions)}):
{json.dumps(suggestions[:5], indent=2)}

Please provide:
1. The 3 most important issues to address (with brief explanations)
2. Priority-ordered action items
3. Expected impact of fixes

Be specific and actionable."""
        
        return prompt
    
    def _build_recommendations_prompt(self, profile: Dict[str, Any]) -> str:
        """Build prompt for recommendations."""
        overview = profile.get('overview', {})
        target = profile.get('target_suggestion')
        correlations = profile.get('correlations', {})
        
        prompt = f"""You are a data science expert. Provide preprocessing and modeling recommendations.

DATASET:
- {overview.get('n_rows')} rows, {overview.get('n_columns')} columns
- Target: {target or 'Not detected'}
- High correlations: {len(correlations.get('high_correlations', []))}

Please provide:
1. Preprocessing steps (handling missing values, encoding, scaling)
2. Feature engineering suggestions
3. Recommended models (3-5 options with brief justification)
4. Validation strategy

Be specific and include brief rationale for each recommendation."""
        
        return prompt
    
    def _template_summary(self, profile: Dict[str, Any]) -> str:
        """Template-based dataset summary."""
        overview = profile.get('overview', {})
        quality = profile.get('quality_score', {})
        target = profile.get('target_suggestion')
        
        lines = [
            "Dataset Overview:",
            f"  • {overview.get('n_rows', 0):,} rows × {overview.get('n_columns', 0)} columns",
            f"  • Quality Score: {quality.get('overall', 0)}/100",
            f"  • Missing: {overview.get('missing_pct', 0):.1f}% of cells",
            "",
        ]
        
        if target:
            target_info = profile.get('columns', {}).get(target, {})
            target_type = target_info.get('type_info', {}).get('type', 'unknown')
            lines.extend([
                f"Suggested Target: {target} ({target_type})",
                "",
            ])
        
        lines.extend([
            "Key Characteristics:",
            f"  • Completeness: {quality.get('completeness', 0)}/100",
            f"  • Uniqueness: {quality.get('uniqueness', 0)}/100",
            f"  • Consistency: {quality.get('consistency', 0)}/100",
        ])
        
        return "\n".join(lines)
    
    def _template_insights(self, insights: List[Dict], suggestions: List[Dict]) -> str:
        """Template-based insights summary."""
        lines = ["Key Findings:", ""]
        
        critical = [i for i in insights if i['severity'] == 'critical']
        high = [i for i in insights if i['severity'] == 'high']
        
        if critical:
            lines.extend(["Critical Issues (Fix Immediately):"])
            for i in critical[:3]:
                lines.append(f"  ⚠️  {i['message']}")
            lines.append("")
        
        if high:
            lines.extend(["High Priority Issues:"])
            for i in high[:3]:
                lines.append(f"  ⚡ {i['message']}")
            lines.append("")
        
        if suggestions:
            lines.extend(["Top Recommendations:"])
            for s in suggestions[:5]:
                lines.append(f"  → {s['action']}")
                lines.append(f"     Reason: {s['reason']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _template_recommendations(self, profile: Dict[str, Any]) -> str:
        """Template-based recommendations."""
        target = profile.get('target_suggestion')
        target_info = profile.get('columns', {}).get(target, {}) if target else {}
        target_type = target_info.get('type_info', {}).get('type', 'unknown') if target_info else 'unknown'
        
        lines = [
            "Recommended Next Steps:",
            "",
            "1. Preprocessing:",
            "   • Handle missing values (mean/median imputation or drop)",
            "   • Encode categorical variables (one-hot or ordinal)",
            "   • Scale numeric features (StandardScaler or RobustScaler)",
            "",
        ]
        
        if target:
            if target_type in ['categorical_nominal', 'boolean', 'categorical_ordinal']:
                lines.extend([
                    "2. Modeling (Classification):",
                    "   • Random Forest (handles mixed data types)",
                    "   • XGBoost/LightGBM (excellent performance)",
                    "   • Logistic Regression (baseline, interpretable)",
                ])
            else:
                lines.extend([
                    "2. Modeling (Regression):",
                    "   • XGBoost/LightGBM (best performance)",
                    "   • Random Forest (handles non-linearity)",
                    "   • Ridge/Lasso (regularized linear models)",
                ])
        else:
            lines.extend([
                "2. Modeling (Unsupervised):",
                "   • K-Means clustering (explore natural groupings)",
                "   • DBSCAN (find dense clusters)",
                "   • PCA (dimensionality reduction)",
            ])
        
        lines.extend([
            "",
            "3. Validation:",
            "   • Use stratified K-fold if classification",
            "   • Train/validation/test split (70/15/15)",
            "   • Cross-validation for robust metrics",
        ])
        
        return "\n".join(lines)


class AIFallback:
    """Provides local templated responses when LLM is unavailable."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Load response templates."""
        return {
            'summary': {
                'small_clean': "This is a small, clean dataset ready for modeling.",
                'large_clean': "This is a large dataset with good quality. Consider sampling for initial exploration.",
                'missing_heavy': "This dataset has significant missing values. Address these before modeling.",
                'high_cardinality': "Several columns have many unique values. Consider feature engineering.",
            },
            'task_type': {
                'binary': "Binary classification task detected. Use precision, recall, F1, and ROC-AUC metrics.",
                'multiclass': "Multi-class classification. Consider class imbalance and use appropriate metrics.",
                'regression': "Regression task. Check for outliers and consider MAE/RMSE metrics.",
            }
        }
    
    def get_summary(self, profile: Dict[str, Any]) -> str:
        """Get template-based summary."""
        quality = profile.get('quality_score', {}).get('overall', 50)
        missing = profile.get('overview', {}).get('missing_pct', 0)
        n_rows = profile.get('overview', {}).get('n_rows', 0)
        
        if missing > 30:
            return self.templates['summary']['missing_heavy']
        elif n_rows > 100000:
            return self.templates['summary']['large_clean']
        else:
            return self.templates['summary']['small_clean']
