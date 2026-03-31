"""Demo dataset generator for immediate app testing.

Creates sample datasets that users can try without uploading their own data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class DemoDatasetGenerator:
    """Generates various demo datasets for testing the app."""
    
    @staticmethod
    def generate_customer_churn(n_samples: int = 1000) -> pd.DataFrame:
        """Generate a customer churn prediction dataset."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'tenure': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples).round(2),
            'total_charges': np.random.uniform(100, 8000, n_samples).round(2),
            'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'payment_method': np.random.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                n_samples
            ),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
            'senior_citizen': np.random.choice([0, 1], n_samples),
            'partner': np.random.choice(['Yes', 'No'], n_samples),
            'dependents': np.random.choice(['Yes', 'No'], n_samples),
        })
        
        # Generate churn based on features (simulated)
        churn_prob = (
            0.3 * (df['tenure'] < 12) +
            0.2 * (df['contract'] == 'Month-to-month') +
            0.15 * (df['payment_method'] == 'Electronic check') +
            0.1 * (df['tech_support'] == 'No') +
            0.1 * (df['online_security'] == 'No') +
            0.05 * df['senior_citizen'] +
            np.random.uniform(0, 0.2, n_samples)
        )
        
        df['churn'] = (churn_prob > 0.5).astype(int)
        
        return df
    
    @staticmethod
    def generate_house_prices(n_samples: int = 1000) -> pd.DataFrame:
        """Generate a house price prediction dataset."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'id': range(1, n_samples + 1),
            'square_feet': np.random.randint(800, 4000, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples),
            'floors': np.random.randint(1, 3, n_samples),
            'year_built': np.random.randint(1950, 2023, n_samples),
            'lot_size': np.random.uniform(0.1, 2.0, n_samples).round(2),
            'garage_cars': np.random.randint(0, 4, n_samples),
            'has_pool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'has_basement': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'neighborhood': np.random.choice(
                ['Downtown', 'Suburban', 'Rural', 'Waterfront', 'Historic'],
                n_samples
            ),
            'condition': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples),
            'zipcode': np.random.choice(['10001', '90210', '60601', '77001', '85001'], n_samples),
        })
        
        # Generate price based on features
        base_price = 50000
        df['price'] = (
            base_price +
            df['square_feet'] * 150 +
            df['bedrooms'] * 25000 +
            df['bathrooms'] * 35000 +
            (2023 - df['year_built']) * -1000 +
            df['lot_size'] * 50000 +
            df['garage_cars'] * 15000 +
            df['has_pool'] * 30000 +
            df['has_basement'] * 20000 +
            df['neighborhood'].map({
                'Downtown': 100000, 'Waterfront': 150000, 'Historic': 80000,
                'Suburban': 50000, 'Rural': 20000
            }) +
            df['condition'].map({
                'Excellent': 50000, 'Good': 25000, 'Fair': 0, 'Poor': -30000
            }) +
            np.random.normal(0, 50000, n_samples)
        ).round(2)
        
        return df
    
    @staticmethod
    def generate_sales_forecast(n_samples: int = 500) -> pd.DataFrame:
        """Generate a sales forecasting dataset with time series."""
        np.random.seed(42)
        
        dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'quarter': dates.quarter,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'marketing_spend': np.random.uniform(0, 5000, n_samples).round(2),
            'temperature': np.random.uniform(30, 90, n_samples).round(1),
            'competitor_price': np.random.uniform(50, 150, n_samples).round(2),
            'product_category': np.random.choice(
                ['Electronics', 'Clothing', 'Home', 'Sports', 'Books'],
                n_samples
            ),
        })
        
        # Generate sales with trend and seasonality
        trend = np.linspace(1000, 2000, n_samples)
        seasonality = 200 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
        
        df['sales'] = (
            trend +
            seasonality +
            df['marketing_spend'] * 0.3 +
            df['is_holiday'] * 500 +
            df['is_weekend'] * 200 +
            df['product_category'].map({
                'Electronics': 500, 'Clothing': 300, 'Home': 200, 'Sports': 400, 'Books': 100
            }) +
            np.random.normal(0, 200, n_samples)
        ).round(2)
        
        return df
    
    @staticmethod
    def generate_iris_extended(n_samples: int = 300) -> pd.DataFrame:
        """Generate an extended iris-like dataset for classification."""
        np.random.seed(42)
        
        species = ['Setosa', 'Versicolor', 'Virginica']
        
        data = []
        for species_name in species:
            n = n_samples // 3
            
            if species_name == 'Setosa':
                sepal_length = np.random.normal(5.0, 0.3, n)
                sepal_width = np.random.normal(3.4, 0.3, n)
                petal_length = np.random.normal(1.5, 0.2, n)
                petal_width = np.random.normal(0.2, 0.1, n)
            elif species_name == 'Versicolor':
                sepal_length = np.random.normal(5.9, 0.4, n)
                sepal_width = np.random.normal(2.8, 0.3, n)
                petal_length = np.random.normal(4.3, 0.3, n)
                petal_width = np.random.normal(1.3, 0.2, n)
            else:  # Virginica
                sepal_length = np.random.normal(6.5, 0.5, n)
                sepal_width = np.random.normal(3.0, 0.3, n)
                petal_length = np.random.normal(5.5, 0.4, n)
                petal_width = np.random.normal(2.0, 0.3, n)
            
            for i in range(n):
                data.append({
                    'sepal_length': round(sepal_length[i], 2),
                    'sepal_width': round(sepal_width[i], 2),
                    'petal_length': round(petal_length[i], 2),
                    'petal_width': round(petal_width[i], 2),
                    'species': species_name,
                    'origin': np.random.choice(['Garden', 'Wild', 'Greenhouse']),
                    'measurement_quality': np.random.choice(['High', 'Medium', 'Low']),
                })
        
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def generate_employee_attrition(n_samples: int = 800) -> pd.DataFrame:
        """Generate an employee attrition/hr analytics dataset."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'employee_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 65, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'department': np.random.choice(
                ['Sales', 'R&D', 'HR', 'IT', 'Finance', 'Marketing'],
                n_samples
            ),
            'job_level': np.random.randint(1, 6, n_samples),
            'monthly_income': np.random.randint(2000, 20000, n_samples),
            'years_at_company': np.random.randint(0, 20, n_samples),
            'years_in_current_role': np.random.randint(0, 10, n_samples),
            'total_working_years': np.random.randint(0, 40, n_samples),
            'num_companies_worked': np.random.randint(0, 10, n_samples),
            'distance_from_home': np.random.randint(1, 30, n_samples),
            'work_life_balance': np.random.randint(1, 5, n_samples),
            'job_satisfaction': np.random.randint(1, 5, n_samples),
            'performance_rating': np.random.randint(3, 5, n_samples),
            'overtime': np.random.choice(['Yes', 'No'], n_samples),
            'business_travel': np.random.choice(
                ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
                n_samples
            ),
            'education': np.random.randint(1, 6, n_samples),
            'education_field': np.random.choice(
                ['Life Sciences', 'Medical', 'Marketing', 'Technical', 'Other'],
                n_samples
            ),
        })
        
        # Generate attrition based on factors
        attrition_prob = (
            0.02 * (df['age'] < 25) +
            0.01 * (df['years_at_company'] < 2) +
            0.02 * (df['job_satisfaction'] < 3) +
            0.015 * (df['work_life_balance'] < 3) +
            0.02 * (df['overtime'] == 'Yes') +
            0.01 * (df['distance_from_home'] > 20) +
            0.015 * (df['monthly_income'] < 3000) +
            0.01 * (df['years_in_current_role'] > 5) +
            np.random.uniform(0, 0.1, n_samples)
        )
        
        df['attrition'] = (attrition_prob > 0.5).astype(int)
        
        return df
    
    @staticmethod
    def get_demo_datasets() -> Dict[str, Dict]:
        """Get information about all available demo datasets."""
        return {
            'customer_churn': {
                'name': 'Customer Churn',
                'description': 'Telecom customer data for predicting churn',
                'task': 'Binary Classification',
                'target': 'churn',
                'n_samples': 1000,
                'generator': DemoDatasetGenerator.generate_customer_churn,
                'icon': '🔄'
            },
            'house_prices': {
                'name': 'House Prices',
                'description': 'Real estate data for price prediction',
                'task': 'Regression',
                'target': 'price',
                'n_samples': 1000,
                'generator': DemoDatasetGenerator.generate_house_prices,
                'icon': '🏠'
            },
            'sales_forecast': {
                'name': 'Sales Forecast',
                'description': 'Time series sales data with marketing factors',
                'task': 'Regression/Time Series',
                'target': 'sales',
                'n_samples': 500,
                'generator': DemoDatasetGenerator.generate_sales_forecast,
                'icon': '📈'
            },
            'iris_extended': {
                'name': 'Iris Extended',
                'description': 'Classic flower classification dataset with extra features',
                'task': 'Multi-class Classification',
                'target': 'species',
                'n_samples': 300,
                'generator': DemoDatasetGenerator.generate_iris_extended,
                'icon': '🌸'
            },
            'employee_attrition': {
                'name': 'Employee Attrition',
                'description': 'HR analytics data for predicting employee turnover',
                'task': 'Binary Classification',
                'target': 'attrition',
                'n_samples': 800,
                'generator': DemoDatasetGenerator.generate_employee_attrition,
                'icon': '👔'
            }
        }
    
    @staticmethod
    def load_demo_dataset(name: str, save_path: Optional[str] = None) -> pd.DataFrame:
        """Load a demo dataset by name.
        
        Args:
            name: Name of the demo dataset
            save_path: Optional path to save the CSV file
            
        Returns:
            DataFrame with the demo data
        """
        demos = DemoDatasetGenerator.get_demo_datasets()
        
        if name not in demos:
            raise ValueError(f"Unknown demo dataset: {name}. Available: {list(demos.keys())}")
        
        df = demos[name]['generator']()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
        
        return df
    
    @staticmethod
    def save_all_demos(output_dir: str = "demo_data") -> Dict[str, str]:
        """Generate and save all demo datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved = {}
        demos = DemoDatasetGenerator.get_demo_datasets()
        
        for key, info in demos.items():
            path = output_path / f"{key}.csv"
            df = info['generator']()
            df.to_csv(path, index=False)
            saved[key] = str(path)
        
        return saved


# Convenience functions for quick access
def load_demo_dataset(name: str, save_path=None):
    """Module-level function to load a demo dataset by name."""
    return DemoDatasetGenerator.load_demo_dataset(name, save_path)


def load_churn_demo() -> pd.DataFrame:
    """Load customer churn demo dataset."""
    return DemoDatasetGenerator.generate_customer_churn()


def load_house_prices_demo() -> pd.DataFrame:
    """Load house prices demo dataset."""
    return DemoDatasetGenerator.generate_house_prices()


def load_iris_demo() -> pd.DataFrame:
    """Load extended iris demo dataset."""
    return DemoDatasetGenerator.generate_iris_extended()
