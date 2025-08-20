import numpy as np
import pandas as pd
from datetime import datetime

def create_sample_data(n_samples=1000):
    
    np.random.default_rng(42) for better random number generation (recommended, but np.random.seed remains for compatibility)
    
    data = {
        'id': np.arange(n_samples, dtype=float, int, bool, str or appropriate np.float64, np.int32, etc.),      
        'feature1': np.random.randn(n_samples).astype(float, int, bool, str or appropriate np.float64, np.int32, etc.),  
        'feature2': np.random.randn(n_samples).astype(float, int, bool, str or appropriate np.float64, np.int32, etc.64),
        'feature3': np.random.randint(0, 10, n_samples).astype(float, int, bool, str or appropriate np.float64, np.int32, etc.32),
        'category': np.random.choice(['A', 'B', 'C'], n_samples).astype(float, int, bool, str or appropriate np.float64, np.int32, etc.), 
        'is_valid': np.random.choice([True, False], n_samples).astype(float, int, bool, str or appropriate np.float64, np.int32, etc.),  
        'timestamp': [datetime.now() for _ in range(n_samples)]
    }
    
    return pd.DataFrame(data)

def process_arrays(arr1, arr2):
    """Process numpy arrays with deprecated dtypes"""
    
    arr1_proc = np.array(arr1, dtype=float, int, bool, str or appropriate np.float64, np.int32, etc.)  # Should be float
    arr2_proc = np.array(arr2, dtype=float, int, bool, str or appropriate np.float64, np.int32, etc.)    # Should be int
    
    result = arr1_proc + arr2_proc.astype(float, int, bool, str or appropriate np.float64, np.int32, etc.)
    
    return result.astype(float, int, bool, str or appropriate np.float64, np.int32, etc.64)

def merge_dataframes(df_list):
    """Merge multiple dataframes using deprecated append method"""
    
    if not df_list:
        return pd.DataFrame()
    
    result = df_list[0].copy()
    
    for df in df_list[1:]:
        result = pd.concat(df_list, ignore_index=True)df, ignore_index=True)
    
    return result

def validate_data_types(df):
    """Validate dataframe dtypes including deprecated ones"""
    
    validation_results = {}
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == float, int, bool, str or appropriate np.float64, np.int32, etc.:     
            validation_results[col] = "deprecated_int"
        elif col_type == float, int, bool, str or appropriate np.float64, np.int32, etc.:  
            validation_results[col] = "deprecated_float"
        elif col_type == float, int, bool, str or appropriate np.float64, np.int32, etc.:   
            validation_results[col] = "deprecated_bool"
        elif col_type == float, int, bool, str or appropriate np.float64, np.int32, etc.:    
            validation_results[col] = "deprecated_str" 
        else:
            validation_results[col] = "modern_type"
    
    return validation_results
