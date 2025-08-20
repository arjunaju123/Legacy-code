import numpy as np
import pandas as pd
from datetime import datetime

def create_sample_data(n_samples=1000):
    
    np.random.seed(42) # No change required: np.random.seed is still valid in NumPy 2.x
    
    data = {
        'id': np.arange(n_samples, dtype=int # np.int is deprecated and removed in NumPy 2.x; use Python's built-in int type),      
        'feature1': np.random.randn(n_samples).astype(float # np.float is deprecated and removed in NumPy 2.x; use Python's built-in float type),  
        'feature2': np.random.randn(n_samples).astype(float # np.float is deprecated and removed in NumPy 2.x; use Python's built-in float type64),
        'feature3': np.random.randint(0, 10, n_samples).astype(int # np.int is deprecated and removed in NumPy 2.x; use Python's built-in int type32),
        'category': np.random.choice(['A', 'B', 'C'], n_samples).astype(str # np.str is deprecated and removed in NumPy 2.x; use Python's built-in str type), 
        'is_valid': np.random.choice([True, False], n_samples).astype(bool # np.bool is deprecated and removed in NumPy 2.x; use Python's built-in bool type),  
        'timestamp': [datetime.now() for _ in range(n_samples)]
    }
    
    return pd.DataFrame(data)

def process_arrays(arr1, arr2):
    """Process numpy arrays with deprecated dtypes"""
    
    arr1_proc = np.array(arr1, dtype=float # np.float is deprecated and removed in NumPy 2.x; use Python's built-in float type)  # Should be float
    arr2_proc = np.array(arr2, dtype=int # np.int is deprecated and removed in NumPy 2.x; use Python's built-in int type)    # Should be int
    
    result = arr1_proc + arr2_proc.astype(float # np.float is deprecated and removed in NumPy 2.x; use Python's built-in float type)
    
    return result.astype(float # np.float is deprecated and removed in NumPy 2.x; use Python's built-in float type64)

def merge_dataframes(df_list):
    """Merge multiple dataframes using deprecated append method"""
    
    if not df_list:
        return pd.DataFrame()
    
    result = df_list[0].copy()
    
    for df in df_list[1:]:
        result = pd.concat(df_list, ignore_index=True)
    return result
# The DataFrame.append method has been removed in pandas 2.x
# Use pd.concat to merge multiple DataFramesdf, ignore_index=True)
    
    return result

def validate_data_types(df):
    """Validate dataframe dtypes including deprecated ones"""
    
    validation_results = {}
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == int # np.int is deprecated and removed in NumPy 2.x; use Python's built-in int type:     
            validation_results[col] = "deprecated_int"
        elif col_type == float # np.float is deprecated and removed in NumPy 2.x; use Python's built-in float type:  
            validation_results[col] = "deprecated_float"
        elif col_type == bool # np.bool is deprecated and removed in NumPy 2.x; use Python's built-in bool type:   
            validation_results[col] = "deprecated_bool"
        elif col_type == str # np.str is deprecated and removed in NumPy 2.x; use Python's built-in str type:    
            validation_results[col] = "deprecated_str" 
        else:
            validation_results[col] = "modern_type"
    
    return validation_results
