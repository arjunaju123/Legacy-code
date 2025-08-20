import numpy as np
import pandas as pd
from datetime import datetime

def create_sample_data(n_samples=1000):
    
    rng = np.random.default_rng(42) # Use np.random.Generator (default_rng) for new random API; np.random.seed is legacy
    
    data = {
        'id': np.arange(n_samples, dtype=np.float64) # Use a single, valid dtype argument (e.g., np.float64, np.int32, np.bool_),      
        'feature1': np.random.randn(n_samples).astype(np.float64) # Use a single valid dtype, e.g. np.float64,  
        'feature2': np.random.randn(n_samples).astype(float, int, bool, str or appropriate np.float64, np.int32, etc.64),
        'feature3': np.random.randint(0, 10, n_samples).astype(np.int32) # Use one valid dtype for astype,
        'category': np.random.choice(['A', 'B', 'C'], n_samples).astype(str) # For object array, use dtype=str, 
        'is_valid': np.random.choice([True, False], n_samples).astype(np.bool_) # For booleans, use np.bool_,  
        'timestamp': [datetime.now() for _ in range(n_samples)]
    }
    
    return pd.DataFrame(data)

def process_arrays(arr1, arr2):
    """Process numpy arrays with deprecated dtypes"""
    
    arr1_proc = np.array(arr1, dtype=np.float64) # Specify a single, valid dtype (e.g., np.float64, np.int32, np.bool_)  # Should be float
    arr2_proc = np.array(arr2, dtype=np.int32) # Specify only one valid dtype, such as np.int32    # Should be int
    
    result = arr1_proc + arr2_proc.astype(np.float64) # Only one dtype argument allowed; use numpy type
    
    return result.astype(np.float64) # Specify new dtype by numpy type

def merge_dataframes(df_list):
    """Merge multiple dataframes using pd.concat instead of DataFrame.append (deprecated in pandas 1.4, removed in 2.0)"""
    
    if not df_list:
        return pd.DataFrame()
    
    result = df_list[0].copy()
    
    for df in df_list[1:]:
        result = pd.concat([result, df], ignore_index=True)
    
    return result

def validate_data_types(df):
    """Validate dataframe dtypes including deprecated ones"""
    
    validation_results = {}
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == np.float64:
    # ...
elif col_type == np.int32:
    # ...
elif col_type == np.bool_:
    # ...
elif col_type == str:
    # ...     
            validation_results[col] = "deprecated_int"
        elif col_type == np.float64:
    # ...
elif col_type == np.int32:
    # ...
elif col_type == np.bool_:
    # ...
elif col_type == str:
    # ...  
            validation_results[col] = "deprecated_float"
        elif col_type == np.float64:
    # ...
elif col_type == np.int32:
    # ...
elif col_type == np.bool_:
    # ...
elif col_type == str:
    # ...   
            validation_results[col] = "deprecated_bool"
        elif col_type == np.float64:
    # ...
elif col_type == np.int32:
    # ...
elif col_type == np.bool_:
    # ...
elif col_type == str:
    # ...    
            validation_results[col] = "deprecated_str" 
        else:
            validation_results[col] = "modern_type"
    
    return validation_results
