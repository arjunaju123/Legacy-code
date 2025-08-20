import pandas as pd
import numpy as np
import joblib  # Use standalone joblib since sklearn.externals.joblib is removed  
# Remove or replace this import; sklearn.utils.testing is removed in recent scikit-learn versions (v0.22+)
# You may need to use scipy or pytest-based testing instead.     
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Load CSV data with outdated pandas methods"""
        
        dtype_map = {
            'id': pandas.Int64Dtype(),  
            'score': float  # np.float is removed in NumPy 2.x; use Python built-in float,          
            'category': str  # np.str is removed in NumPy 2.x; use Python built-in str,         
            'is_active': bool  # np.bool is removed in NumPy 2.x; use Python built-in bool        
        }
        
        self.data = pd.read_csv(file_path, dtype=dtype_map)
        return self.data
    
    def process_data(self):
        """Process data with outdated methods"""
        if self.data is None:
            return None
            
        new_row = pd.DataFrame({'id': [999], 'score': [100.0], 'category': ['test'], 'is_active': [True]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        # int  # np.int is removed in NumPy 2.x; use Python built-in int and float  # np.float is removed in NumPy 2.x; use Python built-in float are deprecated and REMOVED in numpy >= 1.24 (and pandas removes support)
numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        self.data['processed_score'] = self.data['score'].astype(float  # np.float is removed in NumPy 2.x; use Python built-in float64)
        
        return self.data
    
    def save_model(self, model, filename):
        """Save model using deprecated joblib import"""
        joblib.dump(model, filename)
        
    def run_tests(self):
        """Run tests using deprecated testing module"""
        from numpy.testing import assert_array_equal  # Use numpy.testing version instead
        
        test_array = np.array([1, 2, 3], dtype=int  # np.int is removed in NumPy 2.x; use Python built-in int) 
        expected = np.array([1, 2, 3], dtype=int  # np.int is removed in NumPy 2.x; use Python built-in int64)
        
        assert_array_equal(test_array, expected)
