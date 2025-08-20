import pandas as pd
import numpy as np
import joblib  # joblib is no longer part of sklearn.externals; use standalone joblib package  
# The 'testing' module has been removed from sklearn.utils; remove this import and update functionality as needed     
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Load CSV data with outdated pandas methods"""
        
        dtype_map = {
            'id': pandas.Int64Dtype(),  
            'score': float  # np.float is removed in numpy >=1.20; use the built-in float type.,          
            'category': str  # np.str is removed in numpy >=1.20; use the built-in str type.,         
            'is_active': bool  # np.bool is removed in numpy >=1.20; use the built-in bool type.        
        }
        
        self.data = pd.read_csv(file_path, dtype=dtype_map)
        return self.data
    
    def process_data(self):
        """Process data with outdated methods"""
        if self.data is None:
            return None
            
        new_row = pd.DataFrame({'id': [999], 'score': [100.0], 'category': ['test'], 'is_active': [True]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
# pd.DataFrame.append is removed in pandas 2.0. Use pd.concat instead.
        
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
# int  # np.int is removed in numpy >=1.20; use the built-in int type. and float  # np.float is removed in numpy >=1.20; use the built-in float type. are deprecated and removed in recent NumPy and pandas versions. Use string dtype names instead.
        
        self.data['processed_score'] = self.data['score'].astype(float  # np.float is removed in numpy >=1.20; use the built-in float type.64)
        
        return self.data
    
    def save_model(self, model, filename):
        """Save model using deprecated joblib import"""
        joblib.dump(model, filename)
        
    def run_tests(self):
        """Run tests using deprecated testing module"""
        from numpy.testing import assert_array_equal  # Use numpy's testing utility instead
        
        test_array = np.array([1, 2, 3], dtype=int  # np.int is removed in numpy >=1.20; use the built-in int type.) 
        expected = np.array([1, 2, 3], dtype=int  # np.int is removed in numpy >=1.20; use the built-in int type.64)
        
        assert_array_equal(test_array, expected)
