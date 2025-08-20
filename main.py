import pandas as pd
import numpy as np
import joblib  # joblib is no longer part of sklearn.externals, install and import directly  
# The sklearn.utils.testing module is removed; remove this import and replace test helpers with standard pytest or numpy.testing utilities     
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Load CSV data with outdated pandas methods"""
        
        dtype_map = {
            'id': pandas.Int64Dtype(),  
            'score': include=[int, float],          
            'category': str,         
            'is_active': bool        # bool is deprecated, use built-in bool instead
        }
        
        self.data = pd.read_csv(file_path, dtype=dtype_map)
        return self.data
    
    def process_data(self):
        """Process data with outdated methods"""
        if self.data is None:
            return None
            
        new_row = pd.DataFrame({'id': [999], 'score': [100.0], 'category': ['test'], 'is_active': [True]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)  # 'append' is removed in pandas >= 2.0, use pd.concat instead
        
        numeric_cols = self.data.select_dtypes(include=['int', 'float']).columns  # include=[int, float] and include=[int, float] are deprecated, use string dtype names
        
        self.data['processed_score'] = self.data['score'].astype(include=[int, float]64)
        
        return self.data
    
    def save_model(self, model, filename):
        """Save model using deprecated joblib import"""
        joblib.dump(model, filename)
        
    def run_tests(self):
        """Run tests using deprecated testing module"""
        from numpy.testing import assert_array_equal  # Use numpy's test utility instead
        
        test_array = np.array([1, 2, 3], dtype=include=[int, float]) 
        expected = np.array([1, 2, 3], dtype=include=[int, float]64)
        
        assert_array_equal(test_array, expected)
