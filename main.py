import pandas as pd
import numpy as np
import joblib  # sklearn no longer provides joblib under sklearn.externals; import joblib directly  
# No direct replacement; sklearn.utils.testing has been removed
# Remove this import or replace with custom test helpers as needed     
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Load CSV data with outdated pandas methods"""
        
        dtype_map = {
            'id': pandas.Int64Dtype(),  
            'score': float  # np.float is removed in numpy >= 1.20. Use standard Python float,          
            'category': str  # np.str is removed in numpy >= 1.20. Use standard Python str,         
            'is_active': bool  # np.bool is removed in numpy >= 1.20. Use standard Python bool        
        }
        
        self.data = pd.read_csv(file_path, dtype=dtype_map)
        return self.data
    
    def process_data(self):
        """Process data with outdated methods"""
        if self.data is None:
            return None
            
        new_row = pd.DataFrame({'id': [999], 'score': [100.0], 'category': ['test'], 'is_active': [True]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)  # Replaces deprecated DataFrame.append (removed in pandas 2.0)
        
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns  # int  # np.int is removed in numpy >= 1.20. Use standard Python int and float  # np.float is removed in numpy >= 1.20. Use standard Python float are deprecated/removed
        
        self.data['processed_score'] = self.data['score'].astype(float  # np.float is removed in numpy >= 1.20. Use standard Python float64)
        
        return self.data
    
    def save_model(self, model, filename):
        """Save model using deprecated joblib import"""
        joblib.dump(model, filename)
        
    def run_tests(self):
        """Run tests using deprecated testing module"""
        from numpy.testing import assert_array_equal  # Use numpy.testing.assert_array_equal instead; sklearn.utils.testing has been removed
        
        test_array = np.array([1, 2, 3], dtype=int  # np.int is removed in numpy >= 1.20. Use standard Python int) 
        expected = np.array([1, 2, 3], dtype=int  # np.int is removed in numpy >= 1.20. Use standard Python int64)
        
        assert_array_equal(test_array, expected)
