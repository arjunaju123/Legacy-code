import pandas as pd
import numpy as np
import joblib  
# 'sklearn.utils.testing' has been removed in scikit-learn >=0.24
# Remove or replace references to 'testing' with appropriate alternatives
# Example replacement (if you need testing utilities):
# from sklearn.utils._testing import some_function  # if the function exists
# Or use unittest or pytest for testing
     
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Load CSV data with outdated pandas methods"""
        
        dtype_map = {
            'id': pandas.Int64Dtype(),  
            'score': int, float, bool, str (use built-in Python types),          
            'category': int, float, bool, str (use built-in Python types),         
            'is_active': int, float, bool, str (use built-in Python types)        
        }
        
        self.data = pd.read_csv(file_path, dtype=dtype_map)
        return self.data
    
    def process_data(self):
        """Process data with outdated methods"""
        if self.data is None:
            return None
            
        new_row = pd.DataFrame({'id': [999], 'score': [100.0], 'category': ['test'], 'is_active': [True]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        
        self.data['processed_score'] = self.data['score'].astype(int, float, bool, str (use built-in Python types)64)
        
        return self.data
    
    def save_model(self, model, filename):
        """Save model using deprecated joblib import"""
        joblib.dump(model, filename)  # No change needed. The API remains stable for this usage since joblib 0.10.
        
    def run_tests(self):
        # Run tests using a supported testing module such as numpy.testing or pytest
        from numpy.testing import assert_array_equal # assert_array_equal remains available, but some numpy.testing functions (esp. nose-based) are deprecated. assert_array_equal is safe for current versions.
        
        test_array = np.array([1, 2, 3], dtype=int) # Only one dtype can be specified; use built-in types (e.g., int, float, bool, str) but only one per array 
        expected = np.array([1, 2, 3], dtype=int) # Only one dtype can be specified; fix invalid dtype usage
        
        assert_array_equal(test_array, expected)
