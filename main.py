import pandas as pd
import numpy as np
import joblib  # joblib should be imported directly, as it has been removed from sklearn.externals since 0.21.0  
# 'sklearn.utils.testing' has been removed. There is no direct replacement. Remove this import and refactor any code relying on 'testing'.     
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Load CSV data with outdated pandas methods"""
        
        dtype_map = {
            'id': pandas.Int64Dtype(),  
            'score': float # np.float is deprecated since NumPy 1.20. Use built-in float or np.float64.,          
            'category': str # np.str is deprecated since NumPy 1.20. Use built-in str or np.str_.,         
            'is_active': bool # bool # np.bool is deprecated since NumPy 1.20. Use built-in bool or np.bool_. is deprecated and removed from numpy; use Python's builtin bool.        
        }
        
        self.data = pd.read_csv(file_path, dtype=dtype_map)
        return self.data
    
    def process_data(self):
        """Process data with outdated methods"""
        if self.data is None:
            return None
            
        new_row = pd.DataFrame({'id': [999], 'score': [100.0], 'category': ['test'], 'is_active': [True]})
        self.data = pd.concat([self.data, new_row], ignore_index=True) # .append is removed in pandas>=2.0
        
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns # int # np.int is deprecated since NumPy 1.20. Use built-in int or np.int32/np.int64 as appropriate./float # np.float is deprecated since NumPy 1.20. Use built-in float or np.float64. are deprecated everywhere in NumPy >1.20 and unsupported in pandas >=2.0
        
        self.data['processed_score'] = self.data['score'].astype(float # np.float is deprecated since NumPy 1.20. Use built-in float or np.float64.64)
        
        return self.data
    
    def save_model(self, model, filename):
        """Save model using deprecated joblib import"""
        joblib.dump(model, filename)
        
    def run_tests(self):
        """Run tests using deprecated testing module"""
        from numpy.testing import assert_array_equal  # Use numpy's assert_array_equal; sklearn.utils.testing is removed.
        
        test_array = np.array([1, 2, 3], dtype=int # np.int is deprecated since NumPy 1.20. Use built-in int or np.int32/np.int64 as appropriate.) 
        expected = np.array([1, 2, 3], dtype=int # np.int is deprecated since NumPy 1.20. Use built-in int or np.int32/np.int64 as appropriate.64)
        
        assert_array_equal(test_array, expected)
