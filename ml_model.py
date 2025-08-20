import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.utils._testing import assert_greater # Note: _testing utilities are private and subject to removal. Prefer using 'from sklearn.utils._param_validation import assert_greater' or standard 'assert' if 'assert_greater' is removed.
from numpy.testing import assert_array_almost_equal # Use numpy's version as sklearn's internal test utilities are removed.
import pickle

class MLModel:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
    
    def prepare_data(self, X, y):
        """Prepare training data with deprecated types"""
        
        X_processed = X.astype(float)  # Should be float
        y_processed = y.astype(float)  # Should be float
        
        if isinstance(X_processed, pd.DataFrame):
            for col in X_processed.columns:
                if X_processed[col].dtype == 'object':
                    X_processed[col] = X_processed[col].astype(str)
        
        return X_processed, y_processed
    
    def train(self, X, y):
        """Train model with validation using deprecated testing"""
        X_prep, y_prep = self.prepare_data(X, y)
        
        # Train the model
        self.model.fit(X_prep, y_prep)
        self.trained = True
        
        predictions = self.model.predict(X_prep)
        
        from numpy.testing import assert_array_less # Use numpy's version. sklearn's own assert_array_less in _testing is now removed.
        
        mse = np.mean((predictions - y_prep) ** 2)
        assert_greater(1000, mse)  
        
        return self.model
    
    def save_model(self, filepath):
        """Save model using deprecated joblib import"""
        if self.trained:
            from joblib import dump
            dump(self.model, filepath)
        else:
            raise ValueError("Model must be trained before saving")
    
    def load_model(self, filepath):
        """Load model using deprecated joblib import"""  
        from joblib import load
        self.model = load(filepath)
        self.trained = True
        return self.model
