import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_array_almost_equal
import pickle

class MLModel:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
    
    def prepare_data(self, X, y):
        """Prepare training data with deprecated types"""
        
        X_processed = X.astype(np.float)  # Should be float
        y_processed = y.astype(np.float)  # Should be float
        
        if isinstance(X_processed, pd.DataFrame):
            for col in X_processed.columns:
                if X_processed[col].dtype == 'object':
                    X_processed[col] = X_processed[col].astype(np.str)
        
        return X_processed, y_processed
    
    def train(self, X, y):
        """Train model with validation using deprecated testing"""
        X_prep, y_prep = self.prepare_data(X, y)
        
        # Train the model
        self.model.fit(X_prep, y_prep)
        self.trained = True
        
        predictions = self.model.predict(X_prep)
        
        from sklearn.utils.testing import assert_array_less
        
        mse = np.mean((predictions - y_prep) ** 2)
        assert_greater(1000, mse)  
        
        return self.model
    
    def save_model(self, filepath):
        """Save model using deprecated joblib import"""
        if self.trained:
            from sklearn.externals.joblib import dump
            dump(self.model, filepath)
        else:
            raise ValueError("Model must be trained before saving")
    
    def load_model(self, filepath):
        """Load model using deprecated joblib import"""  
        from sklearn.externals.joblib import load
        self.model = load(filepath)
        self.trained = True
        return self.model
