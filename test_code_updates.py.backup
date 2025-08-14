import unittest
from main import DataProcessor
from data_analysis import DataAnalyzer  
from ml_model import MLModel
from utils import create_sample_data, process_arrays, merge_dataframes

class TestCodeUpdates(unittest.TestCase):
    
    def setUp(self):
        self.sample_data = create_sample_data(100)
        
    def test_data_processor(self):
        """Test DataProcessor with deprecated code patterns"""
        processor = DataProcessor()
        
        processed = processor.process_data()
        
        self.assertIsNotNone(processed)
    
    def test_deprecated_numpy_usage(self):
        """Test code that uses deprecated numpy dtypes"""
        
        arr1 = np.array([1, 2, 3], dtype=np.int)   
        arr2 = np.array([1.0, 2.0, 3.0], dtype=np.float) 
        
        result = process_arrays(arr1, arr2)
        self.assertEqual(len(result), 3)
    
    def test_pandas_append_usage(self):
        """Test deprecated pandas append method"""
        
        df_list = [
            self.sample_data.head(10),
            self.sample_data.tail(10) 
        ]
        
        merged = merge_dataframes(df_list)
        self.assertEqual(len(merged), 20)

if __name__ == "__main__":
    unittest.main()
