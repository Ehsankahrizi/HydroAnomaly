"""
Tests for the USGS data module
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from hydroanomaly.usgs_data import USGSDataRetriever, get_usgs_data


class TestUSGSDataRetriever(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.retriever = USGSDataRetriever()
        self.sample_site = "08158000"
        self.sample_param = "00060"
        self.sample_start = "2023-01-01"
        self.sample_end = "2023-01-02"
    
    def test_init(self):
        """Test USGSDataRetriever initialization"""
        self.assertEqual(self.retriever.base_url, "https://waterservices.usgs.gov/nwis/iv/")
        self.assertIsNone(self.retriever.last_request_url)
        self.assertIsNone(self.retriever.last_response)
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs"""
        # Should not raise any exception
        try:
            self.retriever._validate_inputs(
                self.sample_site, 
                self.sample_param, 
                self.sample_start, 
                self.sample_end
            )
        except Exception:
            self.fail("_validate_inputs raised an exception with valid inputs")
    
    def test_validate_inputs_invalid_site(self):
        """Test input validation with invalid site number"""
        with self.assertRaises(ValueError):
            self.retriever._validate_inputs("", self.sample_param, self.sample_start, self.sample_end)
    
    def test_validate_inputs_invalid_dates(self):
        """Test input validation with invalid dates"""
        with self.assertRaises(ValueError):
            self.retriever._validate_inputs(self.sample_site, self.sample_param, "invalid-date", self.sample_end)
        
        with self.assertRaises(ValueError):
            self.retriever._validate_inputs(self.sample_site, self.sample_param, "2023-01-02", "2023-01-01")
    
    def test_build_url(self):
        """Test URL building"""
        url = self.retriever._build_url(self.sample_site, self.sample_param, self.sample_start, self.sample_end)
        expected_url = (
            f"https://waterservices.usgs.gov/nwis/iv/?sites={self.sample_site}"
            f"&parameterCd={self.sample_param}&startDT={self.sample_start}&endDT={self.sample_end}&format=rdb"
        )
        self.assertEqual(url, expected_url)
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation"""
        synthetic_data = self.retriever._create_synthetic_data(
            self.sample_start, 
            self.sample_end, 
            self.sample_param
        )
        
        self.assertIsInstance(synthetic_data, pd.DataFrame)
        self.assertIn('datetime', synthetic_data.columns)
        self.assertIn('value', synthetic_data.columns)
        self.assertTrue(len(synthetic_data) > 0)
        self.assertTrue(all(synthetic_data['value'] >= 0))  # Values should be positive
    
    def test_process_response_no_data(self):
        """Test processing response with no data"""
        no_data_response = "No sites found matching criteria"
        result = self.retriever._process_response(no_data_response, self.sample_param)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
        self.assertListEqual(list(result.columns), ["datetime", "value"])
    
    @patch('requests.get')
    def test_retrieve_data_success(self, mock_get):
        """Test successful data retrieval"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """# Sample USGS data
# Comments here
agency_cd	site_no	datetime	01234_00060	01234_00060_cd
USGS	08158000	2023-01-01 00:00	100.0	A
USGS	08158000	2023-01-01 01:00	105.0	A"""
        mock_get.return_value = mock_response
        
        # Override _process_response to return test data
        test_data = pd.DataFrame({
            'datetime': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00']),
            'value': [100.0, 105.0]
        })
        
        with patch.object(self.retriever, '_process_response', return_value=test_data):
            result = self.retriever.retrieve_data(
                self.sample_site, 
                self.sample_param, 
                self.sample_start, 
                self.sample_end,
                create_synthetic=False
            )
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertIn('datetime', result.columns)
            self.assertIn('value', result.columns)
    
    @patch('requests.get')
    def test_retrieve_data_http_error(self, mock_get):
        """Test data retrieval with HTTP error"""
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_get.return_value = mock_response
        
        result = self.retriever.retrieve_data(
            self.sample_site, 
            self.sample_param, 
            self.sample_start, 
            self.sample_end,
            create_synthetic=True  # Should create synthetic data
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(len(result) > 0)  # Should have synthetic data
    
    def test_save_data(self):
        """Test data saving functionality"""
        test_data = pd.DataFrame({
            'datetime': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'value': [10.5, 12.3]
        })
        
        filename = "test_output.csv"
        result_filename = self.retriever.save_data(test_data, filename, "TestParameter")
        
        self.assertEqual(result_filename, filename)
        
        # Check if file was created and has correct content
        import os
        self.assertTrue(os.path.exists(filename))
        
        # Read the file back
        saved_data = pd.read_csv(filename)
        self.assertIn('TestParameter', saved_data.columns)
        self.assertIn('datetime', saved_data.columns)
        self.assertIn('date', saved_data.columns)
        
        # Clean up
        os.remove(filename)
    
    def test_get_data_summary(self):
        """Test data summary generation"""
        test_data = pd.DataFrame({
            'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'value': [10.0, 20.0, 15.0]
        })
        
        summary = self.retriever.get_data_summary(test_data)
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['record_count'], 3)
        self.assertIn('date_range', summary)
        self.assertIn('value_stats', summary)
        self.assertEqual(summary['value_stats']['min'], 10.0)
        self.assertEqual(summary['value_stats']['max'], 20.0)
        self.assertEqual(summary['value_stats']['mean'], 15.0)
    
    def test_get_data_summary_empty(self):
        """Test data summary with empty data"""
        empty_data = pd.DataFrame(columns=['datetime', 'value'])
        summary = self.retriever.get_data_summary(empty_data)
        
        self.assertIn('error', summary)


class TestConvenienceFunction(unittest.TestCase):
    
    @patch('hydroanomaly.usgs_data.USGSDataRetriever')
    def test_get_usgs_data(self, mock_retriever_class):
        """Test the convenience function"""
        # Mock the retriever instance
        mock_retriever = MagicMock()
        mock_data = pd.DataFrame({
            'datetime': pd.to_datetime(['2023-01-01']),
            'value': [100.0]
        })
        mock_retriever.retrieve_data.return_value = mock_data
        mock_retriever_class.return_value = mock_retriever
        
        result = get_usgs_data("08158000", "00060", "2023-01-01", "2023-01-02")
        
        self.assertIsInstance(result, pd.DataFrame)
        mock_retriever.retrieve_data.assert_called_once()
        mock_retriever_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()
