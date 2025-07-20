"""
Tests for the math_utils module
"""
import unittest
from hydroanomaly.math_utils import add, multiply, divide


class TestMathUtils(unittest.TestCase):
    
    def test_add_positive_numbers(self):
        """Test addition of positive numbers"""
        result = add(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test addition of negative numbers"""
        result = add(-2, -3)
        self.assertEqual(result, -5)
    
    def test_multiply_positive_numbers(self):
        """Test multiplication of positive numbers"""
        result = multiply(3, 4)
        self.assertEqual(result, 12)
    
    def test_multiply_by_zero(self):
        """Test multiplication by zero"""
        result = multiply(5, 0)
        self.assertEqual(result, 0)
    
    def test_divide_positive_numbers(self):
        """Test division of positive numbers"""
        result = divide(10, 2)
        self.assertEqual(result, 5.0)
    
    def test_divide_by_zero_raises_error(self):
        """Test that division by zero raises ValueError"""
        with self.assertRaises(ValueError):
            divide(10, 0)


if __name__ == "__main__":
    unittest.main()
