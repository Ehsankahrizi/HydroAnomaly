"""
Tests for the hello module
"""
import unittest
from hydroanomaly.hello import greet, say_goodbye


class TestHello(unittest.TestCase):
    
    def test_greet_default(self):
        """Test greeting with default name"""
        result = greet()
        self.assertEqual(result, "Hello, World!")
    
    def test_greet_with_name(self):
        """Test greeting with custom name"""
        result = greet("Alice")
        self.assertEqual(result, "Hello, Alice!")
    
    def test_say_goodbye_default(self):
        """Test goodbye with default name"""
        result = say_goodbye()
        self.assertEqual(result, "Goodbye, World!")
    
    def test_say_goodbye_with_name(self):
        """Test goodbye with custom name"""
        result = say_goodbye("Bob")
        self.assertEqual(result, "Goodbye, Bob!")


if __name__ == "__main__":
    unittest.main()
