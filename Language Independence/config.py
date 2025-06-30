"""
Configuration file for Gemini COT Analysis
"""

import os
from typing import Optional

class Config:
    """Configuration class for the application"""
    
    # API Configuration
    GEMINI_API_KEY: Optional[str] = None
    
    # Dataset Configuration
    NUM_SAMPLES: int = 100
    DATASET_URL: str = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    
    # API Rate Limiting
    API_DELAY: float = 1.0  # seconds between API calls
    MAX_RETRIES: int = 3
    
    # Output Configuration
    OUTPUT_FILENAME: str = "gemini_cot_results.csv"
    
    @classmethod
    def load_api_key(cls) -> str:
        """Load API key from environment variable or prompt user"""
        # Try to get from environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key:
            cls.GEMINI_API_KEY = api_key
            return api_key
        
        # Prompt user for API key
        print("Gemini API key not found in environment variables.")
        print("You can set it by:")
        print("1. Setting GEMINI_API_KEY environment variable")
        print("2. Entering it when prompted")
        print()
        
        api_key = input("Please enter your Gemini API key: ").strip()
        
        if not api_key:
            raise ValueError("API key is required to run the analysis")
        
        cls.GEMINI_API_KEY = api_key
        return api_key
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get the API key, loading it if necessary"""
        if cls.GEMINI_API_KEY is None:
            return cls.load_api_key()
        return cls.GEMINI_API_KEY 