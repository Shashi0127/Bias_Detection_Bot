"""
Configuration settings for the Bias Detection Bot.
Central configuration file for all constants, thresholds, and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DATA_DIR = DATA_DIR / "training_data"
TEMP_DIR = DATA_DIR / "temp"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [MODELS_DIR, TRAINING_DATA_DIR, TEMP_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Environment and API Keys
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# File Constraints
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
MAX_ROWS = int(os.getenv("MAX_ROWS", "50000"))
MIN_ROWS = 10
ALLOWED_EXTENSIONS = [".csv"]

# User Limits
DAILY_ANALYSIS_LIMIT = int(os.getenv("DAILY_ANALYSIS_LIMIT", "10"))

# Protected Attributes Patterns
# These patterns help identify demographic columns in any dataset
PROTECTED_ATTRIBUTES = [
    "race", "ethnicity", "ethnic", "racial",
    "gender", "sex", "male", "female",
    "age", "birth", "dob", "year_born",
    "religion", "religious", "faith",
    "disability", "disabled", "handicap",
    "national", "nationality", "citizen",
    "marital", "married", "single",
    "pregnancy", "pregnant", "maternity",
    "veteran", "military", "service"
]

# Outcome Variable Patterns
# These patterns help identify decision/outcome columns
OUTCOME_PATTERNS = [
    "hired", "hire", "selected", "select",
    "approved", "approve", "accepted", "accept",
    "admitted", "admit", "granted", "grant",
    "passed", "pass", "failed", "fail",
    "qualified", "qualify", "eligible",
    "promoted", "promote", "rejected", "reject",
    "outcome", "decision", "result", "status",
    "label", "target", "class", "prediction",
    "risk", "score", "rating", "rank"
]

# Bias Detection Thresholds
BIAS_THRESHOLD = float(os.getenv("BIAS_THRESHOLD", "0.2"))  # 20% disparity
DISPARATE_IMPACT_THRESHOLD = 0.8  # 80% rule (EEOC guideline)
STATISTICAL_SIGNIFICANCE = 0.05  # p-value threshold
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Risk Levels
RISK_LEVELS = {
    "low": {"min": 0, "max": 0.2, "description": "Minimal bias detected"},
    "medium": {"min": 0.2, "max": 0.5, "description": "Moderate bias requiring attention"},
    "high": {"min": 0.5, "max": 1.0, "description": "Significant bias needing immediate action"}
}

# Model Configuration
MODEL_NAME_PREFIX = "bias_detector"
ENSEMBLE_MODELS = ["gradient_boosting", "random_forest", "logistic_regression"]
TEST_SPLIT_RATIO = 0.3  # 30% for testing, 70% for training
CROSS_VALIDATION_FOLDS = 5
RANDOM_STATE = 42

# Training Configuration
MIN_SAMPLES_PER_CLASS = 100
SYNTHETIC_DATASETS_COUNT = 1000
SAMPLES_PER_DATASET = 1000
BIAS_INJECTION_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# LLM Configuration
LLM_MODEL = "mixtral-8x7b-32768"
LLM_MAX_TOKENS = 1000
LLM_TEMPERATURE = 0.7
LLM_TIMEOUT = 30  # seconds

# Visualization Settings
CHART_WIDTH = 10
CHART_HEIGHT = 6
CHART_DPI = 100
CHART_FORMAT = "png"

# Logging Configuration
LOG_LEVEL = "INFO" if not DEBUG else "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_risk_level(bias_percentage):
    """
    Determine risk level based on bias percentage.
    
    Args:
        bias_percentage: Float between 0 and 1
        
    Returns:
        String risk level (low/medium/high)
    """
    for level, config in RISK_LEVELS.items():
        if config["min"] <= bias_percentage < config["max"]:
            return level
    return "high"

def validate_configuration():
    """
    Validate that all required configuration is present.
    
    Raises:
        ValueError: If required configuration is missing
    """
    errors = []
    
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        errors.append("TELEGRAM_BOT_TOKEN not configured")
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        errors.append("GROQ_API_KEY not configured")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

if __name__ == "__main__":
    print("Configuration Settings")
    print("-" * 50)
    print(f"Environment: {ENVIRONMENT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Max File Size: {MAX_FILE_SIZE_MB} MB")
    print(f"Bias Threshold: {BIAS_THRESHOLD * 100}%")
    print(f"Protected Attributes: {len(PROTECTED_ATTRIBUTES)} patterns")
    print(f"Outcome Patterns: {len(OUTCOME_PATTERNS)} patterns")
    
    try:
        validate_configuration()
        print("\nConfiguration validation: PASSED")
    except ValueError as e:
        print(f"\nConfiguration validation: FAILED")
        print(f"Error: {e}")