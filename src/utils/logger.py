"""
Logging configuration for the Bias Detection Bot.
Provides centralized logging with file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config.settings import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, LOGS_DIR

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class BiasDetectionLogger:
    """
    Custom logger for the bias detection system.
    Creates separate log files for different components.
    """
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name, log_file=None):
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name (usually module name)
            log_file: Optional specific log file name
            
        Returns:
            Logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_path = LOGS_DIR / log_file
        else:
            file_path = LOGS_DIR / f"{name}.log"
        
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger

def get_logger(name, log_file=None):
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        log_file: Optional log file name
        
    Returns:
        Logger instance
    """
    return BiasDetectionLogger.get_logger(name, log_file)

def log_analysis(user_id, file_name, results):
    """
    Log bias analysis results.
    
    Args:
        user_id: User identifier
        file_name: Analyzed file name
        results: Analysis results dictionary
    """
    logger = get_logger('analysis')
    logger.info(f"Analysis completed - User: {user_id}, File: {file_name}, "
                f"Bias: {results.get('bias_detected', 'unknown')}, "
                f"Confidence: {results.get('confidence', 0):.2f}")

def log_error(component, error, context=None):
    """
    Log error with context.
    
    Args:
        component: Component where error occurred
        error: Exception or error message
        context: Additional context dictionary
    """
    logger = get_logger('errors')
    error_msg = f"Error in {component}: {str(error)}"
    if context:
        error_msg += f", Context: {context}"
    logger.error(error_msg, exc_info=True)

def log_performance(operation, duration, details=None):
    """
    Log performance metrics.
    
    Args:
        operation: Operation name
        duration: Duration in seconds
        details: Additional details dictionary
    """
    logger = get_logger('performance')
    perf_msg = f"{operation} completed in {duration:.2f}s"
    if details:
        perf_msg += f", Details: {details}"
    logger.info(perf_msg)

if __name__ == "__main__":
    # Test logging setup
    test_logger = get_logger("test")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    
    print(f"\nLog files are stored in: {LOGS_DIR}")
    print(f"Log level: {LOG_LEVEL}")
    
    # Test specialized logging functions
    log_analysis("test_user", "test.csv", {"bias_detected": True, "confidence": 0.85})
    log_error("test_component", ValueError("Test error"), {"file": "test.csv"})
    log_performance("test_operation", 1.234, {"rows": 1000})
    
    print("\nLogging test completed. Check log files for output.")