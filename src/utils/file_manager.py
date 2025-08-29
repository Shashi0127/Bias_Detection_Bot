"""
File management utilities for the Bias Detection Bot.
Handles file operations, validation, and cleanup.
"""

import os
import shutil
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

from config.settings import (
    TEMP_DIR, REPORTS_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    MIN_ROWS, MAX_ROWS
)
from src.utils.logger import get_logger, log_error

class FileManager:
    """
    Manages file operations for the bias detection system.
    """
    
    def __init__(self):
        self.logger = get_logger('file_manager')
        self.temp_files = {}  # Track temporary files for cleanup
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate uploaded file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File not found"
            
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE:
                max_mb = MAX_FILE_SIZE / (1024 * 1024)
                return False, f"File too large. Maximum size: {max_mb:.1f} MB"
            
            if file_size == 0:
                return False, "File is empty"
            
            return True, "File is valid"
            
        except Exception as e:
            log_error('file_validation', e, {'file': file_path})
            return False, f"Validation error: {str(e)}"
    
    def save_temp_file(self, file_content: bytes, user_id: str, 
                      original_filename: str) -> Optional[str]:
        """
        Save uploaded file to temporary storage.
        
        Args:
            file_content: File content as bytes
            user_id: User identifier
            original_filename: Original file name
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(file_content).hexdigest()[:8]
            file_ext = Path(original_filename).suffix
            unique_filename = f"{user_id}_{timestamp}_{file_hash}{file_ext}"
            
            # Save file
            file_path = TEMP_DIR / unique_filename
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Track for cleanup
            self.temp_files[str(file_path)] = datetime.now()
            
            self.logger.info(f"Saved temporary file: {unique_filename}")
            return str(file_path)
            
        except Exception as e:
            log_error('save_temp_file', e, {
                'user_id': user_id,
                'filename': original_filename
            })
            return None
    
    def load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load and validate CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame or None if failed
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.logger.info(f"CSV loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Unable to decode CSV with supported encodings")
            
            # Validate data constraints
            if len(df) < MIN_ROWS:
                raise ValueError(f"Dataset too small. Minimum {MIN_ROWS} rows required")
            
            if len(df) > MAX_ROWS:
                self.logger.warning(f"Dataset truncated from {len(df)} to {MAX_ROWS} rows")
                df = df.head(MAX_ROWS)
            
            if len(df.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns")
            
            self.logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            log_error('load_csv', e, {'file': file_path})
            return None
    
    def save_report(self, content: str, user_id: str, 
                   report_type: str = "analysis") -> Optional[str]:
        """
        Save analysis report.
        
        Args:
            content: Report content
            user_id: User identifier
            report_type: Type of report
            
        Returns:
            Path to saved report or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}_{report_type}_{timestamp}.txt"
            file_path = REPORTS_DIR / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Report saved: {filename}")
            return str(file_path)
            
        except Exception as e:
            log_error('save_report', e, {
                'user_id': user_id,
                'report_type': report_type
            })
            return None
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep
            
        Returns:
            Number of files cleaned
        """
        try:
            cleaned = 0
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            
            # Clean tracked files
            for file_path, created_time in list(self.temp_files.items()):
                if created_time < cutoff:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            cleaned += 1
                        del self.temp_files[file_path]
                    except Exception:
                        pass
            
            # Clean any untracked old files in temp directory
            for file_path in TEMP_DIR.glob("*.csv"):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff:
                        file_path.unlink()
                        cleaned += 1
                except Exception:
                    pass
            
            if cleaned > 0:
                self.logger.info(f"Cleaned {cleaned} temporary files")
            
            return cleaned
            
        except Exception as e:
            log_error('cleanup_temp_files', e)
            return 0
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                
                # Remove from tracking if present
                if file_path in self.temp_files:
                    del self.temp_files[file_path]
                
                self.logger.info(f"Deleted file: {file_path}")
                return True
            return False
            
        except Exception as e:
            log_error('delete_file', e, {'file': file_path})
            return False

# Global file manager instance
file_manager = FileManager()

if __name__ == "__main__":
    print("File Manager Test")
    print("-" * 50)
    
    # Test file validation
    test_file = TEMP_DIR / "test.csv"
    
    # Create test CSV
    df = pd.DataFrame({
        'column1': range(20),
        'column2': ['A', 'B'] * 10
    })
    df.to_csv(test_file, index=False)
    
    # Test validation
    is_valid, message = file_manager.validate_file(str(test_file))
    print(f"Validation: {message}")
    
    # Test loading
    loaded_df = file_manager.load_csv(str(test_file))
    if loaded_df is not None:
        print(f"Loaded: {loaded_df.shape}")
    
    # Test cleanup
    cleaned = file_manager.cleanup_temp_files(0)  # Clean all files
    print(f"Cleaned: {cleaned} files")
    
    print("\nFile manager test completed")