"""
Centralized error handling utilities for consistent error management across the application
"""
import logging
import traceback
from typing import Optional, Dict, Any, Callable
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Base exception for prediction-related errors"""
    pass

class ModelLoadError(PredictionError):
    """Raised when model loading fails"""
    pass

class DataFetchError(PredictionError):
    """Raised when data fetching fails"""
    pass

class ValidationError(PredictionError):
    """Raised when input validation fails"""
    pass

class FeatureEngineeringError(PredictionError):
    """Raised when feature engineering fails"""
    pass

def handle_errors(
    error_message: str = "An error occurred",
    return_value: Any = None,
    log_error: bool = True,
    raise_exception: bool = False
):
    """
    Decorator for consistent error handling
    
    Args:
        error_message: Custom error message
        return_value: Value to return on error (if not raising)
        log_error: Whether to log the error
        raise_exception: Whether to raise exception or return return_value
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                if log_error:
                    logger.warning(f"Validation error in {func.__name__}: {e}")
                if raise_exception:
                    raise
                return return_value
            except (ModelLoadError, DataFetchError, FeatureEngineeringError) as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                if raise_exception:
                    raise
                return return_value
            except Exception as e:
                if log_error:
                    logger.error(f"{error_message} in {func.__name__}: {e}")
                    logger.debug(traceback.format_exc())
                if raise_exception:
                    raise
                return return_value
        return wrapper
    return decorator

def safe_execute(
    func: Callable,
    *args,
    error_message: str = "Error executing function",
    return_value: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Positional arguments
        error_message: Custom error message
        return_value: Value to return on error
        log_error: Whether to log the error
        **kwargs: Keyword arguments
    
    Returns:
        Function result or return_value on error
    """
    try:
        return func(*args, **kwargs)
    except ValidationError as e:
        if log_error:
            logger.warning(f"{error_message}: {e}")
        return return_value
    except (ModelLoadError, DataFetchError, FeatureEngineeringError) as e:
        if log_error:
            logger.error(f"{error_message}: {e}")
        return return_value
    except Exception as e:
        if log_error:
            logger.error(f"{error_message}: {e}")
            logger.debug(traceback.format_exc())
        return return_value

def validate_input(
    value: Any,
    value_name: str,
    validation_func: Optional[Callable] = None,
    error_message: Optional[str] = None
) -> bool:
    """
    Validate input value
    
    Args:
        value: Value to validate
        value_name: Name of the value for error messages
        validation_func: Optional custom validation function
        error_message: Optional custom error message
    
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if value is None:
        raise ValidationError(f"{value_name} cannot be None")
    
    if isinstance(value, str) and not value.strip():
        raise ValidationError(f"{value_name} cannot be empty")
    
    if validation_func and not validation_func(value):
        msg = error_message or f"{value_name} failed validation"
        raise ValidationError(msg)
    
    return True

def safe_float_conversion(value: Any, default: float = 0.0, value_name: str = "value") -> float:
    """
    Safely convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        value_name: Name for error messages
    
    Returns:
        Float value or default
    """
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove commas and whitespace
            cleaned = value.replace(',', '').strip()
            return float(cleaned) if cleaned else default
        return default
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert {value_name} to float: {e}, using default {default}")
        return default

def safe_int_conversion(value: Any, default: int = 0, value_name: str = "value") -> int:
    """
    Safely convert value to int
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        value_name: Name for error messages
    
    Returns:
        Int value or default
    """
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            cleaned = value.replace(',', '').strip()
            return int(float(cleaned)) if cleaned else default
        return default
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert {value_name} to int: {e}, using default {default}")
        return default

def format_error_for_user(error: Exception, context: str = "") -> str:
    """
    Format error message for user display
    
    Args:
        error: Exception that occurred
        context: Additional context
    
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    
    if isinstance(error, ValidationError):
        return f"Invalid input: {str(error)}"
    elif isinstance(error, ModelLoadError):
        return "Model could not be loaded. Please ensure the model exists for this combination."
    elif isinstance(error, DataFetchError):
        return "Could not fetch data. Please try again later."
    elif isinstance(error, FeatureEngineeringError):
        return "Error processing features. Please check your inputs."
    else:
        base_msg = "An error occurred"
        if context:
            base_msg += f" while {context}"
        return f"{base_msg}. Please try again or contact support if the issue persists."

def get_error_details(error: Exception) -> Dict[str, Any]:
    """
    Get detailed error information for logging
    
    Args:
        error: Exception that occurred
    
    Returns:
        Dictionary with error details
    """
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc()
    }

