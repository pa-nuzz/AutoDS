"""User-friendly error handling with actionable messages.

Converts technical errors into helpful, actionable guidance for users.
"""
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ErrorResponse:
    """Structured error response for users."""
    title: str
    message: str
    suggestion: str
    category: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    action: Optional[str] = None


class ErrorHandler:
    """Converts exceptions into user-friendly error messages."""
    
    # Common error patterns and their friendly translations
    ERROR_PATTERNS = {
        # File upload errors
        r'File too large': ErrorResponse(
            title="File Too Large",
            message="The file you're trying to upload exceeds our size limit.",
            suggestion="Please upload a file smaller than 1GB, or compress your data before uploading.",
            category="file_size",
            severity="warning"
        ),
        r'Executable.*detected': ErrorResponse(
            title="File Not Allowed",
            message="We detected that this file might be an executable or script.",
            suggestion="Please upload only data files (CSV, Excel, JSON, images, audio). For security reasons, we don't allow executable files.",
            category="security",
            severity="error"
        ),
        r'File extension.*not allowed': ErrorResponse(
            title="Unsupported File Format",
            message="This file type isn't supported yet.",
            suggestion="We support: CSV, Excel (.xlsx), JSON, Parquet, images, audio, and archives. Please convert your data to one of these formats.",
            category="file_format",
            severity="warning"
        ),
        r'Path traversal': ErrorResponse(
            title="Invalid Filename",
            message="The filename contains characters that could be unsafe.",
            suggestion="Please rename your file to only use letters, numbers, and basic punctuation (avoiding ../ or special characters).",
            category="security",
            severity="warning"
        ),
        
        # Data errors
        r'No columns to parse': ErrorResponse(
            title="Empty or Invalid File",
            message="We couldn't find any data in this file.",
            suggestion="Please check that your file isn't empty and has properly formatted columns. Try opening it in a spreadsheet program first.",
            category="data",
            severity="error"
        ),
        r'UTF-8.*codec.*decode': ErrorResponse(
            title="Character Encoding Issue",
            message="We couldn't read this file due to special characters.",
            suggestion="Try saving your file as UTF-8 encoded CSV. In Excel: File → Save As → CSV UTF-8.",
            category="encoding",
            severity="warning"
        ),
        r'missing.*values|can.*t.*convert': ErrorResponse(
            title="Data Format Issue",
            message="Some data couldn't be parsed correctly.",
            suggestion="Check for empty cells, mixed data types in columns, or text in numeric columns. Clean your data in a spreadsheet first.",
            category="data",
            severity="warning"
        ),
        
        # URL/download errors
        r'Invalid URL': ErrorResponse(
            title="Invalid URL",
            message="The URL you provided doesn't seem valid.",
            suggestion="Please check the URL and make sure it starts with https://. We support Kaggle, GitHub, Google Drive, and direct file links.",
            category="url",
            severity="warning"
        ),
        r'Blocked host': ErrorResponse(
            title="URL Not Allowed",
            message="This URL points to a restricted location.",
            suggestion="Please use a public URL (not localhost or private network). Try uploading the file directly instead.",
            category="security",
            severity="warning"
        ),
        r'404|Not Found': ErrorResponse(
            title="File Not Found",
            message="We couldn't access the file at this URL.",
            suggestion="Please verify the URL is correct and the file is publicly accessible. For Kaggle, use the dataset page URL.",
            category="url",
            severity="warning"
        ),
        r'403|Forbidden': ErrorResponse(
            title="Access Denied",
            message="We don't have permission to access this file.",
            suggestion="The file might be private or require authentication. Try downloading it manually and uploading directly.",
            category="url",
            severity="warning"
        ),
        
        # API/LLM errors
        r'API key.*invalid|Unauthorized': ErrorResponse(
            title="API Authentication Failed",
            message="We couldn't connect to the AI service.",
            suggestion="This might be a temporary issue. We're using fallback options. If problems persist, AI-enhanced features will be limited.",
            category="api",
            severity="warning"
        ),
        r'Rate limit|Too Many Requests': ErrorResponse(
            title="Service Busy",
            message="The AI service is experiencing high traffic.",
            suggestion="Please wait a moment and try again. We're automatically retrying with backup providers.",
            category="api",
            severity="info"
        ),
        r'No API keys available': ErrorResponse(
            title="AI Features Unavailable",
            message="No AI providers are currently configured.",
            suggestion="The app will work without AI features. Basic analysis and modeling are still available. Contact the admin to add API keys.",
            category="api",
            severity="warning"
        ),
        
        # Memory/performance errors
        r'MemoryError|out of memory': ErrorResponse(
            title="File Too Large for Processing",
            message="This file is too big to process in memory.",
            suggestion="Try uploading a smaller sample (first 10,000 rows), or use a machine with more RAM. Consider processing in chunks.",
            category="performance",
            severity="error"
        ),
        r'timeout|timed out': ErrorResponse(
            title="Processing Timeout",
            message="This operation took too long to complete.",
            suggestion="Try with a smaller dataset first, or use the 'Auto Mode' which optimizes for your data size. You can also try again later.",
            category="performance",
            severity="warning"
        ),
        
        # Data processing errors
        r'target.*not found|column.*not found': ErrorResponse(
            title="Column Not Found",
            message="The specified column doesn't exist in your data.",
            suggestion="Please check the column name and try again. Column names are case-sensitive.",
            category="data",
            severity="warning"
        ),
        r'insufficient.*data|need.*more': ErrorResponse(
            title="Not Enough Data",
            message="Your dataset is too small for this analysis.",
            suggestion="We need at least 10-20 rows for meaningful analysis. Please upload a larger dataset or use the sample data.",
            category="data",
            severity="warning"
        ),
    }
    
    @classmethod
    def handle(cls, exception: Exception, context: Optional[str] = None) -> ErrorResponse:
        """Convert an exception to a user-friendly response."""
        error_str = str(exception)
        error_type = type(exception).__name__
        
        # Try to match known patterns
        for pattern, response in cls.ERROR_PATTERNS.items():
            if re.search(pattern, error_str, re.IGNORECASE):
                return response
        
        # Fallback based on exception type
        if 'SecurityError' in error_type:
            return ErrorResponse(
                title="Security Check Failed",
                message="We couldn't verify this file is safe.",
                suggestion="Please check the file isn't corrupted or malicious. If you're sure it's safe, try a different format.",
                category="security",
                severity="error"
            )
        elif 'Connection' in error_type or 'HTTP' in error_type:
            return ErrorResponse(
                title="Connection Issue",
                message="We had trouble connecting to download the file.",
                suggestion="Please check your internet connection and try again. If the problem persists, download the file manually and upload it.",
                category="network",
                severity="warning"
            )
        elif 'Parser' in error_type or 'Decode' in error_type:
            return ErrorResponse(
                title="File Format Error",
                message="We couldn't read this file format correctly.",
                suggestion="The file might be corrupted or in an unexpected format. Try opening it in the original application and re-saving it.",
                category="file_format",
                severity="warning"
            )
        elif 'Permission' in error_type:
            return ErrorResponse(
                title="Permission Denied",
                message="We don't have permission to access this file or folder.",
                suggestion="Please check file permissions or choose a different location. You might need to run with different permissions.",
                category="system",
                severity="error"
            )
        
        # Generic fallback
        return ErrorResponse(
            title="Something Went Wrong",
            message=f"We encountered an unexpected issue: {error_str[:100]}",
            suggestion="Please try again. If the problem persists, try a different file or contact support with this error message.",
            category="unknown",
            severity="error"
        )
    
    @classmethod
    def format_for_streamlit(cls, exception: Exception, context: Optional[str] = None) -> str:
        """Format error as HTML for Streamlit display."""
        response = cls.handle(exception, context)
        
        severity_colors = {
            'info': ('#dbeafe', '#3b82f6'),
            'warning': ('#fef3c7', '#f59e0b'),
            'error': ('#fee2e2', '#ef4444'),
            'critical': ('#fecaca', '#dc2626')
        }
        
        bg_color, border_color = severity_colors.get(response.severity, severity_colors['error'])
        icon = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'critical': '🚨'}.get(response.severity, '⚠️')
        
        return f"""
        <div style="
            background: linear-gradient(135deg, {bg_color} 0%, white 100%);
            border-left: 4px solid {border_color};
            border-radius: 12px;
            padding: 1.25rem;
            margin: 1rem 0;
        ">
            <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">
                {icon} {response.title}
            </div>
            <div style="color: #475569; margin-bottom: 0.75rem;">
                {response.message}
            </div>
            <div style="
                background: white;
                border-radius: 8px;
                padding: 0.75rem;
                color: #64748b;
                font-size: 0.95rem;
            ">
                <strong>💡 How to fix:</strong> {response.suggestion}
            </div>
        </div>
        """
    
    @classmethod
    def format_short(cls, exception: Exception) -> str:
        """Get just a short user-friendly message."""
        response = cls.handle(exception)
        return f"{response.title}: {response.suggestion}"


# Decorator for automatic error handling
def user_friendly_errors(func):
    """Decorator that catches exceptions and converts to user-friendly messages."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the technical error for debugging
            import logging
            logging.exception(f"Error in {func.__name__}: {e}")
            
            # Re-raise as a user-friendly exception
            response = ErrorHandler.handle(e)
            raise UserFriendlyException(response) from e
    return wrapper


class UserFriendlyException(Exception):
    """Exception that contains a user-friendly error response."""
    
    def __init__(self, response: ErrorResponse):
        self.response = response
        super().__init__(f"{response.title}: {response.message}")
    
    def to_html(self) -> str:
        """Convert to HTML for display."""
        severity_colors = {
            'info': ('#dbeafe', '#3b82f6'),
            'warning': ('#fef3c7', '#f59e0b'),
            'error': ('#fee2e2', '#ef4444'),
            'critical': ('#fecaca', '#dc2626')
        }
        
        bg_color, border_color = severity_colors.get(self.response.severity, severity_colors['error'])
        icon = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌', 'critical': '🚨'}.get(self.response.severity, '⚠️')
        
        return f"""
        <div style="
            background: linear-gradient(135deg, {bg_color} 0%, white 100%);
            border-left: 4px solid {border_color};
            border-radius: 12px;
            padding: 1.25rem;
            margin: 1rem 0;
        ">
            <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">
                {icon} {self.response.title}
            </div>
            <div style="color: #475569; margin-bottom: 0.75rem;">
                {self.response.message}
            </div>
            <div style="
                background: white;
                border-radius: 8px;
                padding: 0.75rem;
                color: #64748b;
                font-size: 0.95rem;
            ">
                <strong>💡 How to fix:</strong> {self.response.suggestion}
            </div>
        </div>
        """
