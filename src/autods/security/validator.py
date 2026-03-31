"""Security module for input validation and sanitization."""
import logging
import os
import re
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, unquote
from typing import Optional, List, Set
import validators
import pathvalidate

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security validation fails."""
    pass


class InputValidator:
    """Validates and sanitizes all inputs to the AutoDS system."""
    
    # Allowed file extensions for datasets
    ALLOWED_EXTENSIONS: Set[str] = {
        # Tabular
        '.csv', '.xlsx', '.xls', '.json', '.parquet', '.feather', '.hdf5', '.h5',
        # Text
        '.txt', '.md', '.log', '.tsv',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg',
        # Audio
        '.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma',
        # Archives
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar'
    }
    
    # Dangerous patterns in filenames
    DANGEROUS_PATTERNS = [
        r'\.\.',           # Directory traversal
        r'[<>"|?*]',       # Invalid Windows chars
        r'^[\.\s]',        # Hidden files
        r'\.sh$',          # Shell scripts
        r'\.exe$',         # Executables
        r'\.bat$',         # Batch files
        r'\.py[wc]?$',     # Python bytecode
        r'\.(dll|so|dylib)$',  # Libraries
    ]
    
    # Maximum file size (1GB default)
    MAX_FILE_SIZE = 1024 * 1024 * 1024
    
    # Maximum filename length
    MAX_FILENAME_LENGTH = 255
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> bool:
        """Validate if file extension is allowed."""
        ext = Path(filename).suffix.lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise SecurityError(
                f"File extension '{ext}' not allowed. Allowed: {cls.ALLOWED_EXTENSIONS}"
            )
        return True
    
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Validate and sanitize filename."""
        # Decode URL-encoded characters
        filename = unquote(filename)
        
        # Check length
        if len(filename) > cls.MAX_FILENAME_LENGTH:
            raise SecurityError(f"Filename too long: {len(filename)} chars")
        
        # Check dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected in filename: {pattern}")
        
        # Validate with pathvalidate
        try:
            sanitized = pathvalidate.sanitize_filename(filename)
        except Exception as e:
            raise SecurityError(f"Filename sanitization failed: {e}")
        
        # Ensure no path traversal
        if '..' in sanitized or '/' in sanitized or '\\' in sanitized:
            raise SecurityError("Path traversal attempt detected")
        
        return sanitized
    
    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate and sanitize URL."""
        # Basic URL validation
        if not validators.url(url):
            raise SecurityError(f"Invalid URL format: {url}")
        
        parsed = urlparse(url)
        
        # Only allow http/https
        if parsed.scheme not in ('http', 'https'):
            raise SecurityError(f"URL scheme not allowed: {parsed.scheme}. Only http/https are permitted.")
        
        # Block localhost and private IPs
        hostname = parsed.hostname or ''
        blocked_hosts = [
            'localhost', '127.0.0.1', '0.0.0.0',
            '10.', '192.168.', '172.16.', '172.17.', '172.18.',
            '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
            '172.24.', '172.25.', '172.26.', '172.27.', '172.28.',
            '172.29.', '172.30.', '172.31.', '169.254.'
        ]
        
        for blocked in blocked_hosts:
            if hostname.startswith(blocked) or hostname == blocked.rstrip('.'):
                raise SecurityError(f"Blocked host detected: {hostname}")
        
        # Validate file extension in URL path
        path = unquote(parsed.path)
        if '.' in path:
            filename = os.path.basename(path)
            if filename:
                cls.validate_file_extension(filename)
                cls.validate_filename(filename)
        
        return url
    
    @classmethod
    def validate_file_size(cls, file_path: str) -> bool:
        """Validate file size is within limits."""
        size = os.path.getsize(file_path)
        if size > cls.MAX_FILE_SIZE:
            raise SecurityError(
                f"File too large: {size / (1024**3):.2f}GB (max: {cls.MAX_FILE_SIZE / (1024**3):.2f}GB)"
            )
        return True
    
    @classmethod
    def validate_file_content(cls, file_path: str) -> bool:
        """Validate file content is safe (comprehensive check).
        
        Checks for:
        - Executable signatures (Windows, Linux, macOS)
        - Script signatures (PHP, Python, JS, Shell)
        - Archive bombs (zip bombs)
        - Suspicious patterns in text files
        """
        # Check file size first (already done, but double-check)
        size = os.path.getsize(file_path)
        if size > cls.MAX_FILE_SIZE:
            raise SecurityError(
                f"File too large: {size / (1024**3):.2f}GB (max: {cls.MAX_FILE_SIZE / (1024**3):.2f}GB)"
            )
        
        # Read first 1024 bytes for comprehensive header check
        with open(file_path, 'rb') as f:
            header = f.read(1024)
        
        # Executable signatures to block
        exe_signatures = [
            # Windows
            (b'\x4d\x5a', "Windows executable (EXE/DLL)"),
            (b'MZ', "Windows executable (EXE/DLL)"),
            # Linux
            (b'\x7fELF', "Linux executable (ELF)"),
            # macOS
            (b'\xca\xfe\xba\xbe', "macOS executable (Mach-O)"),
            (b'\xfe\xed\xfa\xcf', "macOS executable (Mach-O 64-bit)"),
            (b'\xcf\xfa\xed\xfe', "macOS executable (Mach-O LE)"),
            # Scripts
            (b'#!', "Shell script (shebang)"),
            (b'<?php', "PHP script"),
            (b'<%', "ASP/JSP script"),
            (b'<script', "HTML/JS script"),
            # Java
            (b'\xca\xfe\xba\xbe', "Java class file"),
            # Archives that might contain executables
            (b'PK\x03\x04', "ZIP archive (check contents manually)"),
            (b'Rar!', "RAR archive"),
            (b'7z\xbc\xaf\x27\x1c', "7-Zip archive"),
        ]
        
        for sig, description in exe_signatures:
            if header.startswith(sig):
                # Special handling for ZIP - check if it's a valid data format
                if sig == b'PK\x03\x04':
                    # ZIP is allowed but should be handled carefully
                    # Check for suspicious patterns inside ZIP
                    if cls._is_malicious_zip(file_path):
                        raise SecurityError(
                            f"Potentially malicious archive detected: {description}"
                        )
                    return True
                
                raise SecurityError(
                    f"Executable/script file detected ({description}). "
                    f"Only data files (CSV, Excel, JSON, images, audio) are allowed."
                )
        
        # Check for suspicious strings in text files
        text_suspicious = [
            b'eval(',
            b'exec(',
            b'system(',
            b'os.system',
            b'subprocess.call',
            b'__import__',
            b'import os',
            b'import subprocess',
        ]
        
        # Only check first 10KB of text-like files
        if size < 10240:
            for pattern in text_suspicious:
                if pattern in header.lower():
                    # Could be a false positive in data, log but don't block
                    logger.warning(
                        f"Suspicious pattern '{pattern.decode()}' found in {file_path}"
                    )
        
        return True
    
    @classmethod
    def _is_malicious_zip(cls, file_path: str) -> bool:
        """Check if ZIP file contains malicious entries."""
        import zipfile
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Check for zip bomb (compression ratio)
                total_size = sum(info.file_size for info in zf.infolist())
                compressed_size = sum(info.compress_size for info in zf.infolist())
                
                if compressed_size > 0:
                    ratio = total_size / compressed_size
                    if ratio > 100:  # 100:1 compression ratio = likely zip bomb
                        raise SecurityError(
                            f"Possible zip bomb detected (compression ratio: {ratio:.1f}:1)"
                        )
                
                # Check for executable files inside
                dangerous_extensions = {'.exe', '.dll', '.bat', '.cmd', '.sh', 
                                        '.php', '.py', '.js', '.jar', '.class'}
                for name in zf.namelist():
                    ext = os.path.splitext(name)[1].lower()
                    if ext in dangerous_extensions:
                        raise SecurityError(
                            f"Executable file found in archive: {name}"
                        )
                    # Check for path traversal in zip entries
                    if '..' in name or name.startswith('/'):
                        raise SecurityError(
                            f"Path traversal attempt in archive: {name}"
                        )
                
                return False
        except zipfile.BadZipFile:
            raise SecurityError("Invalid or corrupted ZIP file")
        except Exception as e:
            raise SecurityError(f"Error checking ZIP file: {e}")
    
    @classmethod
    def validate_file_safe(cls, file_path: str, filename: str) -> bool:
        """Complete file validation pipeline.
        
        Returns True if file is safe, raises SecurityError otherwise.
        """
        # Step 1: Validate extension
        cls.validate_file_extension(filename)
        
        # Step 2: Validate filename (no path traversal, etc)
        cls.validate_filename(filename)
        
        # Step 3: Validate file size
        cls.validate_file_size(file_path)
        
        # Step 4: Validate content (no executables)
        cls.validate_file_content(file_path)
        
        return True
    
    @classmethod
    def get_file_info(cls, file_path: str) -> dict:
        """Get safe file information for logging."""
        try:
            stat = os.stat(file_path)
            return {
                "size_bytes": stat.st_size,
                "size_human": f"{stat.st_size / (1024**2):.2f} MB",
                "mtime": stat.st_mtime,
            }
        except Exception:
            return {"error": "Could not get file info"}
    
    @classmethod
    def validate_directory_path(cls, path: str) -> str:
        """Validate directory path is safe."""
        try:
            pathvalidate.sanitize_filepath(path)
        except Exception as e:
            raise SecurityError(f"Invalid directory path: {e}")
        
        # Check for path traversal
        resolved = os.path.abspath(path)
        if '..' in path:
            raise SecurityError("Path traversal attempt in directory path")
        
        return resolved


class URLClassifier:
    """Classifies URLs to determine the appropriate handler."""
    
    @staticmethod
    def is_kaggle_url(url: str) -> bool:
        """Check if URL is a Kaggle dataset URL."""
        patterns = [
            r'kaggle\.com/datasets',
            r'kaggle\.com/competitions',
            r'kaggle\.com/code',
        ]
        return any(re.search(p, url, re.IGNORECASE) for p in patterns)
    
    @staticmethod
    def is_github_url(url: str) -> bool:
        """Check if URL is a GitHub raw file URL."""
        patterns = [
            r'raw\.githubusercontent\.com',
            r'github\.com/.*/raw/',
            r'github\.com/.*/blob/',
        ]
        return any(re.search(p, url, re.IGNORECASE) for p in patterns)
    
    @staticmethod
    def is_google_drive_url(url: str) -> bool:
        """Check if URL is a Google Drive URL."""
        patterns = [
            r'drive\.google\.com/file/d/',
            r'drive\.google\.com/open\?id=',
            r'drive\.google\.com/uc\?id=',
        ]
        return any(re.search(p, url, re.IGNORECASE) for p in patterns)
    
    @staticmethod
    def is_direct_download_url(url: str) -> bool:
        """Check if URL is a direct download link."""
        direct_extensions = r'\.(csv|json|xlsx|zip|tar|gz|h5|parquet)(\?.*)?$'
        return bool(re.search(direct_extensions, url, re.IGNORECASE))


def sanitize_path(base_path: str, *paths: str) -> str:
    """Safely join paths and sanitize result."""
    full_path = os.path.join(base_path, *paths)
    resolved = os.path.abspath(full_path)
    base_resolved = os.path.abspath(base_path)
    
    # Ensure the resolved path is within base_path
    if not resolved.startswith(base_resolved):
        raise SecurityError("Path traversal attempt detected")
    
    return resolved
