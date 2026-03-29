"""Data ingestion module combining URL and file handling."""
import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

from ..input.file_handler import FileHandler, DataType
from ..input.url_handlers import URLHandlerFactory, DownloadError
from ..security.validator import InputValidator, SecurityError


class DataIngestion:
    """Main entry point for data ingestion."""
    
    def __init__(self, 
                 raw_data_dir: str = "raw_data",
                 processed_data_dir: str = "processed_data"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_handler = FileHandler(str(self.raw_data_dir))
    
    def from_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Load data from local file."""
        return self.file_handler.load(file_path, **kwargs)
    
    def from_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Download and load data from URL."""
        # Download
        download_path = URLHandlerFactory.download(url, str(self.raw_data_dir), **kwargs)
        
        # If it's a directory (extracted archive), load all files
        if os.path.isdir(download_path):
            files = []
            for item in Path(download_path).rglob('*'):
                if item.is_file():
                    try:
                        InputValidator.validate_file_extension(str(item))
                        files.append(self.file_handler.load(str(item), **kwargs))
                    except SecurityError as e:
                        print(f"Skipping {item}: {e}")
            return {
                'download_path': download_path,
                'files': files,
                'type': 'directory'
            }
        
        # Single file
        return self.file_handler.load(download_path, **kwargs)
    
    def from_directory(self, directory: str, pattern: str = "*", 
                       recursive: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """Load all data files from a directory."""
        return self.file_handler.load_directory(directory, pattern, recursive, **kwargs)
    
    def get_data_info(self, source: str) -> Union[Dict, List[Dict]]:
        """Get information about data source without loading."""
        # Check if URL
        if source.startswith(('http://', 'https://')):
            return {'type': 'url', 'source': source, 'note': 'Use from_url() to download and inspect'}
        
        # Check if directory
        if os.path.isdir(source):
            infos = []
            for item in Path(source).rglob('*'):
                if item.is_file():
                    try:
                        InputValidator.validate_file_extension(str(item))
                        infos.append(self.file_handler.get_info(str(item)))
                    except SecurityError:
                        continue
            return {'type': 'directory', 'files': infos}
        
        # Single file
        return self.file_handler.get_info(source)
    
    def preview(self, source: str, n_rows: int = 5) -> Dict[str, Any]:
        """Quick preview of data source."""
        info = self.get_data_info(source)
        
        if isinstance(info, list):
            return {'files': info[:5], 'total_files': len(info)}
        
        if info.get('type') == 'directory':
            files = info.get('files', [])
            return {'files': files[:5], 'total_files': len(files)}
        
        # Try to load small sample
        if info.get('data_type') == DataType.TABULAR:
            try:
                data = self.file_handler._load_tabular(source, nrows=n_rows)
                return {
                    'info': info,
                    'preview': data.to_dict('records') if hasattr(data, 'to_dict') else data
                }
            except Exception as e:
                return {'info': info, 'error': str(e)}
        
        return info


# Convenience functions for direct use
def load_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """Load a file directly."""
    handler = FileHandler()
    return handler.load(file_path, **kwargs)


def load_from_url(url: str, **kwargs) -> Dict[str, Any]:
    """Download and load from URL."""
    ingestion = DataIngestion()
    return ingestion.from_url(url, **kwargs)


def load_directory(directory: str, **kwargs) -> List[Dict[str, Any]]:
    """Load all files from directory."""
    handler = FileHandler()
    return handler.load_directory(directory, **kwargs)


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file information."""
    handler = FileHandler()
    return handler.get_info(file_path)
