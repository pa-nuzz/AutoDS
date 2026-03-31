"""Input handler for various file formats."""
import os
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import pandas as pd
import numpy as np

# Optional imports with fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from ..security.validator import InputValidator, SecurityError


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


class DataType:
    """Enum-like class for data types."""
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class FileHandler:
    """Handles loading of various file formats."""
    
    # File type mapping
    EXTENSION_MAP = {
        # Tabular
        '.csv': DataType.TABULAR,
        '.xlsx': DataType.TABULAR,
        '.xls': DataType.TABULAR,
        '.json': DataType.TABULAR,
        '.parquet': DataType.TABULAR,
        '.feather': DataType.TABULAR,
        '.hdf5': DataType.TABULAR,
        '.h5': DataType.TABULAR,
        '.tsv': DataType.TABULAR,
        # Text
        '.txt': DataType.TEXT,
        '.md': DataType.TEXT,
        '.log': DataType.TEXT,
        # Images
        '.jpg': DataType.IMAGE,
        '.jpeg': DataType.IMAGE,
        '.png': DataType.IMAGE,
        '.gif': DataType.IMAGE,
        '.bmp': DataType.IMAGE,
        '.tiff': DataType.IMAGE,
        '.webp': DataType.IMAGE,
        '.svg': DataType.IMAGE,
        # Audio
        '.wav': DataType.AUDIO,
        '.mp3': DataType.AUDIO,
        '.flac': DataType.AUDIO,
        '.aac': DataType.AUDIO,
        '.ogg': DataType.AUDIO,
        '.m4a': DataType.AUDIO,
        '.wma': DataType.AUDIO,
    }
    
    def __init__(self, base_path: str = "raw_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def detect_type(self, file_path: str) -> str:
        """Detect data type from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.EXTENSION_MAP.get(ext, DataType.UNKNOWN)
    
    def load(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Load file and return data with metadata."""
        # Security validation
        file_path = str(Path(file_path).resolve())
        if not os.path.exists(file_path):
            raise DataLoadError(f"File not found: {file_path}")
        
        InputValidator.validate_file_extension(file_path)
        InputValidator.validate_file_size(file_path)
        InputValidator.validate_file_content(file_path)
        
        data_type = self.detect_type(file_path)
        
        result = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'data_type': data_type,
            'file_size': os.path.getsize(file_path),
        }
        
        # Load based on type
        if data_type == DataType.TABULAR:
            result['data'] = self._load_tabular(file_path, **kwargs)
            result['shape'] = result['data'].shape if hasattr(result['data'], 'shape') else None
        elif data_type == DataType.TEXT:
            result['data'] = self._load_text(file_path, **kwargs)
        elif data_type == DataType.IMAGE:
            result['data'] = self._load_image(file_path, **kwargs)
            result['shape'] = result['data'].shape if hasattr(result['data'], 'shape') else None
        elif data_type == DataType.AUDIO:
            result['data'], result['sample_rate'] = self._load_audio(file_path, **kwargs)
            result['duration'] = len(result['data']) / result['sample_rate']
        else:
            raise DataLoadError(f"Unsupported file type: {file_path}")
        
        return result
    
    def _load_tabular(self, file_path: str, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load tabular data files."""
        ext = Path(file_path).suffix.lower()
        
        try:
            if ext == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        return pd.read_csv(file_path, encoding=encoding, **kwargs)
                    except UnicodeDecodeError:
                        continue
                raise DataLoadError("Could not decode CSV file with any encoding")
            
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path, **kwargs)
            
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Try to convert to DataFrame if it's a list of dicts
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                return data
            
            elif ext == '.parquet':
                return pd.read_parquet(file_path, **kwargs)
            
            elif ext == '.feather':
                return pd.read_feather(file_path, **kwargs)
            
            elif ext in ['.hdf5', '.h5']:
                key = kwargs.get('key')
                return pd.read_hdf(file_path, key=key)
            
            elif ext == '.tsv':
                return pd.read_csv(file_path, sep='\t', **kwargs)
            
        except Exception as e:
            raise DataLoadError(f"Failed to load tabular file: {e}")
        
        raise DataLoadError(f"Unsupported tabular extension: {ext}")
    
    def _load_text(self, file_path: str, **kwargs) -> str:
        """Load text files."""
        encodings = kwargs.get('encodings', ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252'])
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise DataLoadError("Could not decode text file with any encoding")
    
    def _load_image(self, file_path: str, **kwargs) -> np.ndarray:
        """Load image files."""
        if not PIL_AVAILABLE:
            raise DataLoadError("Pillow is not installed. Run: pip install Pillow")
        try:
            img = Image.open(file_path)
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            raise DataLoadError(f"Failed to load image: {e}")
    
    def _load_audio(self, file_path: str, **kwargs) -> tuple:
        """Load audio files. Returns (audio_data, sample_rate)."""
        if not LIBROSA_AVAILABLE:
            raise DataLoadError("librosa is not installed. Run: pip install librosa")
        try:
            sr = kwargs.get('sr', None)  # None preserves native sample rate
            audio, sample_rate = librosa.load(file_path, sr=sr, mono=kwargs.get('mono', True))
            return audio, sample_rate
        except Exception as e:
            raise DataLoadError(f"Failed to load audio: {e}")
    
    def load_directory(self, directory: str, pattern: str = "*", 
                       recursive: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """Load all matching files from a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise DataLoadError(f"Directory not found: {directory}")
        
        results = []
        
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)
        
        for file_path in files:
            if file_path.is_file():
                try:
                    InputValidator.validate_file_extension(str(file_path))
                    results.append(self.load(str(file_path), **kwargs))
                except (SecurityError, DataLoadError) as e:
                    print(f"Skipping {file_path}: {e}")
                    continue
        
        return results
    
    def get_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information without loading full data."""
        file_path = str(Path(file_path).resolve())
        
        if not os.path.exists(file_path):
            raise DataLoadError(f"File not found: {file_path}")
        
        info = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'data_type': self.detect_type(file_path),
            'extension': Path(file_path).suffix.lower(),
        }
        
        # Add preview for tabular data
        if info['data_type'] == DataType.TABULAR:
            try:
                df = self._load_tabular(file_path, nrows=5)
                if isinstance(df, pd.DataFrame):
                    info['columns'] = list(df.columns)
                    info['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    info['num_rows'] = pd.read_csv(file_path, usecols=[0]).shape[0] if info['extension'] == '.csv' else None
            except Exception:
                pass
        
        return info
