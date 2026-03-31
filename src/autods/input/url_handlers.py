"""URL handlers for different data sources."""
import os
import re
import shutil
import zipfile
import tarfile
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

# Optional imports with fallback
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

from ..security.validator import InputValidator, SecurityError, URLClassifier


class DownloadError(Exception):
    """Raised when a download fails."""
    pass


class BaseURLHandler:
    """Base class for URL handlers."""
    
    def __init__(self, download_dir: str = "raw_data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AutoDS-AI-Assistant/1.0 (Data Science Automation)'
        })
    
    def download(self, url: str, **kwargs) -> str:
        """Download file from URL. Returns path to downloaded file."""
        raise NotImplementedError
    
    def _download_with_progress(self, url: str, destination: Path, 
                               headers: Optional[Dict] = None) -> str:
        """Download file with progress bar."""
        response = self.session.get(url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        InputValidator.validate_file_size(str(destination))
        InputValidator.validate_file_content(str(destination))
        
        return str(destination)
    
    def _extract_archive(self, archive_path: str, extract_to: Optional[str] = None) -> str:
        """Extract archive file."""
        if extract_to is None:
            extract_to = os.path.splitext(archive_path)[0]
        
        os.makedirs(extract_to, exist_ok=True)
        
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            return archive_path  # Not an archive, return as-is
        
        os.remove(archive_path)  # Clean up archive
        return extract_to


class KaggleHandler(BaseURLHandler):
    """Handler for Kaggle datasets."""
    
    def __init__(self, download_dir: str = "raw_data"):
        super().__init__(download_dir)
        self._check_kaggle_auth()
    
    def _check_kaggle_auth(self):
        """Check if Kaggle API credentials are available."""
        if not KAGGLE_AVAILABLE:
            raise DownloadError(
                "kaggle package is not installed. Run: pip install kaggle"
            )
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        env_vars = os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')
        
        if not kaggle_json.exists() and not env_vars:
            raise DownloadError(
                "Kaggle credentials not found. Please set up kaggle.json or "
                "KAGGLE_USERNAME and KAGGLE_KEY environment variables."
            )
    
    def download(self, url: str, **kwargs) -> str:
        """Download Kaggle dataset."""
        InputValidator.validate_url(url)
        
        # Extract dataset identifier from URL
        # Format: kaggle.com/datasets/{owner}/{dataset-name}
        match = re.search(r'kaggle\.com/datasets/([^/]+)/([^/?]+)', url)
        if not match:
            raise DownloadError(f"Could not parse Kaggle URL: {url}")
        
        owner, dataset = match.groups()
        dataset_path = f"{owner}/{dataset}"
        
        # Create download directory
        download_path = self.download_dir / f"kaggle_{owner}_{dataset}"
        download_path.mkdir(parents=True, exist_ok=True)
        
        # Use Kaggle API to download
        try:
            kaggle.api.dataset_download_files(
                dataset_path,
                path=str(download_path),
                unzip=True,
                quiet=False
            )
        except Exception as e:
            raise DownloadError(f"Kaggle download failed: {e}")
        
        return str(download_path)


class GitHubHandler(BaseURLHandler):
    """Handler for GitHub raw file URLs."""
    
    def download(self, url: str, **kwargs) -> str:
        """Download file from GitHub."""
        InputValidator.validate_url(url)
        
        # Convert blob URLs to raw URLs
        if '/blob/' in url:
            url = url.replace('/blob/', '/raw/')
        
        # Extract filename from URL
        parsed = requests.utils.urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            filename = 'github_download'
        
        filename = InputValidator.validate_filename(filename)
        
        # Create destination path
        dest_path = self.download_dir / f"github_{filename}"
        
        # Download
        downloaded = self._download_with_progress(url, dest_path)
        
        # Extract if archive
        if zipfile.is_zipfile(downloaded) or tarfile.is_tarfile(downloaded):
            return self._extract_archive(downloaded)
        
        return downloaded


class GoogleDriveHandler(BaseURLHandler):
    """Handler for Google Drive file URLs."""
    
    def download(self, url: str, **kwargs) -> str:
        """Download file from Google Drive."""
        InputValidator.validate_url(url)
        
        # Extract file ID from URL
        file_id = self._extract_file_id(url)
        if not file_id:
            raise DownloadError(f"Could not extract Google Drive file ID from: {url}")
        
        # Create destination
        output = kwargs.get('output') or f"gdrive_{file_id}"
        dest_path = self.download_dir / output
        
        # Download using gdown
        if not GDOWN_AVAILABLE:
            raise DownloadError("gdown is not installed. Run: pip install gdown")
        try:
            gdown.download(id=file_id, output=str(dest_path), quiet=False, fuzzy=True)
        except Exception as e:
            raise DownloadError(f"Google Drive download failed: {e}")
        
        InputValidator.validate_file_size(str(dest_path))
        InputValidator.validate_file_content(str(dest_path))
        
        # Extract if archive
        if zipfile.is_zipfile(dest_path) or tarfile.is_tarfile(dest_path):
            return self._extract_archive(str(dest_path))
        
        return str(dest_path)
    
    def _extract_file_id(self, url: str) -> Optional[str]:
        """Extract Google Drive file ID from URL."""
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'[?&]id=([a-zA-Z0-9_-]+)',
            r'/uc\?id=([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None


class DirectDownloadHandler(BaseURLHandler):
    """Handler for direct file download URLs."""
    
    def download(self, url: str, **kwargs) -> str:
        """Download file from direct URL."""
        InputValidator.validate_url(url)
        
        # Extract filename from URL
        parsed = requests.utils.urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            # Try to get filename from Content-Disposition header
            response = self.session.head(url, allow_redirects=True, timeout=30)
            cd = response.headers.get('content-disposition', '')
            if 'filename=' in cd:
                filename = cd.split('filename=')[1].strip('"\'')
        
        if not filename:
            filename = 'download'
        
        filename = InputValidator.validate_filename(filename)
        
        # Create destination path
        dest_path = self.download_dir / filename
        
        # Download
        downloaded = self._download_with_progress(url, dest_path)
        
        # Extract if archive
        if zipfile.is_zipfile(downloaded) or tarfile.is_tarfile(downloaded):
            return self._extract_archive(downloaded)
        
        return downloaded


class URLHandlerFactory:
    """Factory for getting appropriate URL handler."""
    
    _handlers = {
        'kaggle': KaggleHandler,
        'github': GitHubHandler,
        'google_drive': GoogleDriveHandler,
        'direct': DirectDownloadHandler,
    }
    
    @classmethod
    def get_handler(cls, url: str, download_dir: str = "raw_data") -> BaseURLHandler:
        """Get appropriate handler for URL."""
        InputValidator.validate_url(url)
        
        if URLClassifier.is_kaggle_url(url):
            return KaggleHandler(download_dir)
        elif URLClassifier.is_github_url(url):
            return GitHubHandler(download_dir)
        elif URLClassifier.is_google_drive_url(url):
            return GoogleDriveHandler(download_dir)
        else:
            return DirectDownloadHandler(download_dir)
    
    @classmethod
    def download(cls, url: str, download_dir: str = "raw_data", **kwargs) -> str:
        """Download from URL using appropriate handler."""
        handler = cls.get_handler(url, download_dir)
        return handler.download(url, **kwargs)
