"""Session persistence for saving previous runs and datasets.

Allows users to save and quickly reuse previous analysis sessions.
"""
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata for a saved session."""
    session_id: str
    name: str
    created_at: str
    updated_at: str
    data_shape: tuple
    has_target: bool
    target_column: Optional[str]
    preprocessing_applied: bool
    models_trained: bool
    tags: List[str]
    description: str


class SessionManager:
    """Manages saving and loading of analysis sessions.
    
    Sessions are stored in a structured directory:
    sessions/
      {session_id}/
        metadata.json
        data.parquet (original data)
        processed_data.parquet (if preprocessing applied)
        profile.json (data profile)
        model_results.json (if models trained)
        preprocessing_steps.json
    """
    
    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_session_id(self, df: pd.DataFrame) -> str:
        """Generate unique session ID based on data hash and timestamp."""
        # Create hash from data sample
        sample = df.head(100).to_json()
        data_hash = hashlib.md5(sample.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{data_hash}"
    
    def save_session(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        processed_df: Optional[pd.DataFrame] = None,
        profile: Optional[Dict] = None,
        model_results: Optional[Dict] = None,
        preprocessing_steps: Optional[List] = None,
        target_column: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = ""
    ) -> str:
        """Save a complete analysis session.
        
        Returns:
            session_id: The ID of the saved session
        """
        session_id = self._generate_session_id(df)
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        now = datetime.now().isoformat()
        
        # Create metadata
        metadata = SessionMetadata(
            session_id=session_id,
            name=name or f"Session {session_id[:8]}",
            created_at=now,
            updated_at=now,
            data_shape=df.shape,
            has_target=target_column is not None,
            target_column=target_column,
            preprocessing_applied=processed_df is not None,
            models_trained=model_results is not None,
            tags=tags or [],
            description=description
        )
        
        # Save metadata
        with open(session_dir / "metadata.json", 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Save data
        df.to_parquet(session_dir / "data.parquet", index=False)
        
        # Save processed data if available
        if processed_df is not None:
            processed_df.to_parquet(session_dir / "processed_data.parquet", index=False)
        
        # Save profile if available
        if profile is not None:
            with open(session_dir / "profile.json", 'w') as f:
                json.dump(profile, f, indent=2, default=str)
        
        # Save model results if available
        if model_results is not None:
            with open(session_dir / "model_results.json", 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
        
        # Save preprocessing steps
        if preprocessing_steps is not None:
            with open(session_dir / "preprocessing_steps.json", 'w') as f:
                json.dump(preprocessing_steps, f, indent=2)
        
        logger.info(f"Saved session {session_id} with {df.shape[0]} rows")
        return session_id
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Load a complete session.
        
        Returns:
            Dictionary with data, metadata, and any processed results
        """
        session_dir = self.base_dir / session_id
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        
        result = {}
        
        # Load metadata
        with open(session_dir / "metadata.json", 'r') as f:
            result['metadata'] = json.load(f)
        
        # Load data
        result['data'] = pd.read_parquet(session_dir / "data.parquet")
        
        # Load processed data if available
        processed_path = session_dir / "processed_data.parquet"
        if processed_path.exists():
            result['processed_data'] = pd.read_parquet(processed_path)
        
        # Load profile if available
        profile_path = session_dir / "profile.json"
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                result['profile'] = json.load(f)
        
        # Load model results if available
        model_path = session_dir / "model_results.json"
        if model_path.exists():
            with open(model_path, 'r') as f:
                result['model_results'] = json.load(f)
        
        # Load preprocessing steps if available
        steps_path = session_dir / "preprocessing_steps.json"
        if steps_path.exists():
            with open(steps_path, 'r') as f:
                result['preprocessing_steps'] = json.load(f)
        
        return result
    
    def list_sessions(self) -> List[SessionMetadata]:
        """List all saved sessions with metadata."""
        sessions = []
        
        for session_dir in self.base_dir.iterdir():
            if session_dir.is_dir():
                metadata_path = session_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            data = json.load(f)
                            sessions.append(SessionMetadata(**data))
                    except Exception as e:
                        logger.warning(f"Failed to load session {session_dir.name}: {e}")
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        session_dir = self.base_dir / session_id
        
        if not session_dir.exists():
            return False
        
        # Delete all files in session directory
        for file_path in session_dir.iterdir():
            file_path.unlink()
        
        session_dir.rmdir()
        logger.info(f"Deleted session {session_id}")
        return True
    
    def update_session(
        self,
        session_id: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> bool:
        """Update session metadata."""
        session_dir = self.base_dir / session_id
        metadata_path = session_dir / "metadata.json"
        
        if not metadata_path.exists():
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if name is not None:
            metadata['name'] = name
        if tags is not None:
            metadata['tags'] = tags
        if description is not None:
            metadata['description'] = description
        
        metadata['updated_at'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    def export_session(self, session_id: str, export_path: str) -> str:
        """Export session as a ZIP archive."""
        import zipfile
        
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in session_dir.iterdir():
                zf.write(file_path, file_path.name)
        
        return str(export_file)
    
    def import_session(self, zip_path: str) -> str:
        """Import a session from a ZIP archive."""
        import zipfile
        
        zip_file = Path(zip_path)
        if not zip_file.exists():
            raise FileNotFoundError(f"Import file not found: {zip_path}")
        
        # Extract to a new session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"imported_{timestamp}"
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(session_dir)
        
        # Update metadata
        metadata_path = session_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['session_id'] = session_id
            metadata['created_at'] = datetime.now().isoformat()
            metadata['updated_at'] = datetime.now().isoformat()
            metadata['name'] = f"Imported: {metadata.get('name', 'Unknown')}"
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Imported session as {session_id}")
        return session_id


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def auto_save_session(
    df: pd.DataFrame,
    name: Optional[str] = None,
    **kwargs
) -> str:
    """Quick save a session using the global manager."""
    manager = get_session_manager()
    return manager.save_session(df, name=name, **kwargs)
