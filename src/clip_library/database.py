"""
Clip library database
Stores and manages generated music clips with metadata
"""

import logging
import sqlite3
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ClipMetadata:
    """Metadata for a music clip"""
    clip_id: str
    name: str
    prompt: str
    lyrics: Optional[str]
    duration: float
    bpm: Optional[int]
    key: Optional[str]
    genre: Optional[str]
    mood: Optional[str]
    created_at: str
    file_path: str
    tags: List[str]
    custom_data: Dict[str, Any]


class ClipLibrary:
    """
    Manages clip library using SQLite
    Stores clip metadata and file references
    """
    
    def __init__(self, db_path: str = "data/emma_clips.db"):
        self.db_path = db_path
        self._ensure_db_exists()
        
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                lyrics TEXT,
                duration REAL NOT NULL,
                bpm INTEGER,
                key TEXT,
                genre TEXT,
                mood TEXT,
                created_at TEXT NOT NULL,
                file_path TEXT NOT NULL,
                tags TEXT,
                custom_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Clip library database initialized at {self.db_path}")
    
    def add_clip(self, metadata: ClipMetadata) -> bool:
        """
        Add clip to library
        
        Args:
            metadata: Clip metadata
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO clips (
                    clip_id, name, prompt, lyrics, duration, bpm, key, genre, mood,
                    created_at, file_path, tags, custom_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.clip_id,
                metadata.name,
                metadata.prompt,
                metadata.lyrics,
                metadata.duration,
                metadata.bpm,
                metadata.key,
                metadata.genre,
                metadata.mood,
                metadata.created_at,
                metadata.file_path,
                json.dumps(metadata.tags),
                json.dumps(metadata.custom_data)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added clip {metadata.clip_id} to library")
            return True
            
        except Exception as e:
            logger.error(f"Error adding clip to library: {e}")
            return False
    
    def get_clip(self, clip_id: str) -> Optional[ClipMetadata]:
        """
        Get clip metadata by ID
        
        Args:
            clip_id: Clip ID
            
        Returns:
            Clip metadata or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM clips WHERE clip_id = ?', (clip_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_metadata(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting clip: {e}")
            return None
    
    def update_clip(self, clip_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update clip metadata
        
        Args:
            clip_id: Clip ID
            updates: Dictionary of fields to update
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build update query
            set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
            values = list(updates.values()) + [clip_id]
            
            cursor.execute(f'UPDATE clips SET {set_clause} WHERE clip_id = ?', values)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated clip {clip_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating clip: {e}")
            return False
    
    def delete_clip(self, clip_id: str, delete_file: bool = False) -> bool:
        """
        Delete clip from library
        
        Args:
            clip_id: Clip ID
            delete_file: Whether to also delete the audio file
            
        Returns:
            True if successful
        """
        try:
            # Get file path if needed
            if delete_file:
                metadata = self.get_clip(clip_id)
                if metadata and Path(metadata.file_path).exists():
                    Path(metadata.file_path).unlink()
                    logger.info(f"Deleted file: {metadata.file_path}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM clips WHERE clip_id = ?', (clip_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted clip {clip_id} from library")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting clip: {e}")
            return False
    
    def list_clips(
        self,
        filter_genre: Optional[str] = None,
        filter_mood: Optional[str] = None,
        filter_key: Optional[str] = None,
        limit: int = 100
    ) -> List[ClipMetadata]:
        """
        List clips with optional filters
        
        Args:
            filter_genre: Filter by genre
            filter_mood: Filter by mood
            filter_key: Filter by musical key
            limit: Maximum number of results
            
        Returns:
            List of clip metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM clips WHERE 1=1'
            params = []
            
            if filter_genre:
                query += ' AND genre = ?'
                params.append(filter_genre)
            
            if filter_mood:
                query += ' AND mood = ?'
                params.append(filter_mood)
            
            if filter_key:
                query += ' AND key = ?'
                params.append(filter_key)
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_metadata(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error listing clips: {e}")
            return []
    
    def search_clips(self, search_text: str) -> List[ClipMetadata]:
        """
        Search clips by name or prompt
        
        Args:
            search_text: Search query
            
        Returns:
            List of matching clips
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM clips 
                WHERE name LIKE ? OR prompt LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{search_text}%', f'%{search_text}%'))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_metadata(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error searching clips: {e}")
            return []
    
    def _row_to_metadata(self, row: tuple) -> ClipMetadata:
        """Convert database row to ClipMetadata"""
        return ClipMetadata(
            clip_id=row[0],
            name=row[1],
            prompt=row[2],
            lyrics=row[3],
            duration=row[4],
            bpm=row[5],
            key=row[6],
            genre=row[7],
            mood=row[8],
            created_at=row[9],
            file_path=row[10],
            tags=json.loads(row[11]) if row[11] else [],
            custom_data=json.loads(row[12]) if row[12] else {}
        )
