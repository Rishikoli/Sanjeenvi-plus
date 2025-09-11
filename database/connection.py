import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiosqlite
from pydantic_settings import BaseSettings


logger = logging.getLogger(__name__)


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    database_url: str = "sqlite+aiosqlite:///sanjeevni.db"
    
    @property
    def database_path(self) -> str:
        """Extract and normalize the database file path from the URL."""
        # Extract the path part from the URL
        if ":///" in self.database_url:
            # Format: sqlite+aiosqlite:///path/to/db
            path_str = self.database_url.split("///")[-1]
        elif "://" in self.database_url:
            # Format: sqlite+aiosqlite://path/to/db
            path_str = self.database_url.split("://")[-1]
        else:
            path_str = self.database_url
            
        # Handle relative paths (starting with ./ or .\ or nothing)
        if path_str.startswith(('./', '.\\')):
            base_path = Path.cwd()
            path_str = str((base_path / path_str).resolve())
        elif not Path(path_str).is_absolute():
            # If it's a relative path without ./ or .\
            path_str = str((Path.cwd() / path_str).resolve())
            
        # Ensure the path uses the correct directory separator for the OS
        path = Path(path_str).resolve()
        
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Database path resolved to: {path}")
        return str(path)


class DatabaseConnection:
    """Manages SQLite database connections with proper error handling."""
    
    def __init__(self, database_path: str):
        # The path is already processed by DatabaseSettings
        self.database_path = database_path
        self._connection: Optional[aiosqlite.Connection] = None
        logger.info(f"Initialized database connection with path: {self.database_path}")
        
        # Ensure the directory exists
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def connect(self) -> aiosqlite.Connection:
        """Establish database connection."""
        try:
            # Ensure database directory exists
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to Windows path if needed and ensure forward slashes
            db_uri = f"file:{db_path.absolute().as_posix()}?mode=rwc"
            
            self._connection = await aiosqlite.connect(db_uri, uri=True)
            self._connection.row_factory = aiosqlite.Row  # Enable dict-like access
            
            # Enable foreign key support
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.commit()
            
            logger.info(f"Connected to database: {db_path.absolute()}")
            return self._connection
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}", exc_info=True)
            raise
    
    async def disconnect(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Context manager for database connections."""
        connection = None
        try:
            db_uri = f"file:{Path(self.database_path).absolute().as_posix()}"
            connection = await aiosqlite.connect(db_uri, uri=True)
            connection.row_factory = aiosqlite.Row
            # Enable foreign key support
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.commit()
            yield connection
        except Exception as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            raise
        finally:
            if connection:
                await connection.close()
    
    async def execute_script(self, script: str) -> None:
        """Execute SQL script (for schema creation)."""
        async with self.get_connection() as conn:
            try:
                await conn.executescript(script)
                await conn.commit()
                logger.info("SQL script executed successfully")
            except Exception as e:
                logger.error(f"Failed to execute SQL script: {e}")
                raise
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            async with self.get_connection() as conn:
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database instance
db_settings = DatabaseSettings()
db_connection = DatabaseConnection(db_settings.database_path)
