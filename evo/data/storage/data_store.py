"""
Data storage for market data.

This module provides data persistence and retrieval capabilities
for storing and accessing market data efficiently.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import pickle
import sqlite3
from pathlib import Path

from evo.core.logging import get_logger

logger = get_logger(__name__)


class DataStore:
    """
    Data store for market data persistence.
    
    Provides methods to store and retrieve market data efficiently
    using various storage backends.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data store.
        
        Args:
            config: Configuration dictionary containing:
                - storage_type: Type of storage ('file', 'sqlite', 'memory')
                - base_path: Base path for file storage
                - database_path: Path for SQLite database
                - compression: Whether to use compression
        """
        self.config = config or {}
        self.storage_type = self.config.get("storage_type", "file")
        self.base_path = Path(self.config.get("base_path", "data"))
        self.database_path = self.config.get("database_path", "data/market_data.db")
        self.compression = self.config.get("compression", False)
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self._init_storage()
        
        logger.info(f"Initialized {self.storage_type} data store")
    
    def _init_storage(self) -> None:
        """Initialize the storage backend."""
        if self.storage_type == "sqlite":
            self._init_sqlite()
        elif self.storage_type == "file":
            self._init_file_storage()
        elif self.storage_type == "memory":
            self._init_memory_storage()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite storage."""
        try:
            # Create database directory
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection
            self._conn = sqlite3.connect(self.database_path)
            self._conn.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
            logger.info(f"Initialized SQLite storage at {self.database_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite storage: {str(e)}")
            raise
    
    def _init_file_storage(self) -> None:
        """Initialize file-based storage."""
        try:
            # Create data directories
            (self.base_path / "bars").mkdir(exist_ok=True)
            (self.base_path / "features").mkdir(exist_ok=True)
            (self.base_path / "metadata").mkdir(exist_ok=True)
            
            logger.info(f"Initialized file storage at {self.base_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize file storage: {str(e)}")
            raise
    
    def _init_memory_storage(self) -> None:
        """Initialize in-memory storage."""
        self._memory_data = {}
        logger.info("Initialized memory storage")
    
    def _create_tables(self) -> None:
        """Create SQLite tables."""
        cursor = self._conn.cursor()
        
        # Bars table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                UNIQUE(symbol, timestamp, feature_name)
            )
        """)
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bars_symbol_timestamp ON bars(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp ON features(symbol, timestamp)")
        
        self._conn.commit()
    
    def store_bars(
        self,
        symbol: str,
        data: pd.DataFrame,
        overwrite: bool = False
    ) -> None:
        """
        Store bar data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
            overwrite: Whether to overwrite existing data
        """
        if data.empty:
            logger.warning(f"Empty data provided for {symbol}")
            return
        
        try:
            if self.storage_type == "sqlite":
                self._store_bars_sqlite(symbol, data, overwrite)
            elif self.storage_type == "file":
                self._store_bars_file(symbol, data, overwrite)
            elif self.storage_type == "memory":
                self._store_bars_memory(symbol, data, overwrite)
            
            logger.info(f"Stored {len(data)} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to store bars for {symbol}: {str(e)}")
            raise
    
    def _store_bars_sqlite(self, symbol: str, data: pd.DataFrame, overwrite: bool) -> None:
        """Store bars in SQLite database."""
        cursor = self._conn.cursor()
        
        if overwrite:
            cursor.execute("DELETE FROM bars WHERE symbol = ?", (symbol,))
        
        # Prepare data for insertion
        records = []
        for _, row in data.iterrows():
            records.append((
                symbol,
                row["timestamp"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"]
            ))
        
        cursor.executemany("""
            INSERT OR REPLACE INTO bars (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        self._conn.commit()
    
    def _store_bars_file(self, symbol: str, data: pd.DataFrame, overwrite: bool) -> None:
        """Store bars in file system."""
        file_path = self.base_path / "bars" / f"{symbol}.parquet"
        
        if overwrite or not file_path.exists():
            data.to_parquet(file_path, compression="snappy" if self.compression else None)
        else:
            # Append to existing file
            existing_data = pd.read_parquet(file_path)
            combined_data = pd.concat([existing_data, data]).drop_duplicates(subset=["timestamp"])
            combined_data.to_parquet(file_path, compression="snappy" if self.compression else None)
    
    def _store_bars_memory(self, symbol: str, data: pd.DataFrame, overwrite: bool) -> None:
        """Store bars in memory."""
        if overwrite or symbol not in self._memory_data:
            self._memory_data[symbol] = data
        else:
            # Append to existing data
            existing_data = self._memory_data[symbol]
            combined_data = pd.concat([existing_data, data]).drop_duplicates(subset=["timestamp"])
            self._memory_data[symbol] = combined_data
    
    def get_bars(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve bar data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.storage_type == "sqlite":
                return self._get_bars_sqlite(symbol, start_time, end_time)
            elif self.storage_type == "file":
                return self._get_bars_file(symbol, start_time, end_time)
            elif self.storage_type == "memory":
                return self._get_bars_memory(symbol, start_time, end_time)
            
        except Exception as e:
            logger.error(f"Failed to retrieve bars for {symbol}: {str(e)}")
            raise
    
    def _get_bars_sqlite(self, symbol: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> pd.DataFrame:
        """Retrieve bars from SQLite database."""
        query = "SELECT * FROM bars WHERE symbol = ?"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, self._conn, params=params)
        return df
    
    def _get_bars_file(self, symbol: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> pd.DataFrame:
        """Retrieve bars from file system."""
        file_path = self.base_path / "bars" / f"{symbol}.parquet"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        
        # Apply time filters
        if start_time:
            df = df[df["timestamp"] >= start_time]
        
        if end_time:
            df = df[df["timestamp"] <= end_time]
        
        return df
    
    def _get_bars_memory(self, symbol: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> pd.DataFrame:
        """Retrieve bars from memory."""
        if symbol not in self._memory_data:
            return pd.DataFrame()
        
        df = self._memory_data[symbol].copy()
        
        # Apply time filters
        if start_time:
            df = df[df["timestamp"] >= start_time]
        
        if end_time:
            df = df[df["timestamp"] <= end_time]
        
        return df
    
    def store_features(
        self,
        symbol: str,
        data: pd.DataFrame,
        overwrite: bool = False
    ) -> None:
        """
        Store feature data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with features
            overwrite: Whether to overwrite existing data
        """
        if data.empty:
            logger.warning(f"Empty feature data provided for {symbol}")
            return
        
        try:
            if self.storage_type == "sqlite":
                self._store_features_sqlite(symbol, data, overwrite)
            elif self.storage_type == "file":
                self._store_features_file(symbol, data, overwrite)
            elif self.storage_type == "memory":
                self._store_features_memory(symbol, data, overwrite)
            
            logger.info(f"Stored features for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {str(e)}")
            raise
    
    def _store_features_sqlite(self, symbol: str, data: pd.DataFrame, overwrite: bool) -> None:
        """Store features in SQLite database."""
        cursor = self._conn.cursor()
        
        if overwrite:
            cursor.execute("DELETE FROM features WHERE symbol = ?", (symbol,))
        
        # Prepare data for insertion
        records = []
        for _, row in data.iterrows():
            timestamp = row["timestamp"]
            for col in data.columns:
                if col != "timestamp" and pd.notna(row[col]):
                    records.append((symbol, timestamp, col, row[col]))
        
        cursor.executemany("""
            INSERT OR REPLACE INTO features (symbol, timestamp, feature_name, feature_value)
            VALUES (?, ?, ?, ?)
        """, records)
        
        self._conn.commit()
    
    def _store_features_file(self, symbol: str, data: pd.DataFrame, overwrite: bool) -> None:
        """Store features in file system."""
        file_path = self.base_path / "features" / f"{symbol}_features.parquet"
        
        if overwrite or not file_path.exists():
            data.to_parquet(file_path, compression="snappy" if self.compression else None)
        else:
            # Append to existing file
            existing_data = pd.read_parquet(file_path)
            combined_data = pd.concat([existing_data, data]).drop_duplicates(subset=["timestamp"])
            combined_data.to_parquet(file_path, compression="snappy" if self.compression else None)
    
    def _store_features_memory(self, symbol: str, data: pd.DataFrame, overwrite: bool) -> None:
        """Store features in memory."""
        key = f"{symbol}_features"
        if overwrite or key not in self._memory_data:
            self._memory_data[key] = data
        else:
            # Append to existing data
            existing_data = self._memory_data[key]
            combined_data = pd.concat([existing_data, data]).drop_duplicates(subset=["timestamp"])
            self._memory_data[key] = combined_data
    
    def get_features(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve feature data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            
        Returns:
            DataFrame with features
        """
        try:
            if self.storage_type == "sqlite":
                return self._get_features_sqlite(symbol, start_time, end_time)
            elif self.storage_type == "file":
                return self._get_features_file(symbol, start_time, end_time)
            elif self.storage_type == "memory":
                return self._get_features_memory(symbol, start_time, end_time)
            
        except Exception as e:
            logger.error(f"Failed to retrieve features for {symbol}: {str(e)}")
            raise
    
    def _get_features_sqlite(self, symbol: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> pd.DataFrame:
        """Retrieve features from SQLite database."""
        query = """
            SELECT timestamp, feature_name, feature_value 
            FROM features 
            WHERE symbol = ?
        """
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, self._conn, params=params)
        
        if df.empty:
            return df
        
        # Pivot to wide format
        df = df.pivot(index="timestamp", columns="feature_name", values="feature_value")
        df.reset_index(inplace=True)
        
        return df
    
    def _get_features_file(self, symbol: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> pd.DataFrame:
        """Retrieve features from file system."""
        file_path = self.base_path / "features" / f"{symbol}_features.parquet"
        
        if not file_path.exists():
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        
        # Apply time filters
        if start_time:
            df = df[df["timestamp"] >= start_time]
        
        if end_time:
            df = df[df["timestamp"] <= end_time]
        
        return df
    
    def _get_features_memory(self, symbol: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> pd.DataFrame:
        """Retrieve features from memory."""
        key = f"{symbol}_features"
        if key not in self._memory_data:
            return pd.DataFrame()
        
        df = self._memory_data[key].copy()
        
        # Apply time filters
        if start_time:
            df = df[df["timestamp"] >= start_time]
        
        if end_time:
            df = df[df["timestamp"] <= end_time]
        
        return df
    
    def get_symbols(self) -> List[str]:
        """
        Get list of symbols with stored data.
        
        Returns:
            List of symbols
        """
        try:
            if self.storage_type == "sqlite":
                cursor = self._conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM bars")
                return [row[0] for row in cursor.fetchall()]
            elif self.storage_type == "file":
                bars_dir = self.base_path / "bars"
                if not bars_dir.exists():
                    return []
                return [f.stem for f in bars_dir.glob("*.parquet")]
            elif self.storage_type == "memory":
                return list(self._memory_data.keys())
            
        except Exception as e:
            logger.error(f"Failed to get symbols: {str(e)}")
            return []
    
    def delete_symbol(self, symbol: str) -> None:
        """
        Delete all data for a symbol.
        
        Args:
            symbol: Trading symbol to delete
        """
        try:
            if self.storage_type == "sqlite":
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM bars WHERE symbol = ?", (symbol,))
                cursor.execute("DELETE FROM features WHERE symbol = ?", (symbol,))
                self._conn.commit()
            elif self.storage_type == "file":
                bars_file = self.base_path / "bars" / f"{symbol}.parquet"
                features_file = self.base_path / "features" / f"{symbol}_features.parquet"
                
                if bars_file.exists():
                    bars_file.unlink()
                if features_file.exists():
                    features_file.unlink()
            elif self.storage_type == "memory":
                if symbol in self._memory_data:
                    del self._memory_data[symbol]
                key = f"{symbol}_features"
                if key in self._memory_data:
                    del self._memory_data[key]
            
            logger.info(f"Deleted data for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to delete data for {symbol}: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the data store and cleanup resources."""
        try:
            if self.storage_type == "sqlite" and hasattr(self, '_conn'):
                self._conn.close()
            
            logger.info("Data store closed")
            
        except Exception as e:
            logger.error(f"Failed to close data store: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 