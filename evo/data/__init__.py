"""
Data layer for the EVO trading system.

This module provides abstractions for data providers, streaming, processing,
and storage components.
"""

from .providers.base_provider import BaseDataProvider
from .streamers.base_streamer import BaseStreamer
from .processors.feature_engineer import FeatureEngineer
from .processors.normalizer import DataNormalizer
from .storage.data_store import DataStore

__all__ = [
    'BaseDataProvider',
    'BaseStreamer', 
    'FeatureEngineer',
    'DataNormalizer',
    'DataStore'
] 