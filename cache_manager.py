"""
Cache manager for frequently accessed data
Uses in-memory caching with TTL (Time To Live) for different data types
"""
import time
import hashlib
import json
from functools import wraps
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self):
        self._cache = {}
        self._default_ttl = {
            'market_data': 3600,  # 1 hour
            'weather_data': 1800,  # 30 minutes
            'coordinates': 86400,  # 24 hours
            'commodities': 3600,  # 1 hour
            'models': 0,  # No expiration (loaded once)
        }
    
    def _generate_key(self, cache_type: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        key_data = {
            'type': cache_type,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, cache_type: str, *args, **kwargs) -> Optional[Any]:
        """Get value from cache if not expired"""
        key = self._generate_key(cache_type, *args, **kwargs)
        
        if key in self._cache:
            entry = self._cache[key]
            ttl = entry.get('ttl', self._default_ttl.get(cache_type, 3600))
            
            # Check if expired
            if time.time() - entry['timestamp'] < ttl:
                logger.debug(f"Cache HIT for {cache_type}")
                return entry['value']
            else:
                # Expired, remove it
                del self._cache[key]
                logger.debug(f"Cache EXPIRED for {cache_type}")
        
        logger.debug(f"Cache MISS for {cache_type}")
        return None
    
    def set(self, cache_type: str, value: Any, ttl: Optional[int] = None, *args, **kwargs):
        """Set value in cache with optional custom TTL"""
        key = self._generate_key(cache_type, *args, **kwargs)
        ttl = ttl or self._default_ttl.get(cache_type, 3600)
        
        self._cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }
        logger.debug(f"Cache SET for {cache_type} (TTL: {ttl}s)")
    
    def clear(self, cache_type: Optional[str] = None):
        """Clear cache, optionally for a specific type"""
        if cache_type:
            # Clear all entries of this type
            keys_to_remove = [k for k, v in self._cache.items() 
                            if v.get('type') == cache_type]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for type: {cache_type}")
        else:
            self._cache.clear()
            logger.info("Cleared all cache")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_entries = len(self._cache)
        expired_count = 0
        current_time = time.time()
        
        for entry in self._cache.values():
            ttl = entry.get('ttl', 3600)
            if current_time - entry['timestamp'] >= ttl:
                expired_count += 1
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_count,
            'expired_entries': expired_count
        }

# Global cache instance
_cache_manager = CacheManager()

def cached(cache_type: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results
    
    Usage:
        @cached('market_data', ttl=3600)
        def fetch_market_data(state, district, crop):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached_value = _cache_manager.get(cache_type, *args, **kwargs)
            if cached_value is not None:
                return cached_value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                _cache_manager.set(cache_type, result, ttl, *args, **kwargs)
            
            return result
        return wrapper
    return decorator

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    return _cache_manager

