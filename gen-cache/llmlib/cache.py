"""Cache Key-Value store for the agent."""
from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class GlobalCache(BaseModel):
    """Global cache for the agent."""
    
    cache_dict: Dict[str, Any] = dict()
    """Cache dictionary."""

    cache_path: str = ""
    """Path to the cache file."""
    
    def __init__(self, **kwargs: Any):
        """Override init to support instantiation by position for backward compat."""
        super().__init__(**kwargs)


    def dump_cache(self):
        """Dump the cache to the cache file."""
        with open(self.cache_path, "w") as f:
            f.write(json.dumps(self.cache_dict))


    def load_cache(self):
        """Load the cache from the cache file."""
        try:
            with open(self.cache_path, "r") as f:
                self.cache_dict = json.loads(f.read())
        except FileNotFoundError:
            logger.debug(f"Cache file not found: {self.cache_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding cache file: {self.cache_path}")


    def set_cache(self, key: Any, value: Any):
        """Set the cache and dump to the cache file"""
        self.cache_dict[key] = value
        self.dump_cache()


    def get_cache(self, key: Any):
        """Get the cache from the cache file"""
        self.load_cache()
        if str(key) not in self.cache_dict:
            return None
        return self.cache_dict[str(key)]


    def get_entire_cache(self):
        """Get the entire cache from the cache file"""
        self.load_cache()
        return self.cache_dict
    

    def delete_cache(self, key: Any):
        """Delete the cache from the cache file"""
        self.load_cache()
        del self.cache_dict[str(key)]
        self.dump_cache()
