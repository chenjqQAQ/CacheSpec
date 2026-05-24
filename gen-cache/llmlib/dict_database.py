from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional
from collections import defaultdict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Database(BaseModel):

    db_path: str = "/home/sarthak/Agent-RCA/data/database.pkl"
    """Path to the database file."""

    db_dict: defaultdict = defaultdict(dict)
    """Temporary Database dictionary."""

    def __init__(self, **kwargs: Any):
        """ Override init to support instantiation by position for backward compat. """
        super().__init__(**kwargs)
        self.load_db()


    def dump_db(self):
        """Dump the database to a file."""
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load_db(self):
        """Load the database from the file."""
        try:
            with open(self.db_path, "rb") as f:
                self.db_dict = pickle.load(f)
        except:
            logger.error(f"Error decoding Database file: {self.db_path}")


    def set_db(self, key: str, value: Any):
        """ Set the database value for a corresponding key """
        self.db_dict[key].update(value)
        self.dump_db()

    
    def get_db(self, key: str):
        """ Get the database value for the corresponding key """
        if key not in self.db_dict:
            return None
        
        return self.db_dict[key]
    

    def len(self, key: str):
        """ Checks the number of records in the database with the same key """
        return len(self.db_dict[key])