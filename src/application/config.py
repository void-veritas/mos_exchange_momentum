import os
import logging
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / '.env'
load_dotenv(dotenv_path=env_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """Application configuration"""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')
    
    # Database
    DB_PATH = os.path.join(BASE_DIR, 'moex_prices.db')
    
    # Data source settings
    DEFAULT_INDEX_ID = "IMOEX"
    
    # API settings
    MOEX_BASE_URL = "https://iss.moex.com/iss"
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")
    TINKOFF_API_TOKEN = os.getenv("TINKOFF_API_TOKEN", "")
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_EXPIRY_SECONDS = 3600  # 1 hour
    
    # SSL verification
    VERIFY_SSL = True
    
    # Default values
    DEFAULT_PRICE_SOURCES = ["FMP", "YAHOO", "MOEX", "TINKOFF", "FINAM", "INVESTING"]
    
    @classmethod
    def init(cls):
        """Initialize configuration (create directories, etc.)"""
        # Create cache directory if it doesn't exist
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        logger.info(f"Cache directory: {cls.CACHE_DIR}")
        
        # Log configuration
        logger.info(f"Database path: {cls.DB_PATH}")
        logger.info(f"Cache enabled: {cls.CACHE_ENABLED}")
        logger.info(f"Default index ID: {cls.DEFAULT_INDEX_ID}")
        
        # Log API key status (not the actual keys for security)
        logger.info(f"FMP API key configured: {bool(cls.FMP_API_KEY)}")
        logger.info(f"Tinkoff API token configured: {bool(cls.TINKOFF_API_TOKEN)}")
    
    @classmethod
    def get_env(cls, name: str, default: Any = None) -> Any:
        """Get environment variable with default"""
        return os.environ.get(name, default)
    
    @classmethod
    def load_from_env(cls):
        """Load configuration from environment variables"""
        # Database settings
        cls.DB_PATH = cls.get_env("DB_PATH", cls.DB_PATH)
        
        # Cache settings
        cls.CACHE_ENABLED = cls.get_env("CACHE_ENABLED", "True").lower() in ("true", "1", "yes")
        cls.CACHE_EXPIRY_SECONDS = int(cls.get_env("CACHE_EXPIRY_SECONDS", cls.CACHE_EXPIRY_SECONDS))
        
        # API settings
        cls.VERIFY_SSL = cls.get_env("VERIFY_SSL", "True").lower() in ("true", "1", "yes")
        
        # Default values
        sources_str = cls.get_env("DEFAULT_PRICE_SOURCES")
        if sources_str:
            cls.DEFAULT_PRICE_SOURCES = [s.strip() for s in sources_str.split(",")]
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "BASE_DIR": cls.BASE_DIR,
            "CACHE_DIR": cls.CACHE_DIR,
            "DB_PATH": cls.DB_PATH,
            "DEFAULT_INDEX_ID": cls.DEFAULT_INDEX_ID,
            "MOEX_BASE_URL": cls.MOEX_BASE_URL,
            "FMP_API_KEY_CONFIGURED": bool(cls.FMP_API_KEY),
            "TINKOFF_API_TOKEN_CONFIGURED": bool(cls.TINKOFF_API_TOKEN),
            "CACHE_ENABLED": cls.CACHE_ENABLED,
            "CACHE_EXPIRY_SECONDS": cls.CACHE_EXPIRY_SECONDS,
            "VERIFY_SSL": cls.VERIFY_SSL,
            "DEFAULT_PRICE_SOURCES": cls.DEFAULT_PRICE_SOURCES
        }


# Load configuration from environment
Config.load_from_env() 