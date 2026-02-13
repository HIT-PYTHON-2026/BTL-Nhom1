from pathlib import Path

class AppPath:
    BACKEND_DIR = Path(__file__).parent.parent
    
    LOG_DIR = BACKEND_DIR / "logs"
    
    CACHE_DIR = BACKEND_DIR / "cache"
    CAPTURED_DATA_DIR = CACHE_DIR / "capture_data"
    
AppPath.LOG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CACHE_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)
