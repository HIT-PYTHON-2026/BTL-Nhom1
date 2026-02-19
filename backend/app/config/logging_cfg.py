from pathlib import Path

class LoggingConfig:
    ROOT_DIR = Path(__file__).parent.parent.parent
    
    LOG_DIR = ROOT_DIR / 'app' / "logs"
    
LoggingConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)