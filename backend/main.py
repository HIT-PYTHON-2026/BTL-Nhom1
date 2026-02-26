import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.middleware import LogMiddleware, setup_cors
from app.routers.base import router

app = FastAPI()

app.add_middleware(LogMiddleware)
setup_cors(app)
app.include_router(router)

# Phục vụ frontend tĩnh — mount SAU api routes để API ưu tiên hơn
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")