import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI
from app.middleware import LogMiddleware, setup_cors
from app.routers.base import router

app = FastAPI()

app.add_middleware(LogMiddleware)
setup_cors(app)
app.include_router(router)