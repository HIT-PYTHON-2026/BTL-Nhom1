from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()

@router.get("/")
async def redirect_to_docs():
    return RedirectResponse("http://127.0.0.1:5000/docs")