"""病灶分割能力探测。真正的执行走通用的 POST /api/sessions/{id}/tools。"""

from fastapi import APIRouter

from app.lesion import engine

router = APIRouter(prefix="/lesion", tags=["lesion"])


@router.get("/status")
def lesion_status() -> dict:
    """同步函数：torch 首次导入要一两秒，交给线程池，不占事件循环。"""
    return engine.status()
