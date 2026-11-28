"""病灶分割能力探测、真值标注上传。真正的执行走通用的 POST /api/sessions/{id}/tools。"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.db import SessionDep
from app.deps import CurrentUser
from app.lesion import engine, masks
from app.models.asset import AssetKind, AssetSource
from app.schemas.asset import AssetOut
from app.services import assets

router = APIRouter(prefix="/lesion", tags=["lesion"])


@router.get("/status")
def lesion_status() -> dict:
    """同步函数：torch 首次导入要一两秒，交给线程池，不占事件循环。"""
    return engine.status()


@router.post("/masks", status_code=status.HTTP_201_CREATED)
async def upload_mask(
    user: CurrentUser,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
) -> AssetOut:
    """上传真实标注（.tif/.png 均可）。归一成灰度 PNG 按 mask 类型入库，不进素材墙。"""
    data = await file.read()
    try:
        png = masks.normalize(data)
    except masks.MaskRejected as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    asset = await assets.create_from_bytes(
        session, user.id, png, AssetKind.MASK, AssetSource.UPLOAD
    )
    return AssetOut.of(asset)
