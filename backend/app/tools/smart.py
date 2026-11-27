from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.asset import AssetKind, AssetSource
from app.models.tool_run import ToolRun
from app.providers import EditRequest, get_image_provider
from app.services import assets, runs, selections
from app.tools.base import ToolSpec
from app.tools.context import flatten_session, require_session


class SmartEditIn(BaseModel):
    instruction: str = Field(min_length=1, max_length=1000)
    negative_prompt: str | None = None


async def smart_edit_exec(session: AsyncSession, run: ToolRun) -> dict:
    """拍平当前画布，按完整指令生成新图并直接采用。"""
    record = await require_session(session, run)
    instruction = run.params["instruction"].strip()

    await runs.report(session, run, 15, "读取当前画面")
    source = await flatten_session(session, record)

    await runs.report(session, run, 30, "智能编辑画面")
    prompt = (
        "根据用户指令编辑这张图片。只修改指令明确要求的内容，"
        "未提及的人物身份、面部、姿态、文字、背景、构图和光影保持原样。\n"
        f"用户指令：{instruction}"
    )

    images = await get_image_provider().edit(
        EditRequest(
            prompt=prompt,
            image=source,
            negative_prompt=run.params.get("negative_prompt"),
        ),
        on_progress=lambda progress, stage: runs.report(
            session, run, progress, stage
        ),
    )

    await runs.report(session, run, 90, "写回画布")
    asset = await assets.create_from_bytes(
        session,
        run.user_id,
        images[0],
        AssetKind.GENERATED,
        AssetSource.TOOL,
    )

    await selections.clear(record.id)

    return {
        "asset_ids": [str(asset.id)],
        "adopt_asset_id": str(asset.id),
    }


SMART_EDIT = ToolSpec(
    name="smart_edit",
    label="智能修图",
    description=(
        "直接按一句自然语言修改当前画面。换衣服或物体颜色、材质、人物、物体、"
        "文字内容，以及其他生成式修改都优先用它。工具会自动读取完整画布并采用结果，"
        "不需要图层 id，也不需要用户先拆层。"
    ),
    params=SmartEditIn,
    handler=smart_edit_exec,
    queued=True,
    session_required=True,
    agent_hidden=("negative_prompt",),
)