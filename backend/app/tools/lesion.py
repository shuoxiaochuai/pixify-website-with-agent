import asyncio
from typing import Literal

from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.layers import Layer, LayerKind
from app.lesion import engine
from app.models.asset import AssetKind, AssetSource
from app.models.tool_run import ToolRun
from app.services import assets, runs
from app.tools.base import LayerRef, ToolSpec
from app.tools.context import ToolError, document_of, require_session
from app.tools.target import layer_image, resolve_target

OPACITY = 0.55
# 图层来源标记。重跑同一类型时先摘掉旧标注，避免半透明红层叠成一片死红。
SOURCE = "lesion:"


class SegmentLesionIn(LayerRef):
    types: list[Literal["EX", "HE", "MA", "SE"]] = Field(
        default_factory=lambda: list(engine.TYPES),
        description="要分割的病灶：EX 出血点、HE 硬性渗出、MA 微血管瘤、SE 软性渗出",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.05,
        le=0.99,
        description="概率阈值，越高越严格；不填用服务端默认值",
    )


async def segment_lesion_exec(session: AsyncSession, run: ToolRun) -> dict:
    ok, reason = engine.available()
    if not ok:
        raise ToolError(reason or engine.NO_GPU)

    record = await require_session(session, run)
    document = document_of(record).model_copy(deep=True)
    layer = resolve_target(document, run.params.get("layer_id"))
    wanted = list(dict.fromkeys(run.params.get("types") or list(engine.TYPES)))
    threshold = float(run.params.get("threshold") or get_settings().lesion_threshold)

    await runs.report(session, run, 10, "读取图层")
    data = await layer_image(session, record, layer)

    await runs.report(session, run, 30, "病灶推理中")
    try:
        masks = await asyncio.to_thread(engine.predict, data, wanted, threshold)
    except engine.Unavailable as exc:
        raise ToolError(str(exc)) from exc

    await runs.report(session, run, 80, "生成标注图层")
    stale = {f"{SOURCE}{name}" for name in wanted}
    document.layers = [item for item in document.layers if item.source_hash not in stale]

    created: list[str] = []
    for name in wanted:
        mask = masks.get(name)
        if mask is None or not mask.any():
            continue
        height, width = mask.shape
        png = await asyncio.to_thread(engine.overlay, mask, engine.COLORS[name])
        asset = await assets.create_from_bytes(
            session, run.user_id, png, AssetKind.GENERATED, AssetSource.TOOL
        )
        document.layers.append(
            Layer(
                id=f"lesion-{name.lower()}",
                kind=LayerKind.IMAGE,
                name=f"{name} {engine.LABELS[name]}",
                width=width,
                height=height,
                asset_id=asset.id,
                # 标注跟随被分割图层的摆放，缩放旋转过的图也能对齐
                transform=layer.transform.model_copy(),
                opacity=OPACITY,
                source_hash=f"{SOURCE}{name}",
            )
        )
        created.append(str(asset.id))

    if not created:
        raise ToolError("未检出所选类型的病灶，可换更低的阈值再试")

    await runs.report(session, run, 95, "写回画布")
    return {"document": document.model_dump(mode="json"), "asset_ids": created}


SEGMENT_LESION = ToolSpec(
    name="segment_lesion",
    label="病灶分割",
    description=(
        "在眼底图上分割糖尿病视网膜病灶，每类生成一个半透明彩色标注图层（可单独显隐、调透明度、撤销）。"
        "types 取值：EX 出血点、HE 硬性渗出、MA 微血管瘤、SE 软性渗出，不填则四类全跑。"
        "只适用于眼底/视网膜照片；普通商品图不要用它。"
    ),
    params=SegmentLesionIn,
    handler=segment_lesion_exec,
    queued=True,
    session_required=True,
)
