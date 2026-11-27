import uuid
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from app.lesion import engine
from app.tasks.tools import run_tool
from tests.canvas import error_of, invoke, scene
from tests.test_sessions import make_image, open_session


@pytest.fixture
async def signed_in(client: httpx.AsyncClient, credentials):
    await client.post("/api/auth/register", json=credentials)
    return client


def _png(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """box 为 (top, bottom, left, right)。"""
    mask = np.zeros(size, dtype=bool)
    mask[box[0] : box[1], box[2] : box[3]] = True
    return mask


def test_prepare_matches_training_pipeline():
    sample, size = engine.prepare(_png((1280, 960), (40, 120, 200)))

    assert size == (1280, 960)
    assert sample.shape == (engine.SIZE, engine.SIZE)
    assert sample.dtype == np.float32
    # 训练脚本直接 .float()，没有除 255：绿通道原值应保留在 0~255 区间。
    # CLAHE 会微调亮度，只验证量级不比对精确值
    assert sample.max() > 1
    assert sample[0, 0] == pytest.approx(120, abs=10)


def test_prepare_keeps_aspect_by_resizing_the_short_side():
    # 竖图短边是宽：720x1280 → 642x1142 → 中心裁 640x640
    sample, size = engine.prepare(_png((720, 1280), (10, 20, 30)))

    assert size == (720, 1280)
    assert sample.shape == (engine.SIZE, engine.SIZE)


def test_restore_puts_the_mask_back_at_the_crop_offset():
    restored = engine.restore(_mask((640, 640), (0, 100, 0, 100)), (800, 1200))

    assert restored.shape == (1200, 800)
    # 短边 800 → 642x963，纵向裁掉 162 行；映射回 1200 高后病灶从 200 行左右开始
    assert not restored[:150].any()
    assert restored[195:215, 0:8].any()


def test_restore_round_trips_on_a_square_image():
    original = _mask((640, 640), (200, 300, 200, 300))
    restored = engine.restore(original, (512, 512))
    back = np.asarray(
        Image.fromarray(restored.astype(np.uint8), "L").resize((640, 640), Image.NEAREST)
    )

    assert back[260, 260] == 1
    assert back[10, 10] == 0


def test_overlay_paints_only_the_mask_in_rgba():
    mask = _mask((100, 200), (10, 20, 30, 40))

    image = Image.open(__import__("io").BytesIO(engine.overlay(mask, (255, 0, 0))))

    assert image.mode == "RGBA"
    assert image.size == (200, 100)
    assert image.getpixel((35, 15)) == (255, 0, 0, 255)
    assert image.getpixel((0, 0))[3] == 0


def test_available_reports_missing_weights(monkeypatch):
    monkeypatch.setattr(engine, "model_paths", lambda: {"EX": Path("/nonexistent/ex.pth.tar")})

    ok, reason = engine.available()

    assert ok is False
    assert "缺少" in (reason or "")


def test_status_lists_every_type(monkeypatch):
    monkeypatch.setattr(engine, "model_paths", lambda: {})

    body = engine.status()

    assert body["available"] is False
    assert set(body["models"]) == set(engine.TYPES)
    assert not any(body["models"].values())
    assert body["reason"]


async def test_status_endpoint_is_reachable(client: httpx.AsyncClient):
    response = await client.get("/api/lesion/status")

    assert response.status_code == 200
    assert set(response.json()["models"]) == set(engine.TYPES)


async def test_segment_without_gpu_fails_with_a_readable_reason(
    signed_in: httpx.AsyncClient, monkeypatch
):
    monkeypatch.setattr(engine, "available", lambda: (False, engine.NO_GPU))
    session = await open_session(signed_in, image=scene())

    body = await invoke(signed_in, session["id"], "segment_lesion", {"types": ["EX"]})
    await run_tool({}, uuid.UUID(body["run"]["id"]))

    assert "GPU" in await error_of(signed_in, body["run"]["id"])
    unchanged = (await signed_in.get(f"/api/sessions/{session['id']}")).json()
    assert unchanged["revision"] == 1
    assert len(unchanged["document"]["layers"]) == 1


async def test_segment_appends_one_overlay_per_type(signed_in: httpx.AsyncClient, monkeypatch):
    calls: list[float] = []

    def fake_predict(data: bytes, types, threshold: float):
        calls.append(threshold)
        return {name: _mask((240, 320), (80, 140, 100, 220)) for name in types}

    monkeypatch.setattr(engine, "available", lambda: (True, None))
    monkeypatch.setattr(engine, "predict", fake_predict)
    session = await open_session(signed_in, image=make_image((320, 240)))

    body = await invoke(
        signed_in,
        session["id"],
        "segment_lesion",
        {"types": ["EX", "MA"], "threshold": 0.7},
    )
    await run_tool({}, uuid.UUID(body["run"]["id"]))

    updated = (await signed_in.get(f"/api/sessions/{session['id']}")).json()
    overlays = [
        layer for layer in updated["document"]["layers"] if layer["id"].startswith("lesion-")
    ]

    assert calls == [0.7]
    assert updated["revision"] == 2
    assert [layer["id"] for layer in overlays] == ["lesion-ex", "lesion-ma"]
    assert overlays[0]["name"] == "EX 出血点"
    assert (overlays[0]["width"], overlays[0]["height"]) == (320, 240)
    assert overlays[0]["opacity"] == 0.55
    assert updated["document"]["layers"][0]["id"] == "base"


async def test_rerunning_replaces_instead_of_stacking(signed_in: httpx.AsyncClient, monkeypatch):
    monkeypatch.setattr(engine, "available", lambda: (True, None))
    monkeypatch.setattr(
        engine,
        "predict",
        lambda data, types, threshold: {
            name: _mask((240, 320), (10, 20, 10, 20)) for name in types
        },
    )
    session = await open_session(signed_in, image=make_image((320, 240)))
    first = await invoke(signed_in, session["id"], "segment_lesion", {"types": ["EX"]})
    await run_tool({}, uuid.UUID(first["run"]["id"]))

    second = await invoke(signed_in, session["id"], "segment_lesion", {"types": ["EX"]})
    await run_tool({}, uuid.UUID(second["run"]["id"]))

    updated = (await signed_in.get(f"/api/sessions/{session['id']}")).json()
    overlays = [
        layer for layer in updated["document"]["layers"] if layer["id"].startswith("lesion-")
    ]

    assert [layer["id"] for layer in overlays] == ["lesion-ex"]


async def test_empty_result_reports_instead_of_adding_blank_layers(
    signed_in: httpx.AsyncClient, monkeypatch
):
    monkeypatch.setattr(engine, "available", lambda: (True, None))
    monkeypatch.setattr(
        engine,
        "predict",
        lambda data, types, threshold: {name: np.zeros((240, 320), dtype=bool) for name in types},
    )
    session = await open_session(signed_in, image=make_image((320, 240)))

    body = await invoke(signed_in, session["id"], "segment_lesion", {"types": ["EX"]})
    await run_tool({}, uuid.UUID(body["run"]["id"]))

    assert "未检出" in await error_of(signed_in, body["run"]["id"])
    unchanged = (await signed_in.get(f"/api/sessions/{session['id']}")).json()
    assert len(unchanged["document"]["layers"]) == 1


async def test_unknown_type_is_rejected_before_queueing(signed_in: httpx.AsyncClient):
    session = await open_session(signed_in)

    response = await signed_in.post(
        f"/api/sessions/{session['id']}/tools",
        json={"tool": "segment_lesion", "params": {"types": ["XX"]}},
    )

    assert response.status_code == 422
