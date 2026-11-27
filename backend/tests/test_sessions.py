import uuid

import httpx
import pytest
from sqlalchemy import select

from app.db import SessionFactory
from app.models import EditHistory, SessionAsset
from app.services import selections
from tests.canvas import apply, invoke, scene
from tests.test_assets import make_image, upload_payload


@pytest.fixture
async def signed_in(client: httpx.AsyncClient, credentials):
    await client.post("/api/auth/register", json=credentials)
    return client


async def upload(client: httpx.AsyncClient, size=(320, 240), data: bytes | None = None) -> str:
    response = await client.post("/api/assets", files=upload_payload(data or make_image(size)))
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def open_session(client: httpx.AsyncClient, image: bytes | None = None, **overrides) -> dict:
    asset_id = await upload(client, data=image) if image else await upload(client)
    payload = {"current_asset_id": asset_id} | overrides
    response = await client.post("/api/sessions", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


async def test_new_session_starts_at_first_revision(signed_in: httpx.AsyncClient):
    body = await open_session(signed_in, title="  白底 主图  ")

    assert body["title"] == "白底 主图"
    assert body["revision"] == 1
    assert body["original_asset_id"] == body["current_asset_id"]


async def test_document_describes_current_image_as_base_layer(signed_in: httpx.AsyncClient):
    asset_id = await upload(signed_in, (400, 500))

    body = await open_session(signed_in, current_asset_id=asset_id)

    assert (body["document"]["width"], body["document"]["height"]) == (400, 500)
    [layer] = body["document"]["layers"]
    assert (layer["kind"], layer["asset_id"], layer["locked"]) == ("image", asset_id, True)


async def test_blank_title_falls_back_to_placeholder(signed_in: httpx.AsyncClient):
    assert (await open_session(signed_in, title="   "))["title"] == "未命名会话"


async def test_unadopted_candidates_stay_on_the_wall(signed_in: httpx.AsyncClient):
    current = await upload(signed_in)
    others = [await upload(signed_in) for _ in range(2)]

    body = await open_session(signed_in, current_asset_id=current, asset_ids=[current, *others])

    assert [asset["id"] for asset in body["assets"]] == [current, *others]


async def test_a_wall_entry_can_be_detached(signed_in: httpx.AsyncClient):
    """摘掉一张只少一个墙上的引用：图本身还在素材库里，画布修订号也不动。"""
    current = await upload(signed_in)
    extra = await upload(signed_in, (500, 500))
    body = await open_session(signed_in, current_asset_id=current, asset_ids=[current, extra])

    response = await signed_in.delete(f"/api/sessions/{body['id']}/assets/{extra}")

    assert response.status_code == 204
    detail = (await signed_in.get(f"/api/sessions/{body['id']}")).json()
    assert [asset["id"] for asset in detail["assets"]] == [current]
    assert detail["revision"] == body["revision"]
    assert (await signed_in.get(f"/api/assets/{extra}")).status_code == 200


async def test_current_image_cannot_be_detached(signed_in: httpx.AsyncClient):
    """画布上那张还挂在墙上，摘掉就没有切回它的入口，只能拒掉。"""
    body = await open_session(signed_in)

    response = await signed_in.delete(
        f"/api/sessions/{body['id']}/assets/{body['current_asset_id']}"
    )

    assert response.status_code == 409
    assert "画布" in response.json()["detail"]


async def test_asset_used_by_a_layer_cannot_be_detached(signed_in: httpx.AsyncClient):
    """拆层后主体那张只挂在图层上、并不是当前图，摘掉画布就凭空空出一层，同样要拒。"""
    body = await open_session(signed_in, image=scene())
    after = await apply(signed_in, body["id"], "split_layers")
    subject = next(
        layer["asset_id"] for layer in after["document"]["layers"] if layer["id"] == "subject"
    )

    response = await signed_in.delete(f"/api/sessions/{body['id']}/assets/{subject}")

    assert subject != after["current_asset_id"]
    assert response.status_code == 409


async def test_detaching_an_unknown_asset_is_rejected(signed_in: httpx.AsyncClient):
    body = await open_session(signed_in)

    response = await signed_in.delete(f"/api/sessions/{body['id']}/assets/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_switching_current_image_bumps_revision(signed_in: httpx.AsyncClient):
    body = await open_session(signed_in)
    other = await upload(signed_in, (500, 500))

    patched = (
        await signed_in.patch(f"/api/sessions/{body['id']}", json={"current_asset_id": other})
    ).json()

    assert patched["revision"] == 2
    assert patched["current_asset_id"] == other
    assert patched["document"]["width"] == 500
    assert other in [asset["id"] for asset in patched["assets"]]


async def test_switching_to_the_same_image_keeps_revision(signed_in: httpx.AsyncClient):
    body = await open_session(signed_in)

    patched = (
        await signed_in.patch(
            f"/api/sessions/{body['id']}", json={"current_asset_id": body["current_asset_id"]}
        )
    ).json()

    assert patched["revision"] == 1


async def test_picking_the_cropped_image_again_restores_its_full_size(signed_in: httpx.AsyncClient):
    """裁剪只缩小画布、不换资产，点回这张图应当回到原图画幅。"""
    session = await open_session(signed_in)
    cropped = (await invoke(signed_in, session["id"], "crop_canvas", {"ratio": "1:1"}))["session"]
    assert (cropped["document"]["width"], cropped["document"]["height"]) == (240, 240)

    patched = (
        await signed_in.patch(
            f"/api/sessions/{session['id']}",
            json={"current_asset_id": session["current_asset_id"]},
        )
    ).json()

    assert (patched["document"]["width"], patched["document"]["height"]) == (320, 240)
    assert patched["revision"] == cropped["revision"] + 1


async def test_history_records_creation_and_switches(signed_in: httpx.AsyncClient):
    body = await open_session(signed_in)
    other = await upload(signed_in)
    await signed_in.patch(f"/api/sessions/{body['id']}", json={"current_asset_id": other})

    entries = (await signed_in.get(f"/api/sessions/{body['id']}/history")).json()

    assert [entry["action"] for entry in entries] == ["switch_current", "create_session"]
    assert entries[0]["result"]["revision"] == 2


async def test_unknown_asset_is_rejected(signed_in: httpx.AsyncClient):
    response = await signed_in.post("/api/sessions", json={"current_asset_id": str(uuid.uuid4())})

    assert response.status_code == 404


async def test_session_can_be_deleted(signed_in: httpx.AsyncClient):
    """删成功后既查不到也不在列表里，再删一次只能给 404。"""
    body = await open_session(signed_in)

    assert (await signed_in.delete(f"/api/sessions/{body['id']}")).status_code == 204
    assert (await signed_in.get(f"/api/sessions/{body['id']}")).status_code == 404
    assert (await signed_in.get("/api/sessions")).json() == []
    assert (await signed_in.delete(f"/api/sessions/{body['id']}")).status_code == 404


async def test_deleting_a_session_leaves_no_orphan_rows(signed_in: httpx.AsyncClient):
    """会话删干净了：编辑记录与图片墙关联都跟着级联走，但图本身还在素材库里。"""
    body = await open_session(signed_in)
    await invoke(signed_in, body["id"], "crop_canvas", {"ratio": "1:1"})

    assert (await signed_in.delete(f"/api/sessions/{body['id']}")).status_code == 204

    session_id = uuid.UUID(body["id"])
    async with SessionFactory() as session:
        history = await session.scalars(
            select(EditHistory).where(EditHistory.session_id == session_id)
        )
        wall = await session.scalars(
            select(SessionAsset).where(SessionAsset.session_id == session_id)
        )
        assert list(history) == []
        assert list(wall) == []

    asset = await signed_in.get(f"/api/assets/{body['original_asset_id']}")
    assert asset.status_code == 200


async def test_deleting_a_session_drops_its_selection(signed_in: httpx.AsyncClient):
    """选区存在 Redis，不在数据库级联里，删会话时要顺手清掉，免得幽灵选区跑到新会话上。"""
    body = await open_session(signed_in)
    await signed_in.post(
        f"/api/sessions/{body['id']}/selection",
        json={"revision": body["revision"], "points": [{"x": 0.5, "y": 0.5}]},
    )
    session_id = uuid.UUID(body["id"])
    assert await selections.get(session_id, body["revision"]) is not None

    assert (await signed_in.delete(f"/api/sessions/{body['id']}")).status_code == 204
    assert await selections.get(session_id, body["revision"]) is None


async def test_session_requires_authentication(client: httpx.AsyncClient):
    assert (await client.get("/api/sessions")).status_code == 401
    assert (await client.get(f"/api/sessions/{uuid.uuid4()}")).status_code == 401
    assert (await client.delete(f"/api/sessions/{uuid.uuid4()}")).status_code == 401


async def test_other_user_cannot_delete_the_session(
    client: httpx.AsyncClient, credentials, other_credentials
):
    """删别人的会话与看不到它一样，给 404，不告诉对方这条存在。"""
    await client.post("/api/auth/register", json=credentials)
    session_id = (await open_session(client))["id"]

    await client.post("/api/auth/logout")
    await client.post("/api/auth/register", json=other_credentials)
    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 404

    await client.post("/api/auth/logout")
    await client.post("/api/auth/login", json=credentials)
    assert (await client.get(f"/api/sessions/{session_id}")).status_code == 200


async def test_sessions_are_isolated_per_user(
    client: httpx.AsyncClient, credentials, other_credentials
):
    await client.post("/api/auth/register", json=credentials)
    session_id = (await open_session(client))["id"]

    await client.post("/api/auth/logout")
    await client.post("/api/auth/register", json=other_credentials)

    assert (await client.get(f"/api/sessions/{session_id}")).status_code == 404
    assert (await client.get("/api/sessions")).json() == []
