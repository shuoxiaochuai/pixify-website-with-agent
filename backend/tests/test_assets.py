import io

import httpx
import pytest
from PIL import Image

from app.services.images import ImageRejected, probe


def make_image(size=(256, 256), mode="RGB", fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "white").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
async def signed_in(client: httpx.AsyncClient, credentials):
    await client.post("/api/auth/register", json=credentials)
    return client


def upload_payload(data: bytes, name="a.png", content_type="image/png"):
    return {"file": (name, data, content_type)}


async def test_upload_returns_metadata_and_file_url(signed_in: httpx.AsyncClient):
    response = await signed_in.post("/api/assets", files=upload_payload(make_image((320, 200))))

    assert response.status_code == 201
    body = response.json()
    assert (body["width"], body["height"]) == (320, 200)
    assert body["image_format"] == "PNG"
    assert body["kind"] == "original"
    # 图片走同源路径，反代与隧道下都不必再暴露对象存储
    assert body["url"] == f"/api/assets/{body['id']}/file"


async def test_asset_file_returns_same_origin_binary(signed_in: httpx.AsyncClient):
    """返回的 url 直接就能读到字节，Content-Type 与落库格式一致。"""
    asset = (await signed_in.post("/api/assets", files=upload_payload(make_image()))).json()

    response = await signed_in.get(asset["url"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


async def test_asset_file_rejects_other_users_asset(
    client: httpx.AsyncClient, credentials, other_credentials
):
    await client.post("/api/auth/register", json=credentials)
    asset_id = (await client.post("/api/assets", files=upload_payload(make_image()))).json()["id"]

    await client.post("/api/auth/logout")
    await client.post("/api/auth/register", json=other_credentials)

    assert (await client.get(f"/api/assets/{asset_id}/file")).status_code == 404


async def test_upload_detects_alpha_channel(signed_in: httpx.AsyncClient):
    response = await signed_in.post("/api/assets", files=upload_payload(make_image(mode="RGBA")))

    assert response.json()["has_alpha"] is True


async def test_upload_rejects_corrupted_file(signed_in: httpx.AsyncClient):
    response = await signed_in.post("/api/assets", files=upload_payload(b"not-an-image"))

    assert response.status_code == 422


async def test_upload_rejects_unsupported_format(signed_in: httpx.AsyncClient):
    gif = make_image(fmt="GIF")

    response = await signed_in.post("/api/assets", files=upload_payload(gif, "a.gif", "image/gif"))

    assert response.status_code == 422


async def test_upload_ignores_spoofed_extension(signed_in: httpx.AsyncClient):
    """格式以解码结果为准：JPEG 内容伪装成 .png 仍应按 JPEG 记录。"""
    response = await signed_in.post(
        "/api/assets", files=upload_payload(make_image(fmt="JPEG"), "a.png", "image/png")
    )

    assert response.json()["image_format"] == "JPEG"


async def test_upload_requires_authentication(client: httpx.AsyncClient):
    response = await client.post("/api/assets", files=upload_payload(make_image()))

    assert response.status_code == 401


async def test_library_groups_assets_by_session_and_hides_masks(signed_in: httpx.AsyncClient):
    loose = await signed_in.post("/api/assets", files=upload_payload(make_image((320, 200))))
    session = await signed_in.post(
        "/api/sessions", json={"current_asset_id": loose.json()["id"], "title": "作品"}
    )
    extra = await signed_in.post("/api/assets", files=upload_payload(make_image((400, 300))))
    await signed_in.post("/api/assets", files=upload_payload(make_image((256, 256), mode="RGBA")))
    await signed_in.post(
        f"/api/sessions/{session.json()['id']}/selection",
        json={"revision": 1, "points": [{"x": 0.5, "y": 0.5}]},
    )

    groups = (await signed_in.get("/api/assets/library")).json()

    assert [group["title"] for group in groups] == ["作品", "未归入会话"]
    assert groups[0]["session_id"] == session.json()["id"]
    assert extra.json()["id"] in [asset["id"] for asset in groups[1]["assets"]]
    assert all(asset["kind"] != "mask" for group in groups for asset in group["assets"])


async def test_assets_are_isolated_per_user(
    client: httpx.AsyncClient, credentials, other_credentials
):
    await client.post("/api/auth/register", json=credentials)
    created = await client.post("/api/assets", files=upload_payload(make_image()))
    asset_id = created.json()["id"]

    await client.post("/api/auth/logout")
    await client.post("/api/auth/register", json=other_credentials)

    assert (await client.get(f"/api/assets/{asset_id}")).status_code == 404
    assert (await client.get("/api/assets")).json() == []


def test_probe_rejects_tiny_image():
    with pytest.raises(ImageRejected):
        probe(make_image((16, 16)))
