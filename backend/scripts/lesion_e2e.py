"""病灶分割端到端验证：真图上传、segment_lesion 入队、SSE 进度、标注图层落库。"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:7302"
USER = {"username": "lesiontester", "password": "lesion123"}
IMAGE = Path("/root/rivermind-data/Segmentation_dataset3/test_images/IDRiD_55.jpg")


async def main() -> int:
    data = IMAGE.read_bytes()
    async with httpx.AsyncClient(base_url=BASE, timeout=60.0, trust_env=False) as client:
        response = await client.post("/api/auth/register", json=USER)
        if response.status_code == httpx.codes.CONFLICT:
            response = await client.post("/api/auth/login", json=USER)
        assert response.status_code in (200, 201), response.text
        print(f"[1] 登录成功 {response.json()['username']}")

        response = await client.post(
            "/api/assets", files={"file": (IMAGE.name, data, "image/jpeg")}
        )
        assert response.status_code == 201, response.text
        asset = response.json()
        print(f"[2] 上传 {IMAGE.name} {asset['width']}x{asset['height']}")

        response = await client.post("/api/sessions", json={"current_asset_id": asset["id"]})
        assert response.status_code == 201, response.text
        session = response.json()
        print(f"[3] 会话 {session['id']} revision={session['revision']}")

        response = await client.post(
            f"/api/sessions/{session['id']}/tools",
            json={"tool": "segment_lesion", "params": {"types": ["EX", "MA"], "threshold": 0.8}},
        )
        assert response.status_code in (200, 202), response.text
        body = response.json()
        run_id = body["run"]["id"]
        print(f"[4] 入队 run={run_id} status={body['run']['status']}")

        frames = []
        async with client.stream("GET", f"/events/runs/{run_id}", timeout=180.0) as stream:
            assert stream.status_code == httpx.codes.OK, stream.status_code
            async for line in stream.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                frames.append(payload)
                print(
                    f"    SSE {payload['status']:<10} {payload['progress']:>3}% {payload['stage']}"
                )
                if payload["status"] in ("succeeded", "failed", "canceled"):
                    break
        assert frames, "未收到任何 SSE 帧"
        assert frames[-1]["status"] == "succeeded", frames[-1]

        response = await client.get(f"/api/sessions/{session['id']}")
        assert response.status_code == 200, response.text
        detail = response.json()
        layers = [
            layer for layer in detail["document"]["layers"] if layer["id"].startswith("lesion-")
        ]
        assert [layer["id"] for layer in layers] == ["lesion-ex", "lesion-ma"], layers
        assert (layers[0]["width"], layers[0]["height"]) == (asset["width"], asset["height"])
        print(
            f"[5] 图层 {[layer['id'] for layer in layers]} opacity={layers[0]['opacity']} "
            f"revision={detail['revision']}"
        )

        annotated = next(item for item in detail["assets"] if item["id"] == layers[0]["asset_id"])
        # 资产 URL 是站内相对路径且要登录态，用同一个 client 下（前端也是同源 cookie）
        url = str(httpx.URL(BASE).join(annotated["url"]))
        png = await client.get(url)
        assert png.status_code == httpx.codes.OK, png.status_code
        assert png.content[:8] == b"\x89PNG\r\n\x1a\n", "标注图不是 PNG"
        print(f"[6] 标注图下载 OK {len(png.content)} 字节")

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
