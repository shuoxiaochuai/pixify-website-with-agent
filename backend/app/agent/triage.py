"""通道分流：判断这句话是修图指令，还是与修图无关的提问。

判错一律偏向修图通道：那正是改动前的唯一行为，最多多说一句解释；
反过来把修图判成闲聊会直接丢掉工具调用，代价大得多。
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import triager
from app.agent.words import text_of

logger = logging.getLogger(__name__)

EDIT = "edit"
CHAT = "chat"

_SYSTEM = """用户在使用一个 AI 修图产品，界面上已经打开了一张图片。判断他这句话要不要动图片。

只回答一个词，不要标点、不要解释：
- 要修图、抠图、调色、变换、改图层或选区、生成或输出图片、或者在谈论当前这张图 → edit
- 与图片完全无关，只是提问、闲聊、让人写东西、查资料、聊别的软件 → chat"""

# 命中即判定修图，省掉一次分流模型调用。只放图片专有词，刻意避开「帮/写/说/生成」这类通用动词。
_HINTS = (
    "图片", "照片", "这张", "那张", "这幅", "画面", "画布", "素材", "主体", "背景", "前景",
    "图层", "选区", "蒙版", "文字层", "文案", "字体", "字号", "logo",
    "修图", "抠图", "去背", "磨皮", "美颜", "精修", "美化", "换背景", "扩图", "超分",
    "翻转", "镜像", "旋转", "裁剪", "裁成", "调色", "调亮", "调暗", "饱和", "色温",
    "放大", "缩小", "移动", "透明", "白底", "阴影", "水印", "马赛克", "消除",
    "改成", "换成", "换个", "改色", "去掉", "修改", "一张",
    "主图", "详情页", "海报", "场景图", "模特图", "营销图", "投放", "尺寸", "比例",
    "批量", "导出", "生成图", "出图", "做图",
)


def looks_like_edit(goal: str) -> bool:
    text = goal.lower()
    return any(word in text for word in _HINTS)


async def classify(goal: str) -> str:
    """返回 EDIT 或 CHAT。关键词命中就不花一次模型调用。"""
    if looks_like_edit(goal):
        return EDIT
    try:
        verdict = text_of(await triager().ainvoke([SystemMessage(_SYSTEM), HumanMessage(goal)]))
    except Exception:
        # 分流模型坏了也不能让整轮指令挂掉：退回修图通道，改动前的行为就是这样，
        # 真修不了的话规划模型会给出准确原因
        logger.warning("分流模型不可用，按修图通道处理 goal=%s", goal[:40], exc_info=True)
        return EDIT
    return CHAT if verdict.lower().startswith("chat") else EDIT