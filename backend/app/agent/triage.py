"""通道分流：判断这句话是修图指令，还是与修图无关的提问。

判错一律偏向修图通道：那正是改动前的唯一行为，最多多说一句解释；
反过来把修图判成闲聊会直接丢掉工具调用，代价大得多。

一句话里既有提问又有修图动作时，`split` 负责把它们拆开，各走各的通道。
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import triager
from app.agent.words import text_of

logger = logging.getLogger(__name__)

EDIT = "edit"
CHAT = "chat"
# 拆分后的第三条通道：修图段照常出计划，文字段照常回答，两段结果各自独立
MIXED = "mixed"

_SYSTEM = """用户在使用一个 AI 修图产品，界面上已经打开了一张图片。判断他这句话要不要动图片。

只回答一个词，不要标点、不要解释：
- 要修图、抠图、调色、变换、改图层或选区、生成或输出图片、或者在谈论当前这张图 → edit
- 与图片完全无关，只是提问、闲聊、让人写东西、查资料、聊别的软件 → chat"""

# 命中即判定修图，省掉一次分流模型调用。只放图片专有词，刻意避开「帮/写/说/生成」这类通用动词。
_HINTS = (
    "图片", "照片", "这张", "那张", "这幅", "画面", "画布", "素材", "主体", "背景", "前景",
    "图层", "选区", "蒙版", "文字层", "文案", "字体", "字号", "logo",
    "修图", "抠图", "去背", "磨皮", "美颜", "精修", "美化", "换背景", "扩图", "超分",
    "病灶", "出血", "渗出", "血管瘤", "眼底", "视网膜", "分割",
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


# 只在两个动作之间起衔接作用的连接词。刻意不收「再」与「并且」，
# 它们更多出现在一句修图操作内部（翻转再放大、缩小并且居中）。
_SPLITTERS = ("然后", "接着", "并把", "同时", "顺便", "之后", "另外", "还有")
_SPLIT = re.compile(
    r"[。！？!?；;\n]+|，|,|" + "|".join(re.escape(word) for word in _SPLITTERS)
)

# 混合轮只认「明显是在提问」的那一段。没有这层判据，
# 「把衣服改成红色，保持背景不变」这种后半句会被误判成闲聊，把修图操作拆掉。
# 除疑问句式外还要收祈使句式的提问（讲一下/介绍一下/聊聊）：它们没有问号也没有「是谁」，
# 但确实在要一个答案。认不出它们时整句会原样进规划模型，而规划模型面对
# 「提问+修图」的自相矛盾输入会在「用散文回答」与「调用工具」之间摇摆，
# 同一种句式于是有时只答、有时只修。
_ASKS = (
    "告诉我", "是什么", "什么意思", "为什么", "怎么", "如何", "解释", "说说", "讲讲",
    "讲一下", "讲一讲", "说一下", "说一说", "介绍", "聊聊", "谈谈", "科普", "请问",
    "是谁", "多少", "哪个", "哪些", "有没有", "能不能", "吗", "呢", "？", "?",
)


def pieces_of(goal: str) -> list[str]:
    """按句末标点、逗号与连接词切句。切不出两段就说明这句话不该拆。"""
    parts = _SPLIT.split((goal or "").strip())
    return [piece for piece in (part.strip("，, 、") for part in parts) if len(piece) > 1]


def looks_like_ask(text: str) -> bool:
    return any(word in text for word in _ASKS)


async def split(goal: str, judge=None) -> tuple[str, str] | None:
    """句子明显可拆时返回（修图段，文字段），否则返回 None 走单通道。

    判据同时成立才拆：拆得出多段、一段命中修图词、另一段确实像在问东西。
    全是修图时绝不拆，多步编排要留给规划模型一次输出，拆开反而会把一步做成两步。

    这三条都是字符串判据，花不到模型调用。前置它们是为了不让
    「弄亮一点再锐化一下」这种纯修图句白烧几 token 逐段问分流模型。

    judge 只用于测试注入，默认逐段走 `classify`，命中关键词的段落不会花模型调用。
    """
    judge = judge or classify
    pieces = pieces_of(goal)
    if len(pieces) < 2:
        return None
    if not any(looks_like_ask(piece) for piece in pieces):
        return None
    if not any(looks_like_edit(piece) for piece in pieces):
        return None

    kinds = [await judge(piece) for piece in pieces]
    if EDIT not in kinds or CHAT not in kinds:
        return None

    asks = [piece for piece, kind in zip(pieces, kinds, strict=True) if kind == CHAT]
    if not any(looks_like_ask(piece) for piece in asks):
        return None

    def joined(wanted: str) -> str:
        """同通道的片段拼回一句：修图段仍由规划模型一次给出全部步骤。"""
        return "，".join(
            piece for piece, kind in zip(pieces, kinds, strict=True) if kind == wanted
        )

    return joined(EDIT), joined(CHAT)