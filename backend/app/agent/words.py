"""模型输出文本的清洗与防泄漏：只把能给用户看的那句吐出去。"""

import re

from langchain_core.messages import AIMessage

from app.tools import SPECS

REFUSAL_LIMIT = 60
# 闲聊的宽度上限：两句、120 字，防止它叭叭叭说一大堆
CHAT_SENTENCES = 2
CHAT_LIMIT = 120

# 按句切分，保留标点，便于只丢掉某几句
_SENTENCE = re.compile(r"(?<=[。！？])|[\r\n]+")
_TERMINATORS = "。！？…"


def text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    return "".join(
        block.get("text", "") for block in message.content if isinstance(block, dict)
    ).strip()


def mentions_tool(text: str) -> bool:
    """文字里出现内部工具名，说明模型在描述调用而不是在回答用户。"""
    return any(spec.name in text for spec in SPECS)


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split((text or "").strip()) if part.strip()]


def one_sentence(text: str) -> str:
    for sep in ("。", "！", "？", "\n"):
        head, found, _ = text.partition(sep)
        if found and head.strip():
            return head.strip() + (sep if sep != "\n" else "。")
    if len(text) > REFUSAL_LIMIT:
        return text[:REFUSAL_LIMIT].rstrip() + "…"
    return text


def plain(reply: str) -> str:
    """只留第一句不提内部工具名的话。

    模型拒绝时会顺带点名工具解释原因（「replace_region 需要完整提示词」），
    那种句子不能给用户看，但前面的正常说明要保住，不能整段作废。
    """
    for sentence in sentences(reply):
        if not mentions_tool(sentence):
            return one_sentence(sentence)
    return ""


def brief(reply: str) -> str:
    """闲聊的宽度限制：丢掉提工具名的句子，最多两句、120 字。"""
    kept = [sentence for sentence in sentences(reply) if not mentions_tool(sentence)]
    text = "".join(kept[:CHAT_SENTENCES])
    if len(text) > CHAT_LIMIT:
        text = text[:CHAT_LIMIT].rstrip("，,、；;。") + "…"
    return text.strip()


def ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in _TERMINATORS