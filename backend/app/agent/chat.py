"""闲聊通道：回答与修图无关的问题，只答两句，不附加任何引导语。

这里不绑工具、不给画布摘要，所以既不会生成计划，也不会污染修图上下文。
往外产的是逐句增量，拼起来必须与最终落库的那句完全一致，否则收尾时画面会跳变。
"""

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import PlannerUnavailable, chatter
from app.agent.words import brief, delta_of, ends_sentence, until_sentence

# 模型答不上来（或只说了句含工具名的话）时的保底，绝不回落到「没太理解这条指令」
FALLBACK = "这个我不太确定。"

_SYSTEM = """你在一个 AI 修图产品里陪用户聊天，用户这句话和修图无关。

规则：
- 用自己的知识直接回答，最多两句话。不分点、不举例、不展开背景介绍、不给步骤。
- 不确定就直说不确定，禁止编造。
- 不要复述用户的问题，不要用「好的」「当然可以」这类客套开头。
- 不要提任何工具名、参数或系统内部实现，也不要自称通用大模型。
- 界面不渲染 LaTeX，公式直接用普通文本写：E=mc²、1/2、√2 这种写法。
- 不要声明自己更擅长修图，也不要引导用户去提修图需求，答完就停。"""


async def answer(goal: str) -> AsyncIterator[str]:
    """一句一句产出给用户看的文本。模型不可用时报出真实原因，这里不假装成功。"""
    raw = ""
    shown = ""
    try:
        async for chunk in chatter().astream([SystemMessage(_SYSTEM), HumanMessage(goal)]):
            raw += delta_of(chunk)
            # 只放行已经说完的句子：工具名过滤与两句上限都是整句算出来的
            ready = brief(until_sentence(raw))
            piece = _after(ready, shown)
            if piece:
                yield piece
                shown = ready
    except PlannerUnavailable:
        raise
    except Exception as exc:
        # 闲聊轮没有步骤可重试，得让用户看出来是模型挂了而不是指令没说清
        raise PlannerUnavailable("闲聊模型暂时不可用，请稍后再试") from exc

    # 末尾未断句的零头与收尾标点一起补上，拼接结果就是落库的 reply
    reply = compose(brief(raw))
    piece = _after(reply, shown)
    if piece:
        yield piece


def compose(body: str) -> str:
    """正文补上收尾标点。正文为空说明模型没话说，或只说了含工具名的话。"""
    if not body:
        return FALLBACK
    return body if ends_sentence(body) else f"{body}。"


def _after(full: str, emitted: str) -> str:
    """还没说过的那几个字。模型输出只会往尾部追加，所以不会算出重复内容。"""
    return full[len(emitted) :] if len(full) > len(emitted) else ""