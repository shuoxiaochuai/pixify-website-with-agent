"""闲聊通道：回答与修图无关的问题，只答两句，末尾补一句身份说明。

这里不绑工具、不给画布摘要，所以既不会生成计划，也不会污染修图上下文。
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import PlannerUnavailable, chatter
from app.agent.words import brief, ends_sentence, text_of


# 模型答不上来（或只说了句含工具名的话）时的保底，绝不回落到「没太理解这条指令」
FALLBACK = "这个我不太确定。"

_SYSTEM = """你在一个 AI 修图产品里陪用户聊天，用户这句话和修图无关。

规则：
- 用自己的知识直接回答，最多两句话。不分点、不举例、不展开背景介绍、不给步骤。
- 不确定就直说不确定，禁止编造。
- 不要复述用户的问题，不要用「好的」「当然可以」这类客套开头。
- 不要提任何工具名、参数或系统内部实现，也不要自称通用大模型。
- 不要声明自己更擅长修图，也不要引导用户去提修图需求，答完就停。"""


async def answer(goal: str) -> str:
    """简短回答 + 固定身份提示。模型不可用时报出真实原因，这里不假装成功。"""
    try:
        message = await chatter().ainvoke([SystemMessage(_SYSTEM), HumanMessage(goal)])
    except PlannerUnavailable:
        raise
    except Exception as exc:
        # 闲聊轮没有步骤可重试，得让用户看出来是模型挂了而不是指令没说清
        raise PlannerUnavailable("闲聊模型暂时不可用，请稍后再试") from exc
    body = brief(text_of(message))
    if not body:
        return FALLBACK
    return body if ends_sentence(body) else f"{body}。"