from functools import lru_cache

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.tools import SPECS, ToolSpec

# 图像模型必须走 DashScope 原生接口，纯文本的规划模型可用 OpenAI 兼容模式
_COMPATIBLE_PATH = "/compatible-mode/v1"


class PlannerUnavailable(Exception):
    """规划模型未配置或不可用。"""


def _schema_of(spec: ToolSpec) -> dict:
    function = convert_to_openai_function(spec.params)
    function["name"] = spec.name
    function["description"] = spec.description

    parameters = function.get("parameters", {})
    for hidden in spec.agent_hidden:
        parameters.get("properties", {}).pop(hidden, None)
    parameters["required"] = [
        name for name in parameters.get("required", []) if name not in spec.agent_hidden
    ]
    return {"type": "function", "function": function}


def _client(
    model: str,
    *,
    temperature: float,
    max_tokens: int | None = None,
    thinking: bool | None = None,
) -> ChatOpenAI:
    """三个通道共用一套接入方式。缺 key 一律抛 PlannerUnavailable，由服务层如实标记失败。"""
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise PlannerUnavailable("未配置 DASHSCOPE_API_KEY，对话指令不可用")

    kwargs: dict = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if thinking is not None:
        # enable_thinking 不是 OpenAI 参数，只能经 extra_body 透传给 DashScope；
        # 开着思考又不用流式接口会直接 400，所以短输出通道显式关掉
        kwargs["extra_body"] = {"enable_thinking": thinking}
    return ChatOpenAI(
        model=model,
        api_key=settings.dashscope_api_key,
        base_url=f"{settings.dashscope_base_url}{_COMPATIBLE_PATH}",
        **kwargs,
    )


@lru_cache
def planner():
    """绑定全部已注册工具的规划模型。工具增减无需改动此处。"""
    model = _client(get_settings().planner_model, temperature=0)
    return model.bind_tools([_schema_of(spec) for spec in SPECS])


@lru_cache
def triager():
    """分流模型：不绑工具、不开思考，只输出一个词，只要便宜快。"""
    return _client(get_settings().intent_model, temperature=0, max_tokens=16, thinking=False)


@lru_cache
def chatter():
    """闲聊模型：不绑工具、看不到画布，max_tokens 掐住回答长度。"""
    return _client(get_settings().chat_model, temperature=0.3, max_tokens=200)
