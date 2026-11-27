"""模型输出文本的清洗与防泄漏：只把能给用户看的那句吐出去。"""

import re

from langchain_core.messages import AIMessage, AIMessageChunk

from app.tools import SPECS

REFUSAL_LIMIT = 60
# 闲聊的宽度上限：两句、120 字，防止它叭叭叭说一大堆
CHAT_SENTENCES = 2
CHAT_LIMIT = 120

# 按句切分，保留标点，便于只丢掉某几句
_SENTENCE = re.compile(r"(?<=[。！？])|[\r\n]+")
_TERMINATORS = "。！？…"

# 整条回答与流式增量走同一套清洗规则
Message = AIMessage | AIMessageChunk


def text_of(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    return "".join(
        block.get("text", "") for block in message.content if isinstance(block, dict)
    ).strip()


def delta_of(message: Message) -> str:
    """取一段增量文本。不能复用 text_of，它会 strip 掉中英混排的词间空格。"""
    if isinstance(message.content, str):
        return message.content
    return "".join(block.get("text", "") for block in message.content if isinstance(block, dict))


def until_sentence(text: str) -> str:
    """只保留到最后一个句号为止的部分，末尾那句还没说完的先按住。

    流式输出靠它判断哪些字已经能安全给用户看：工具名过滤与长度限制都是整句算出来的。
    """
    index = max((text.rfind(sep) for sep in "。！？"), default=-1)
    return text[: index + 1] if index >= 0 else ""


def mentions_tool(text: str) -> bool:
    """文字里出现内部工具名，说明模型在描述调用而不是在回答用户。"""
    return any(spec.name in text for spec in SPECS)


# 闲聊走的是纯文本渲染，模型一上公式就吐 $E=mc^2$ 这种 LaTeX，得换成能直接读的样子
_MATH_SYMBOLS = {
    r"\times": "\u00d7",
    r"\div": "\u00f7",
    r"\pm": "\u00b1",
    r"\mp": "\u2213",
    r"\cdot": "\u00b7",
    r"\ast": "*",
    r"\star": "\u22c6",
    r"\leq": "\u2264",
    r"\le": "\u2264",
    r"\geq": "\u2265",
    r"\ge": "\u2265",
    r"\neq": "\u2260",
    r"\ne": "\u2260",
    r"\approx": "\u2248",
    r"\equiv": "\u2261",
    r"\sim": "~",
    r"\propto": "\u221d",
    r"\infty": "\u221e",
    r"\partial": "\u2202",
    r"\nabla": "\u2207",
    r"\forall": "\u2200",
    r"\exists": "\u2203",
    r"\in": "\u2208",
    r"\notin": "\u2209",
    r"\subset": "\u2286",
    r"\cup": "\u222a",
    r"\cap": "\u2229",
    r"\sum": "\u2211",
    r"\prod": "\u220f",
    r"\int": "\u222b",
    r"\rightarrow": "\u2192",
    r"\longrightarrow": "\u2192",
    r"\to": "\u2192",
    r"\leftarrow": "\u2190",
    r"\Rightarrow": "\u21d2",
    r"\Leftrightarrow": "\u21d4",
    r"\perp": "\u22a5",
    r"\parallel": "\u2225",
    r"\angle": "\u2220",
    r"\degree": "\u00b0",
    r"\circ": "\u00b0",
    r"\pi": "\u03c0",
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\Delta": "\u0394",
    r"\delta": "\u03b4",
    r"\theta": "\u03b8",
    r"\lambda": "\u03bb",
    r"\mu": "\u03bc",
    r"\nu": "\u03bd",
    r"\rho": "\u03c1",
    r"\sigma": "\u03c3",
    r"\tau": "\u03c4",
    r"\varphi": "\u03c6",
    r"\Phi": "\u03a6",
    r"\omega": "\u03c9",
    r"\Omega": "\u03a9",
}

_SUPERSCRIPT = {
    "0": "\u2070",
    "1": "\u00b9",
    "2": "\u00b2",
    "3": "\u00b3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
    "+": "\u207a",
    "-": "\u207b",
    "=": "\u207c",
    "(": "\u207d",
    ")": "\u207e",
    "n": "\u207f",
    "i": "\u2071",
    "x": "\u02e3",
}

_SUBSCRIPT = {
    **dict(zip("0123456789", "₀₁₂₃₄₅₆₇₈₉", strict=True)),
    "+": "\u208a",
    "-": "\u208b",
    "=": "\u208c",
    "(": "\u208d",
    ")": "\u208e",
    "a": "\u2090",
    "e": "\u2091",
    "i": "\u1d62",
    "j": "\u2c7c",
    "n": "\u2099",
    "o": "\u2092",
    "r": "\u1d63",
    "s": "\u209b",
    "t": "\u209c",
    "u": "\u1d64",
    "v": "\u2099",
    "x": "\u2093",
}

_SYMBOL = re.compile(
    "|".join(re.escape(key) for key in sorted(_MATH_SYMBOLS, key=len, reverse=True))
)
_WRAPPED = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}")
_FRAC = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt(?:\[([^\]]*)\])?\{([^{}]*)\}")
_FENCE = re.compile(r"\\(?:left|right|big[lr]?|Big[lr]?|bigg[lr]?)\s*")
# 矩阵这类环境展不开，去掉外壳按「1 , 0；0 , 1」读，总好过留一堆 begin/end
_ENV = re.compile(r"\\(?:begin|end)\{[^{}]*\}")
_ROW = re.compile(r"\\\\\s*")
_COL = re.compile(r"\s*&\s*")
_ESCAPED = re.compile(r"\\([{}%&_#~^])")
_SUP_GROUP = re.compile(r"\^\{([^{}]*)\}")
_SUB_GROUP = re.compile(r"_\{([^{}]*)\}")
_SUP_ONE = re.compile(r"\^([-+\w])")
_SUB_ONE = re.compile(r"_([-+\w])")
_COMMAND = re.compile(r"\\([a-zA-Z]+)")
_NOISE = re.compile(r"\\[\s,;:!*]|([{}])")
_DELIMITER = re.compile(r"\$\$?|\\[\[\]()~]|`+")
# 正文里被转义的 \$ 是美元号不是公式开头，先把它们藏起来
_CURRENCY = "\ue000"
# 真出现 LaTeX 痕迹才动手：命令、成对的 $、或 \( \[ 定界符。单一个 $ 当货币符号的不算公式
_LATEX = re.compile(r"\\[a-zA-Z]|\$\$|\$[^$\n]{1,80}\$|\\(?:\[|\()")


def _script(body: str, table: dict[str, str], marker: str) -> str:
    """能换成上下标符号就换，换不动的写成 x^(ab)，不硬拼成误导人的 xab。"""
    if body and all(char in table for char in body):
        return "".join(table[char] for char in body)
    return f"{marker}{body}" if len(body) == 1 else f"{marker}({body})"


def _tight(body: str) -> bool:
    """加减号在括号外面，说明这是一整块，直接写 a+b/c 会被读成 a+(b/c)。"""
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif depth == 0 and char in "+-\u2212":
            return True
    return False


def _needs_group(body: str) -> bool:
    # 还带着没拆开的命令或除法，一样不能直接接在外面
    return _tight(body) or "\\" in body or "/" in body


def _root(match: re.Match) -> str:
    degree, body = match.group(1), match.group(2)
    if degree:
        return f"{degree}\u6b21\u6839\u53f7({body})"
    return f"\u221a({body})" if _needs_group(body) else f"\u221a{body}"


def _slash(match: re.Match) -> str:
    """分数线写成 /，两端是整块时补上括号，保住原来的层级。"""
    numerator, denominator = match.group(1), match.group(2)
    top = f"({numerator})" if _tight(numerator) else numerator
    bottom = f"({denominator})" if _needs_group(denominator) else denominator
    return f"{top}/{bottom}"


def _unnest(text: str) -> str:
    r"""一层层拆开套着的分数与根号。

    单次替换只认花括号里不再套花括号的那层，所以 \frac{√a}{2} 这种得跑几遍才能化干净。
    """
    for _ in range(4):
        replaced = _FRAC.sub(_slash, _SQRT.sub(_root, text))
        if replaced == text:
            return replaced
        text = replaced
    return text


def mathify(text: str) -> str:
    """把 LaTeX 公式写成常见数学形式：$E=mc^2$ \u2192 E=mc\u00b2，\\frac{1}{2} \u2192 1/2\u3002

    只动公式写法，不改句子内容；没出现 LaTeX 痕迹直接原样返回，免得把「价格 $100」这种正文拆坏。
    """
    text = text.replace(r"\$", _CURRENCY)
    if not _LATEX.search(text):
        return text.replace(_CURRENCY, "$")

    text = _ENV.sub("", text)
    text = _ROW.sub("\uff1b", text)
    text = _COL.sub(",", text)
    text = _unnest(text)
    text = _WRAPPED.sub(r"\1", text)
    text = _FENCE.sub("", text)
    text = _SYMBOL.sub(lambda m: _MATH_SYMBOLS[m.group(0)], text)
    text = _SUP_GROUP.sub(lambda m: _script(m.group(1), _SUPERSCRIPT, "^"), text)
    text = _SUB_GROUP.sub(lambda m: _script(m.group(1), _SUBSCRIPT, "_"), text)
    text = _SUP_ONE.sub(lambda m: _script(m.group(1), _SUPERSCRIPT, "^"), text)
    text = _SUB_ONE.sub(lambda m: _script(m.group(1), _SUBSCRIPT, "_"), text)
    # \sin\theta 这类函数名本来就不用反斜杠，脱掉就读得通了
    text = _COMMAND.sub(r"\1", text)
    text = _ESCAPED.sub(r"\1", text)
    text = _NOISE.sub(lambda m: m.group(1) or "", text)
    text = _DELIMITER.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text).replace(_CURRENCY, "$")
    return text.strip()


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
    """闲聊的宽度限制：丢掉提工具名的句子，公式换成普通写法，最多两句、120 字。"""
    kept = [mathify(sentence) for sentence in sentences(reply) if not mentions_tool(sentence)]
    text = "".join(kept[:CHAT_SENTENCES])
    if len(text) > CHAT_LIMIT:
        text = text[:CHAT_LIMIT].rstrip("，,、；;。") + "…"
    return text.strip()


def ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in _TERMINATORS