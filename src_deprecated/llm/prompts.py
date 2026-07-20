"""Stable prompt templates for LLM features."""

from __future__ import annotations


SUMMARY_SYSTEM_PROMPT = """你是一个摘要专家。
请根据给定内容生成准确、简洁、忠实原文的中文摘要。
不要编造信息，不要输出项目符号，不要输出解释。"""


SECTION_REFINER_SYSTEM_PROMPT = """你是一个文档 section 精修器。
请根据给定的 section 标题和正文，输出固定分隔符格式的结构化结果。
你可以保留原 section，也可以把一个过长 section 细化成多个子 section。
不要编造事实，不要输出分隔符格式之外的任何内容。"""


SECTION_REFINER_USER_PROMPT_TEMPLATE = """请精修当前 section，并输出固定分隔符格式。

输出要求：
1. 只输出一个或多个 [SECTION] ... [/SECTION] 块
2. 每个 [SECTION] 块都必须包含这些头字段，每行一个：
   title: <string>
   level_delta: <0或1>
   leaf_type: <text|image|table|link，可留空>
   content_kind: <text|url|resource，可留空>
   url: <string，可留空>
   uri: <string，可留空>
   media_type: <string，可留空>
3. 每个 [SECTION] 块都必须包含一个 [TEXT] ... [/TEXT] 块
4. 如果有摘要，使用 [SUMMARY] ... [/SUMMARY] 块；没有可省略
5. 如果不需要拆分，只返回一个 level_delta=0 的 [SECTION]
6. 如果需要拆分，第一项保留当前 section，level_delta=0；新增子 section 使用 level_delta=1
7. 最多返回 {max_parts} 个 [SECTION]
8. 不要输出 markdown 代码块，不要输出 JSON

输出示例：
[SECTION]
title: 示例标题
level_delta: 0
leaf_type:
content_kind: text
url:
uri:
media_type:
[TEXT]
这里是正文。
[/TEXT]
[SUMMARY]
这里是摘要。
[/SUMMARY]
[/SECTION]

输入信息：
source_kind: {source_kind}
title: {title}
text:
{content}
"""


SUMMARY_USER_PROMPT_TEMPLATE = """请为当前节点生成摘要。

要求：
1. 保留主题、结论和关键信息
2. 输出 2-4 句
3. 如果正文较少，可以参考子节点摘要
4. 不要重复标题原文
5. 不要使用“本节主要讲了”这类空话

当前节点标题：
{title}

当前节点正文：
{content}

子节点摘要：
{children_summaries}
"""


def format_child_summaries(child_summaries: list[str] | None = None) -> str:
    if not child_summaries:
        return "无"

    lines = [summary.strip() for summary in child_summaries if summary.strip()]
    if not lines:
        return "无"
    return "\n".join(f"- {line}" for line in lines)


def build_summary_prompt(
    *,
    title: str,
    content: str = "",
    child_summaries: list[str] | None = None,
) -> str:
    return SUMMARY_USER_PROMPT_TEMPLATE.format(
        title=title.strip() or "Untitled",
        content=content.strip() or "无",
        children_summaries=format_child_summaries(child_summaries),
    )


def build_section_refine_prompt(
    *,
    title: str,
    content: str,
    source_kind: str | None = None,
    max_parts: int = 4,
) -> str:
    return SECTION_REFINER_USER_PROMPT_TEMPLATE.format(
        source_kind=(source_kind or "unknown").strip() or "unknown",
        title=title.strip() or "Untitled",
        content=content.strip() or "无",
        max_parts=max_parts,
    )


__all__ = [
    "SECTION_REFINER_SYSTEM_PROMPT",
    "SECTION_REFINER_USER_PROMPT_TEMPLATE",
    "SUMMARY_SYSTEM_PROMPT",
    "SUMMARY_USER_PROMPT_TEMPLATE",
    "build_section_refine_prompt",
    "build_summary_prompt",
    "format_child_summaries",
]
