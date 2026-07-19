# treefyit.builder 数据流

`treefyit.builder` 的目标是把多类型文档输入，转换成稳定的 `treefyit.model.tree.Tree`。

当前链路是：

```
source -> parse -> infer -> refine -> build -> finalize -> convert
```

## 1. source

文件：[`source.py`](file:///Users/bytedance/docs/treefyit/treefyit/builder/source.py)

职责：

- 识别输入是 `pdf` / `html` / `zip` / `text`
- 使用 `libmagic` 优先，suffix 作为 fallback

输入：

- `Path`

输出：

```python
Literal["pdf", "html", "zip", "text"]
```

## 2. parse

文件：[`parse.py`](file:///Users/bytedance/docs/treefyit/treefyit/builder/parse.py)

职责：

- 把不同类型输入归一化成 flat sections
- Markdown 用 `markdown-it-py`
- HTML / PDF 先转成 Markdown，再提取 sections
- 在没有 headings 时退化成单 section

输入：

- `text`
- `Path`

输出：

```python
list[section]
```

section 的基础形态：

```python
{
    "title": str,
    "level": int,
    "line_num": int,
    "text": str,
}
```

此时的 `level` 只是 parser 原始结果，不一定可靠。

## 3. infer

文件：[`infer.py`](file:///Users/bytedance/docs/treefyit/treefyit/builder/infer.py)

职责：

- 修正 flat sections 的层级
- 根据编号、中文标题前缀等规则推断真正深度
- 设计成可插拔策略

输入：

```python
list[section]
```

输出：

```python
list[section]
```

变化点：

- 保持 flat 结构不变
- 主要修正 `level`

例如：

```python
{"title": "2 Related Work", "level": 1, ...}
{"title": "2.1 Rule-based Methods", "level": 2, ...}
```

## 4. refine

文件：[`refine.py`](file:///Users/bytedance/docs/treefyit/treefyit/builder/refine.py)

职责：

- 拆分过长 section
- 把粗粒度 section 细化成多个 section
- 必要时补出子 section 标题
- 清理明显不合理的内容块
- 检测特殊类型，例如表格、链接、图片

输入：

```python
list[section]
```

输出：

```python
list[refined_section]
```

refined section 会在基础字段上增加结构化标记：

```python
{
    "title": str,
    "level": int,
    "line_num": int,
    "text": str,
    "leaf_type": "text" | "table" | "link" | "image" | None,
    "content_kind": "text" | "url" | "resource" | None,
    "url": str,   # 可选
    "uri": str,   # 可选
}
```

## 5. build

文件：[`parse.py`](file:///Users/bytedance/docs/treefyit/treefyit/builder/parse.py)

职责：

- 根据修正后的 `level` 用 stack 组装 nested tree

输入：

```python
list[refined_section]
```

输出：

```python
list[legacy_node]
```

legacy node 形态：

```python
{
    "title": str,
    "text": str,
    "children": list[legacy_node],
    "leaf_type": str | None,
    "content_kind": str | None,
    "url": str | None,
    "uri": str | None,
}
```

## 6. finalize

文件：[`build.py`](file:///Users/bytedance/docs/treefyit/treefyit/builder/build.py)

职责：

- 给 nested tree 分配 `node_id`
- 可选生成摘要

输入：

```python
list[legacy_node]
```

输出：

```python
list[legacy_node]
```

变化点：

- 增加 `node_id`
- 可选增加 `summary`

## 7. convert

文件：[`convert.py`](file:///Users/bytedance/docs/treefyit/treefyit/builder/convert.py)

职责：

- 把 nested dict tree 转成 `treefyit.model.tree.Tree`
- 填充 `depth` / `subtree_size` / `leaf_count`
- 根据 `content_kind` / `leaf_type` 转成正式 content model

输入：

```python
list[legacy_node]
```

输出：

```python
Tree
```

对应关系：

- `content_kind == "text"` -> `TextContent`
- `content_kind == "url"` -> `UrlContent`
- `content_kind == "resource"` -> `ResourceContent`
- `leaf_type` -> `LeafType`

## 总结

这条链路上最重要的边界有两个：

1. `infer` 和 `refine` 都只处理 flat sections，不直接输出树
2. `convert` 才是进入正式 `Tree` 模型的唯一入口

因此：

```text
parse / infer / refine
```

处理的是中间 section 结构，

```text
build / finalize / convert
```

处理的是树结构和最终模型。
