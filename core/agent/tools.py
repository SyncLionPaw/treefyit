"""Tree tools —— host 进程内 CRUD / 持久化 / 单树检索。

供 ``Runner.create(tools=...)`` 注入；与 sandbox 文件工具配合：
读 ``source.md`` → 建树 → ``save_tree`` 落盘 → 他人 ``load_tree`` 继续编辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pagentv4 import FunctionTool, ToolOutput, tool

from ..build.md import markdown_to_tree
from ..model.ops import (
    create_node,
    get_node,
    get_parent,
    mount_node,
    move_node,
    remove_node,
    update_node,
    view_node,
    view_node_detail,
)
from ..build.query import format_hits, search_tree
from ..model.store import TreeStore
from ..model.tree import NodeKind, TreeNode

DEFAULT_EXTRA_SYSTEM = (
    "You build and maintain a document tree from markdown.\n"
    "Workflow:\n"
    "1) Read `/home/agent/source.md` with sandbox tools when present.\n"
    "2) Call `seed_from_markdown` to create a heading skeleton, then refine "
    "with create_child / update_fields / delete_node / relocate_node.\n"
    "3) Call `save_tree` so the tree is persisted to the library.\n"
    "4) Use `list_saved_trees`, `load_tree`, and `search_working_tree` "
    "for later inspection and reuse.\n"
    "Keep answers concise."
)


def parse_kind(kind: str | None) -> NodeKind | None:
    if kind is None or kind == "":
        return None
    try:
        return NodeKind(kind)
    except ValueError as exc:
        raise ValueError(
            f"invalid kind: {kind!r}; expected one of {[k.value for k in NodeKind]}"
        ) from exc


@dataclass
class TreeSession:
    """工作树 + 可选持久化库。"""

    root: TreeNode
    store: TreeStore | None = None
    tree_id: str | None = None
    source_md_path: Path | None = None

    def require_store(self) -> TreeStore:
        if self.store is None:
            raise ValueError("no tree store configured")
        return self.store

    def build_tools(self) -> list[FunctionTool]:
        session = self

        @tool(
            description=(
                "按 id 查看某个节点的大纲（标题、类型、子节点列表）。"
                "适合在浏览树结构、确认某节点下有哪些孩子时使用。"
                "只返回结构信息，不含正文；要读正文请用 view_detail。"
            )
        )
        def view_outline(node_id: str, depth: int = 1) -> ToolOutput:
            """查看节点大纲。

            Args:
                node_id: 目标节点 id。
                depth: 0 只看自身；1 含直接孩子。
            """
            try:
                return ToolOutput.succeed(view_node(session.root, node_id, depth=depth))
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))

        @tool(
            description=(
                "按 id 查看某个节点的详情，含正文（超长会截断）。"
                "适合确认节点具体内容、作为回答依据时使用。"
                "只想看结构不看正文时改用 view_outline。"
            )
        )
        def view_detail(node_id: str, max_content_chars: int = 2000) -> ToolOutput:
            """查看节点详情与正文。

            Args:
                node_id: 目标节点 id。
                max_content_chars: 正文最多返回多少字符，超出截断。
            """
            try:
                return ToolOutput.succeed(
                    view_node_detail(
                        session.root,
                        node_id,
                        max_content_chars=max_content_chars,
                    )
                )
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))

        @tool(
            description=(
                "在指定父节点下新建一个子节点并挂载。"
                "适合往树里补充章节、条目时使用。"
                "id 必须在整棵树内唯一，重复会失败。"
            )
        )
        def create_child(
            parent_id: str,
            id: str,
            title: str,
            kind: str | None = None,
            content: str | None = None,
            summary: str | None = None,
        ) -> ToolOutput:
            """在父节点下新建子节点。

            Args:
                parent_id: 父节点 id。
                id: 新节点 id，需在树内唯一。
                title: 新节点标题。
                kind: 可选类型：text、image、table、link。
                content: 可选正文。
                summary: 可选摘要。
            """
            try:
                child = create_node(
                    id,
                    title,
                    kind=parse_kind(kind),
                    content=content,
                    summary=summary,
                )
                new_root = mount_node(session.root, parent_id, child)
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.root = new_root
            return ToolOutput.succeed(view_node(session.root, parent_id, depth=1))

        @tool(
            description=(
                "按 id 修改节点字段。传入的字段才更新，未传的保持不变。"
                "适合改标题、补正文、加摘要时使用。"
                "要把某字段清空为 null，用对应的 clear_* 开关。"
            )
        )
        def update_fields(
            node_id: str,
            title: str | None = None,
            kind: str | None = None,
            content: str | None = None,
            summary: str | None = None,
            clear_kind: bool = False,
            clear_content: bool = False,
            clear_summary: bool = False,
        ) -> ToolOutput:
            """修改节点字段。

            Args:
                node_id: 目标节点 id。
                title: 新标题；省略则不变。
                kind: 新类型；省略则不变。
                content: 新正文；省略则不变。
                summary: 新摘要；省略则不变。
                clear_kind: 把 kind 置为 null。
                clear_content: 把 content 置为 null。
                clear_summary: 把 summary 置为 null。
            """
            try:
                kwargs: dict = {}
                if title is not None:
                    kwargs["title"] = title
                if clear_kind:
                    kwargs["kind"] = None
                elif kind is not None:
                    kwargs["kind"] = parse_kind(kind)
                if clear_content:
                    kwargs["content"] = None
                elif content is not None:
                    kwargs["content"] = content
                if clear_summary:
                    kwargs["summary"] = None
                elif summary is not None:
                    kwargs["summary"] = summary

                get_node(session.root, node_id)
                new_root = update_node(session.root, node_id, **kwargs)
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.root = new_root
            return ToolOutput.succeed(
                view_node_detail(session.root, node_id, max_content_chars=500)
            )

        @tool(
            description=(
                "按 id 删除节点及其整棵子树。"
                "适合删除多余或错误的分支时使用。"
                "无法删除根节点；删除会连带丢失所有后代，请谨慎。"
            )
        )
        def delete_node(node_id: str) -> ToolOutput:
            """删除节点及其子树。

            Args:
                node_id: 要删除的节点 id。
            """
            try:
                parent = get_parent(session.root, node_id)
                parent_hint = parent.id if parent is not None else None
                new_root = remove_node(session.root, node_id)
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.root = new_root
            if parent_hint is None:
                return ToolOutput.succeed(f"deleted {node_id}")
            return ToolOutput.succeed(
                f"deleted {node_id}\n" + view_node(session.root, parent_hint, depth=1)
            )

        @tool(
            description=(
                "把一个节点（连同子树）移动到新的父节点下。"
                "适合调整章节归属、重排结构时使用。"
                "可用 index 指定移动后在新父节点中的位置，默认追加到末尾。"
            )
        )
        def relocate_node(
            node_id: str,
            new_parent_id: str,
            index: int | None = None,
        ) -> ToolOutput:
            """移动节点到新父节点下。

            Args:
                node_id: 要移动的节点 id。
                new_parent_id: 目标父节点 id。
                index: 移动后在子节点中的位置；默认追加到末尾。
            """
            try:
                new_root = move_node(
                    session.root,
                    node_id,
                    new_parent_id,
                    index=index,
                )
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.root = new_root
            return ToolOutput.succeed(view_node(session.root, new_parent_id, depth=1))

        @tool(
            description=(
                "读取 markdown 文件，按标题层级生成骨架树，写入当前工作树。"
                "适合建树起步：先 seed 出骨架，再用 create/update 等工具细化。"
                "会覆盖当前工作树；不传 path 时用会话默认的源 markdown。"
            )
        )
        def seed_from_markdown(path: str | None = None) -> ToolOutput:
            """从 markdown 生成骨架树。

            Args:
                path: host 上的 markdown 路径；默认用会话的源 markdown。
            """
            try:
                md_path = Path(path) if path else session.source_md_path
                if md_path is None:
                    raise ValueError("markdown path not provided")
                md_path = md_path.expanduser().resolve()
                if not md_path.is_file():
                    raise FileNotFoundError(f"markdown not found: {md_path}")
                text = md_path.read_text(encoding="utf-8")
                seeded = markdown_to_tree(
                    text,
                    root_id=session.root.id,
                    root_title=session.root.title or md_path.stem,
                )
            except (OSError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.root = seeded
            session.source_md_path = md_path
            return ToolOutput.succeed(
                "seeded from markdown\n"
                + view_node(session.root, session.root.id, depth=2)
            )

        @tool(
            description=(
                "把当前工作树持久化到库中，供之后 load / search 复用。"
                "细化完树后调用它落盘，否则改动只在内存里。"
                "不传 tree_id 时沿用会话当前 id，没有则新建一个。"
            )
        )
        def save_tree(
            tree_id: str | None = None, title: str | None = None
        ) -> ToolOutput:
            """持久化工作树到库。

            Args:
                tree_id: 可选 id；默认沿用会话 tree_id 或新建。
                title: 可选库标题；默认用根节点标题。
            """
            try:
                store = session.require_store()
                record = store.save(
                    session.root,
                    tree_id=tree_id or session.tree_id,
                    title=title,
                    source_path=session.source_md_path,
                )
            except (KeyError, ValueError, OSError) as exc:
                return ToolOutput.fail(str(exc))
            session.tree_id = record.tree_id
            return ToolOutput.succeed(
                "saved "
                f"tree_id={record.tree_id} title={record.title!r} "
                f"nodes={record.node_count} path={store.tree_path(record.tree_id)}"
            )

        @tool(
            description=(
                "从库中按 tree_id 加载一棵已保存的树到当前工作会话。"
                "适合继续编辑或查看之前保存的树时使用。"
                "会覆盖当前工作树；先用 list_saved_trees 查可用 id。"
            )
        )
        def load_tree(tree_id: str) -> ToolOutput:
            """从库加载树到工作会话。

            Args:
                tree_id: 库中的树 id。
            """
            try:
                store = session.require_store()
                record = store.load(tree_id)
            except (KeyError, ValueError, OSError) as exc:
                return ToolOutput.fail(str(exc))
            session.root = record.root
            session.tree_id = record.tree_id
            if record.source_path:
                session.source_md_path = Path(record.source_path)
            return ToolOutput.succeed(
                f"loaded tree_id={record.tree_id}\n"
                + view_node(session.root, session.root.id, depth=2)
            )

        @tool(
            description=(
                "列出库中已保存的所有树（id、标题、节点数、更新时间）。"
                "适合在 load 或 search 之前先看看库里有哪些树。"
                "库为空时返回提示。"
            )
        )
        def list_saved_trees() -> ToolOutput:
            try:
                store = session.require_store()
                items = store.list()
            except (ValueError, OSError) as exc:
                return ToolOutput.fail(str(exc))
            if not items:
                return ToolOutput.succeed("library empty")
            lines = [f"library={len(items)}"]
            for item in items:
                lines.append(
                    f"- {item['tree_id']}: {item['title']} "
                    f"(nodes={item['node_count']}, updated={item['updated_at']})"
                )
            return ToolOutput.succeed("\n".join(lines))

        @tool(
            description=(
                "在当前内存中的工作树里按关键字搜索节点。"
                "适合在正在编辑的这棵树里定位相关章节时使用。"
                "只搜当前工作树；要跨所有已保存的树请用 search_library。"
            )
        )
        def search_working_tree(query: str, limit: int = 8) -> ToolOutput:
            """检索当前工作树。

            Args:
                query: 匹配标题/摘要/正文的关键词。
                limit: 最多返回几条。
            """
            hits = search_tree(
                session.root,
                query,
                tree_id=session.tree_id or "memory",
                limit=limit,
            )
            return ToolOutput.succeed(format_hits(hits))

        return [
            view_outline,
            view_detail,
            create_child,
            update_fields,
            delete_node,
            relocate_node,
            seed_from_markdown,
            save_tree,
            load_tree,
            list_saved_trees,
            search_working_tree,
        ]


def build_tree_tools(
    root: TreeNode | None = None,
    *,
    store: TreeStore | None = None,
    tree_id: str | None = None,
    source_md_path: str | Path | None = None,
) -> tuple[TreeSession, list[FunctionTool]]:
    """便捷工厂：返回 session 与 pagentv4 tool 列表。

    不传 root 时创建一个空根，agent 之后用 create_child / seed_from_markdown
    往里填内容。
    """
    session = TreeSession(
        root=root if root is not None else create_node("root", "untitled"),
        store=store,
        tree_id=tree_id,
        source_md_path=Path(source_md_path).expanduser().resolve()
        if source_md_path is not None
        else None,
    )
    return session, session.build_tools()
