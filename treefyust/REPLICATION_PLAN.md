# treefyust 20-Step Replication Plan

目标：在 [treefyust](file:///Users/bytedance/docs/treefyit/treefyust) 中，用 Rust 把 [treefyit](file:///Users/bytedance/docs/treefyit/treefyit) 当前已经实现的能力完整复刻一遍。

范围以当前 `treefyit/` 为准，包括：

- typed model: `Tree` / `Forest`
- builder: text / file build
- refine: 父节点保留、正文下沉、父节点只保留摘要
- query/index: tree index、BM25、forest query
- server: health、build、index、search、browse、forest、delete
- store: JSON 文件持久化与恢复
- config: TOML 配置
- docs: OpenAPI + README

不在本轮范围内：

- 真实数据库对接
- 性能优化 / 并发优化
- 与 `src/*` 旧实现做兼容适配

## 20 steps

- [x] 1. 建立 `treefyust` 目标架构：把 `src/main.rs` 拆成 `config/ model/ builder/ query/ server/ store/ llm/` 模块，并确定 crate 边界。
- [x] 2. 选定 Rust 基础依赖：`serde`, `serde_json`, `toml`, `axum` 或 `actix-web`, `tokio`, `uuid`, `regex`, `anyhow`/`thiserror`，并把 `Cargo.toml` 初始化完整。
- [x] 3. 复刻 `treefyit/model/tree.py`：定义 `Tree`, `Node`, `TextContent`, `UrlContent`, `ResourceContent`, `LeafType`，保证 JSON 序列化结构与 Python 版对齐。
- [x] 4. 复刻 `treefyit/model/forest.py`：定义 `Forest`，补齐 `tree_count`, `add_tree`, `get_tree` 等基础行为。
- [x] 5. 复刻配置层：实现 `settings.toml + settings.local.toml` 合并读取，对齐 `llm`, `builder`, `store` 三类配置。
- [x] 6. 先实现 text build 最小链路：支持从纯文本/Markdown 输入构建 `Tree`，作为后续所有功能的基础。
- [x] 7. 复刻 parse 阶段：把 Markdown / text 解析成 flat sections，保证标题、正文、层级切分规则与 Python 版一致。
- [x] 8. 复刻 refine 规则：长 section 拆分后，父节点保留为结构节点，正文下沉到子节点，父节点最多只保留摘要。
- [x] 9. 复刻 tree convert/finalize：为节点分配稳定 `node_id`，补齐 `depth`, `subtree_size`, `leaf_count` 等派生字段。
- [x] 10. 复刻 file build：支持从本地文件构建 tree，并保证 root title 使用原始文件名，而不是临时文件名。
- [x] 11. 为 builder 建测试基线：用当前 `treefyit/tests/test_builder.py` 的语义做 Rust 测试镜像，先覆盖 text build、refine、file build。
- [x] 12. 复刻 query 基础结构：实现 `TreeQueryHit`, `NodeQueryHit`, `Posting`, `NodeDocumentStats`, `CorpusStats`, `TreeIndex`。
- [x] 13. 复刻分词与词项统计：先实现当前同等级规则，包含英文 token、中文分词入口、fallback 行为，以及 `content_to_search_text`。
- [x] 14. 复刻 `build_tree_index(tree)`：完成 node 级词项统计、postings、document_frequency、tree 聚合统计。
- [x] 15. 复刻 `score_nodes_bm25(index, query)` 与 `InMemoryForestQuery`：先保证功能正确，再考虑性能和缓存。
- [x] 16. 为 query/index 建测试基线：镜像当前 `tests/test_query.py` 的核心场景，包括中文分词、tree recall、node ranking、forest query。
- [x] 17. 复刻 store：实现 `RegistryStore`，支持 `Tree` / `TreeIndex` 的 JSON 落盘、恢复、删除联动。
- [x] 18. 复刻 server：实现 `/health`、`/api/trees`、`/api/trees/from-file`、`/api/build`、`/api/trees/{tree_id}/index`、`/api/trees/{tree_id}/search/nodes`、browse、forest、delete 全套接口。
- [x] 19. 把 store 接进 server 生命周期：build/index/delete 同步落盘，服务启动自动恢复 `tree_registry` / `index_registry` / 默认 forest。
- [x] 20. 收尾文档与验收：生成 `openapi.yaml`、补 README、新增最小请求/响应样例，并建立“Rust 版对 Python 版”的能力对照清单。

## 模块映射

Python -> Rust 建议映射：

- `treefyit/config/*` -> `treefyust/src/config/*`
- `treefyit/model/*` -> `treefyust/src/model/*`
- `treefyit/builder/*` -> `treefyust/src/builder/*`
- `treefyit/query/*` -> `treefyust/src/query/*`
- `treefyit/store/*` -> `treefyust/src/store/*`
- `treefyit/server/*` -> `treefyust/src/server/*`
- `treefyit/llm/*` -> `treefyust/src/llm/*`

## 验收标准

- Rust 版接口集与 Python 版当前 `treefyit.server` 对齐
- Rust 版 `Tree` / `TreeIndex` JSON 结构可与 Python 版互相校验
- build / index / search / forest / delete / restore 六条主链路全部跑通
- 至少具备一套与 Python 当前测试语义对应的 Rust 测试
