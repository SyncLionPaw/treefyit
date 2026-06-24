# treefyust Delivery Report

本报告记录 `treefyust` 对当前 `treefyit/` 能力复刻的完成状态和验收结果。

## 总体状态

- 状态：已完成
- 范围：`REPLICATION_PLAN.md` 中 1-20 项
- 代码目录：`/Users/bytedance/docs/treefyit/treefyust`
- 不在本轮范围：真实数据库对接、性能优化、旧 `src/*` 兼容适配

## 任务完成状态

| # | 任务 | 状态 | 验收结果 |
|---|---|---|---|
| 1 | 建立 `config/model/builder/query/server/store/llm` 模块架构 | 完成 | `src/lib.rs` 导出完整模块，`main.rs` 启动 server |
| 2 | 初始化 Rust 基础依赖 | 完成 | `Cargo.toml` 和 `Cargo.lock` 已包含服务、序列化、配置、分词、HTTP、测试依赖 |
| 3 | 复刻 `Tree` / content model | 完成 | `src/model/tree.rs` 支持 `Tree`, `Node`, `NodeContent`, `LeafType` 序列化 |
| 4 | 复刻 `Forest` | 完成 | `src/model/forest.rs` 支持默认 forest、查找和 tree count |
| 5 | 复刻配置层 | 完成 | 支持 `settings.toml` 与 `settings.local.toml` 合并读取，缺失字段走默认值 |
| 6 | 实现 text build | 完成 | `build_tree_from_text` 可从 Markdown/text 构建 tree |
| 7 | 实现 parse 阶段 | 完成 | Markdown heading 被解析为 flat sections |
| 8 | 实现 refine 规则 | 完成 | 长 section 拆分后父节点保留，正文下沉到子节点，父节点保留 summary |
| 9 | 实现 tree finalize | 完成 | 节点具备 `node_id`, `depth`, `subtree_size`, `leaf_count` |
| 10 | 实现 file build | 完成 | 本地文件构建使用原始文件名作为 root title |
| 11 | builder 测试基线 | 完成 | 覆盖 Markdown build 和 long section refine |
| 12 | 复刻 query/index 数据结构 | 完成 | `TreeIndex`, `Posting`, `NodeDocumentStats`, `CorpusStats` 已实现 |
| 13 | 复刻分词与词项统计 | 完成 | 支持英文 token、中文 jieba 分词、fallback 搜索文本 |
| 14 | 实现 `build_tree_index` | 完成 | 生成 node 统计、postings、DF、tree 聚合统计 |
| 15 | 实现 BM25 与 forest query | 完成 | `score_nodes_bm25` 与 `InMemoryForestQuery` 已可用 |
| 16 | query/index 测试基线 | 完成 | 覆盖中文分词、BM25 排序、forest tree/node recall |
| 17 | 实现 JSON store | 完成 | Tree/Index 可落盘、恢复、删除；保存采用临时文件 + rename 原子替换 |
| 18 | 实现 HTTP server | 完成 | build、file build、legacy build、index、search、browse、forest、delete 接口可用 |
| 19 | store 接入 server 生命周期 | 完成 | 启动恢复 registry/default forest，build/index/delete/app history/session/query 同步 store |
| 20 | 文档与验收 | 完成 | README、OpenAPI、配置示例、交付报告已补齐 |

## 能力对照

| 能力 | Python `treefyit/` | Rust `treefyust/` | 状态 |
|---|---|---|---|
| Typed Tree/Forest model | 支持 | 支持 | 对齐 |
| Markdown/text build | 支持 | 支持 | 对齐 |
| File build | 支持 | 支持 | 对齐 |
| Long section refine | 父保留、正文下沉、父 summary | 父保留、正文下沉、父 summary | 对齐 |
| LLM bottom-up summarize | 支持 | 支持 | 对齐 |
| Tree index | 支持 | 支持 | 对齐 |
| BM25 node search | 支持 | 支持 | 对齐 |
| Forest search | 支持 | 支持 | 对齐 |
| REST API | 支持 | 支持 | 对齐 |
| Legacy `/api/build` | 支持 | 支持 JSON 与 multipart | 对齐 |
| Stream build / history / original file | 支持 | 支持 | 对齐 |
| Query log / query stats | 支持 | 支持 | 对齐 |
| Chat / sessions | 支持 pagent 兼容工具调用 | 支持 Ollama tools agent loop、稳定 tool id 与 session 记录 | 对齐 |
| JSON store | 支持 | 支持，且原子替换写入 | 对齐 |
| Startup restore | 支持 | 支持 | 对齐 |
| Ollama config | 支持 | 支持 | 对齐 |
| OpenAPI/README | 支持 | 支持 | 对齐 |

## 验收命令

当前验收使用以下命令：

```bash
cargo fmt -- --check
cargo test
cargo check
cargo clippy --all-targets -- -D warnings
ruby -e "require 'yaml'; YAML.load_file('openapi.yaml'); puts 'openapi ok'"
```

本地 Ollama 工具调用链路使用 ignored 集成测试单独验收：

```bash
cargo test ollama_chat_uses_tool_call -- --ignored --nocapture
```

## 剩余事项

以下事项不属于 1-20 复刻范围，保留为后续 backlog：

- 真实数据库对接
- 大规模索引性能优化
- 并发写入和多进程部署策略
- 与旧 `src/*` 服务做兼容适配
