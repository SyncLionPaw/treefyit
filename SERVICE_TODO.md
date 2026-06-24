# Service Todo

基于 `treefyit/` 新实现，逐步落一套独立对外服务。

原则：

- 只依赖 `treefyit/`，不复用 `src/` 旧服务实现
- 先打通单树链路，再扩到 forest
- 先内存态可用，再考虑持久化
- API 返回保持轻量，避免把整棵树和全部索引一次性塞回响应
- 对外主接口采用 RESTful 命名，同时兼容旧路径

## 当前基础

- [x] 文档 -> Tree 构建链路
- [x] refine 后父节点保留为结构节点，正文下沉到子节点
- [x] Tree 级索引构建
- [x] 基于索引的 BM25 节点打分
- [x] query 层去掉多余结果包装层

## Phase 1: 最小可用服务

- [x] 新建 `treefyit/server/` 服务入口
- [x] 初始化 FastAPI app
- [x] 定义基础 health 接口：`GET /health`
- [x] 定义统一响应错误格式
- [x] 增加基础服务测试

## Phase 2: 单树构建接口

- [x] `POST /api/trees`
- [x] 兼容旧路径：`POST /api/build`
- [x] `POST /api/trees/from-file`
- [x] 请求参数模型：文件名、summarize、refine 配置
- [x] 返回 Tree 元数据，不直接返回全部重内容字段
- [x] 为 build 结果分配 `tree_id`
- [x] 内存注册表：保存 `tree_id -> Tree`
- [x] build 接口测试

## Phase 3: 单树索引接口

- [x] `POST /api/trees/{tree_id}/index`
- [x] 兼容别名：`POST /api/tree/{tree_id}/index`
- [x] 内存注册表：保存 `tree_id -> TreeIndex`
- [x] 支持重复构建索引时覆盖旧索引
- [x] `GET /api/trees/{tree_id}/index/meta`
- [x] 兼容别名：`GET /api/tree/{tree_id}/index/meta`
- [x] 返回索引摘要：文档数、平均长度、term 数量
- [x] index 接口测试

## Phase 4: 单树查询接口

- [x] `GET /api/trees/{tree_id}/search/nodes`
- [x] 兼容别名：`GET /api/tree/{tree_id}/search/nodes`
- [x] 基于 `score_nodes_bm25()` 返回命中节点
- [x] 支持 `q`、`limit`
- [x] 索引不存在时返回明确错误
- [x] 查询接口测试

## Phase 5: 单树浏览接口

- [x] `GET /api/trees/{tree_id}`
- [x] 兼容别名：`GET /api/tree/{tree_id}`
- [x] 返回树概览：标题、根节点、节点数、深度
- [x] `GET /api/trees/{tree_id}/nodes/{path}`
- [x] `GET /api/trees/{tree_id}/children/{path}`
- [x] 兼容别名：`GET /api/tree/{tree_id}/nodes/{path}`
- [x] 兼容别名：`GET /api/tree/{tree_id}/children/{path}`
- [x] 浏览接口测试

## Phase 6: Forest 能力

- [x] 内存 forest 注册表
- [x] `GET /api/forest`
- [x] `GET /api/forest/search/trees`
- [x] `GET /api/forest/search/nodes`
- [x] forest 查询复用当前 query/index 逻辑
- [x] forest 接口测试

## Phase 7: 生命周期管理

- [x] `DELETE /api/trees/{tree_id}`
- [x] 兼容别名：`DELETE /api/tree/{tree_id}`
- [x] 删除 tree 时同步清理 index
- [x] `GET /api/trees`
- [x] 列出当前已注册树
- [x] 生命周期接口测试

## Phase 8: 持久化

- [x] 明确持久化边界：原文件 / Tree / TreeIndex
- [x] 先实现 Tree 持久化
- [x] 再实现 TreeIndex 持久化
- [x] 服务启动时自动恢复注册表
- [x] 持久化测试

## Phase 9: 文档与契约

- [x] 补 `openapi.yaml`
- [x] 补 README 新服务章节
- [x] 给每个接口写最小请求/响应样例

## Backlog: 后续优化

- [ ] 真实数据库对接：把当前 JSON 文件 store 替换或补充为真实数据库后端
- [ ] 持久化优化：评估 Tree / TreeIndex 的表结构、索引策略、批量写入与恢复效率
- [ ] 生命周期一致性：补充数据库场景下的删除、覆盖重建、恢复一致性测试

## 建议实现顺序

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 7
7. Phase 6
8. Phase 8
9. Phase 9

## 第一批建议直接做

- [x] 新建 `treefyit/server/`
- [x] `GET /health`
- [x] `POST /api/trees`
- [x] 兼容旧路径：`POST /api/build`
- [x] `POST /api/trees/{tree_id}/index`
- [x] `GET /api/trees/{tree_id}/search/nodes`
