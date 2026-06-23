# treefyit 健康度修复记录

> 日期：2026-06-23  
> 对应审计中识别的全部问题项，均已在本分支落地。

## 修复清单

| # | 问题 | 修复 | 验证方式 |
|---|------|------|----------|
| 1 | 无 CI，改动易回归 | 新增 `.github/workflows/ci.yml`：`uv sync` → `pytest -m "not integration"` | GitHub Actions / 本地 `uv run pytest -m "not integration"` |
| 2 | HTTP 服务层零测试 | 新增 `tests/test_server.py`、`tests/test_store.py` | pytest |
| 3 | OpenAPI / README 与 Forest API 漂移 | 补充 `/api/forest`、`/api/forest/search` 及 schema | 对照 `server.py` |
| 4 | URL 抓取 SSRF 风险 | `src/parser/url.py` 增加公网地址校验 | `tests/test_url.py::test_*ssrf*` |
| 5 | 测试导入脆弱、`test_llm` 孤立失败 | `pyproject.toml` 配置 `pythonpath`；`conftest.py`；移除手写 `sys.path` | 单独跑任意测试文件 |
| 6 | LLM 失败静默返回 `""` | `LLMError` 在重试耗尽后抛出 | `src/llm/client.py` |
| 7 | 解析失败返回 HTTP 500 | `POST /api/build` 解析异常改为 200 + `error` 字段 | `test_build_parse_error_returns_json` |
| 8 | `_register_or_none` 吞异常 | 改为 `logger.exception` 记录 | 日志可见 |
| 9 | 无 `.env.example` | 新增 `.env.example` | 新环境复制即用 |
| 10 | CORS 硬编码 `*` | 支持 `TREEFYIT_CORS_ORIGINS` 环境变量（默认仍 `*`） | README 说明 |

## 本地验证

```bash
uv sync
uv run pytest -m "not integration" -q          # 默认 CI 套件
uv run pytest -m integration -q                # 需配置 DEEPSEEK_API_KEY
```

## 未纳入本次范围

- **提交未入库的大量 WIP**：需人工 `git add` / `commit`（按功能拆分更佳）。
- **生产级认证 / 限流**：属产品决策，不在工程修复范围内。

## 文件变更索引

```
.github/workflows/ci.yml   # 新增
.env.example               # 新增
FIXES.md                   # 本文档
pyproject.toml             # pytest pythonpath + markers
tests/conftest.py          # 新增
tests/test_server.py       # 新增
tests/test_store.py        # 新增
tests/test_llm.py          # integration 标记
tests/test_*.py            # 移除冗余 sys.path
src/parser/url.py          # SSRF 防护
src/server/server.py       # 解析错误一致性、注册日志、可配置 CORS
src/llm/client.py          # LLMError
src/llm/__init__.py        # 导出 LLMError
openapi.yaml                 # Forest API
README.md                    # Forest API + CORS + LLM 行为
```
