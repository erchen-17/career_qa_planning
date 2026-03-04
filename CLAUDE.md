# CLAUDE.md — Career QA Planning

## 项目概述

基于 RAG 的职业规划问答后端服务。用户上传简历/职业介绍文档 → VLM OCR 提取文本 → 向量化存入 ChromaDB → 通过 LLM 进行智能问答并返回引用来源。

技术栈：FastAPI + LangChain + ChromaDB + OpenAI/Anthropic

## 项目结构

```
app/
├── core/config.py          # 配置管理（YAML + 环境变量覆盖）
├── ingest/                 # 文档摄入管线
│   ├── service.py          # 摄入编排（入口）
│   ├── converters.py       # PDF→图片, DOCX→文本, 图片直通
│   ├── cleaners.py         # OCR 后处理（断行合并、页码清除）
│   ├── chunkers.py         # 中文感知分块（800 字/120 重叠）
│   ├── dedup.py            # 文档去重（相似度检测 + 旧文档删除）
│   ├── file_utils.py       # 文件类型多级检测、扩展名校验、标签解析
│   └── schemas.py          # 请求/响应模型
├── llm/
│   ├── router.py           # LLM 工厂（OpenAI/Anthropic）
│   └── vlm.py              # 多模态 OCR
├── rag/
│   ├── service.py          # 检索策略 + LLM 调用（含联网搜索绑定）
│   ├── prompts.py          # 系统提示词 & 上下文组装
│   └── schemas.py          # Chat 请求/响应模型
├── store/
│   ├── chroma_store.py     # ChromaDB 封装（单例）
│   └── resume_cache.py     # 简历全文 JSON 缓存（pinned 模式）
└── main.py                 # FastAPI 入口 & 路由定义
```

## API 端点

- `GET /health` — 健康检查
- `POST /v1/ingest` — 上传文件（multipart）
- `POST /v1/ingest_base64` — 上传文件（base64 JSON，供 Dify 调用）
- `POST /v1/ingest_url` — 通过 URL 下载并上传（供 Dify 文件变量调用）
- `POST /v1/ingest_batch` — 批量上传（multipart 多文件）
- `POST /v1/chat` — RAG 问答（支持联网搜索、简历全文注入）

## 关键设计

- 用户隔离：所有 chunk 元数据带 `user_id`，检索时按 user_id 过滤
- 简历模式（默认 `pinned`）：`rag`（纯向量检索）/ `pinned`（全文注入，不分块不入库）/ `hybrid`（两者兼用）
- 检索策略：`auto`（关键词自动判断）/ `resume_first` / `career_first` / `blended`
- 联网搜索：仅 OpenAI，通过 `web_search_preview` 工具绑定，全局配置开关
- 文档去重：入库后自动检测相似文档，超过阈值则删除旧文档（可配置）
- 文件类型检测：多级回退（文件名 → URL 路径 → Content-Type → magic bytes），逻辑在 `ingest/file_utils.py`
- 中文分块：使用 `。；，` 等中文标点作为分隔符
- Dify 集成：`dify_openapi_schema.json` 供自定义工具导入，`dify/职业规划助手.yml` 为可直接导入的完整工作流 DSL
- simple_response 模式：`/v1/chat` 默认只返回纯文本 answer，适配 Dify

## 开发约定

- 配置文件：`config.yaml`（含密钥，已 gitignore），模板见 `config.yaml.example`
- 配置分区：Server → LLM → Embedding → Storage → Ingest → RAG
- 运行时数据目录：`data/`（chroma、uploads、resume_cache，均已 gitignore）
- Python 依赖：`requirements.txt`
- 启动方式：`uvicorn app.main:app` 或 `python -m app.main`
- 语言：代码注释和文档使用中文

## 敏感文件（勿提交）

- `config.yaml` — 含 API 密钥
- `data/` — 运行时生成的用户数据
- `.env` — 如存在，含环境变量密钥
