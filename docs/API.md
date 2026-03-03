# API Reference

Career Planning Assistant API 接口文档。

Base URL: `http://localhost:8000`

---

## GET /health

健康检查。

**Response**

```json
{ "status": "ok" }
```

---

## POST /v1/ingest

上传文档（multipart form 方式）。支持 PDF / DOCX / PNG / JPG 文件，后端会进行 OCR 提取、文本分块、向量化存储。

**Content-Type:** `multipart/form-data`

**Parameters**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | file | 是 | - | 上传的文件 |
| `user_id` | string | 是 | - | 用户唯一标识 |
| `doc_type` | string | 是 | - | 文档类型：`resume` / `career_intro` |
| `provider` | string | 否 | `openai` | VLM OCR 使用的 LLM 提供商：`openai` / `anthropic` |
| `model` | string | 否 | `gpt-5.2` | VLM OCR 使用的模型 |
| `tags` | string | 否 | - | 逗号分隔的标签 |
| `resume_mode` | string | 否 | `rag` | 简历处理模式：`rag` / `pinned` / `hybrid` |

**示例**

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@resume.pdf" \
  -F "user_id=user_001" \
  -F "doc_type=resume" \
  -F "provider=openai" \
  -F "model=gpt-5.2" \
  -F "resume_mode=rag"
```

**Response 200**

```json
{
  "status": "ok",
  "doc_id": "a1b2c3d4-...",
  "doc_type": "resume",
  "pages": 3,
  "chunks": 12,
  "text_preview": "张三，软件工程师，5年工作经验..."
}
```

---

## POST /v1/ingest_base64

上传文档（JSON base64 方式）。功能与 `/v1/ingest` 相同，但文件内容以 base64 编码传递。专为 Dify 等不支持 multipart 上传的平台设计。

**Content-Type:** `application/json`

**Request Body**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file_name` | string | 是 | - | 文件名（需含扩展名，如 `resume.pdf`） |
| `file_content_base64` | string | 是 | - | 文件内容的 base64 编码 |
| `user_id` | string | 是 | - | 用户唯一标识 |
| `doc_type` | string | 是 | - | 文档类型：`resume` / `career_intro` |
| `provider` | string | 否 | `openai` | VLM OCR 使用的 LLM 提供商 |
| `model` | string | 否 | `gpt-5.2` | VLM OCR 使用的模型 |
| `tags` | string | 否 | - | 逗号分隔的标签 |
| `resume_mode` | string | 否 | `rag` | 简历处理模式：`rag` / `pinned` / `hybrid` |

**示例**

```bash
curl -X POST http://localhost:8000/v1/ingest_base64 \
  -H "Content-Type: application/json" \
  -d '{
    "file_name": "resume.pdf",
    "file_content_base64": "JVBERi0xLjQK...",
    "user_id": "user_001",
    "doc_type": "resume"
  }'
```

**Response 200**

```json
{
  "status": "ok",
  "doc_id": "a1b2c3d4-...",
  "doc_type": "resume",
  "pages": 3,
  "chunks": 12,
  "text_preview": "张三，软件工程师，5年工作经验..."
}
```

**Error Responses**

| 状态码 | 说明 |
|--------|------|
| 400 | 文件类型不支持、base64 无效或文件为空 |
| 422 | 请求参数校验失败 |
| 500 | 服务端处理异常（OCR / Embedding / 存储） |

---

## POST /v1/chat

RAG 智能问答。基于用户上传的文档进行向量检索，结合 LLM 生成回答，并返回引用来源。

**Content-Type:** `application/json`

**Request Body**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user_id` | string | 是 | - | 用户唯一标识 |
| `query` | string | 是 | - | 用户的问题 |
| `provider` | string | 否 | `openai` | LLM 提供商：`openai` / `anthropic` |
| `model` | string | 否 | `gpt-4.1-mini` | 使用的模型 |
| `top_k` | integer | 否 | `6` | 检索的文档块数量 (1-20) |
| `retrieval_policy` | string | 否 | `auto` | 检索策略：`auto` / `resume_first` / `career_first` / `blended` |
| `resume_mode` | string | 否 | `rag` | 简历模式：`rag` / `pinned` / `hybrid` |

**示例**

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "query": "根据我的简历，适合什么岗位？",
    "retrieval_policy": "auto",
    "resume_mode": "hybrid"
  }'
```

**Response 200**

```json
{
  "answer": "根据您的简历，您在声学算法方面有丰富经验，适合以下岗位...",
  "citations": [
    {
      "doc_id": "a1b2c3d4-...",
      "doc_type": "resume",
      "file_name": "resume.pdf",
      "page": 1,
      "chunk_id": 3,
      "snippet": "5年声学算法开发经验，熟悉..."
    },
    {
      "doc_id": "e5f6g7h8-...",
      "doc_type": "career_intro",
      "file_name": "声学AI算法.png",
      "page": 1,
      "chunk_id": 1,
      "snippet": "岗位要求：声学信号处理..."
    }
  ],
  "debug": {
    "retrieval_policy": "blended",
    "used_resume_pinned": true,
    "retrieved_chunks": 6
  }
}
```

**Error Responses**

| 状态码 | 说明 |
|--------|------|
| 422 | 请求参数校验失败 |
| 500 | LLM 调用或检索异常 |
