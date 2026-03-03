# Career Planning Assistant API

基于 RAG（检索增强生成）的职业规划助手后端服务。用户上传简历或职业介绍文档，系统通过 VLM OCR 提取文字、向量化存储，并提供智能问答接口。

## 功能特性

- **文档上传与解析**：支持 PDF / DOCX / PNG / JPG，通过多模态 LLM (VLM) 进行 OCR 文字提取
- **向量化存储**：使用 OpenAI Embedding + ChromaDB 进行文本分块与向量存储
- **RAG 智能问答**：基于用户上传的文档进行检索增强生成，支持多种检索策略
- **多 LLM 支持**：支持 OpenAI 和 Anthropic，可配置代理地址
- **Dify 集成**：提供 base64 JSON 接口和 OpenAPI Schema，可直接导入 Dify 作为工具调用

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制配置模板并填入你的 API Key：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`：

```yaml
# API Keys
openai_api_key: "sk-your-openai-key"
anthropic_api_key: "sk-ant-your-anthropic-key"

# 代理地址 (可选，不用代理则设为 null)
openai_base_url: null
anthropic_base_url: null

# Embedding 配置
embedding_model: "text-embedding-3-small"
embedding_api_key: "sk-your-embedding-key"    # 不填则复用 openai_api_key
embedding_base_url: "https://api.openai.com/v1"  # 不填则复用 openai_base_url
```

也可以通过环境变量覆盖（优先级高于 YAML）：

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://..."
export EMBEDDING_API_KEY="sk-..."
export EMBEDDING_BASE_URL="https://..."
```

### 3. 启动服务

```bash
# 方式一：直接运行
python app/main.py

# 方式二：uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 方式三：VSCode Debug (选择 "FastAPI Debug" 配置)
```

启动后访问：
- Swagger 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## API 接口

详细的接口文档见 [docs/API.md](docs/API.md)。

### 接口概览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/ingest` | 上传文档 (multipart form) |
| `POST` | `/v1/ingest_base64` | 上传文档 (JSON base64，适配 Dify) |
| `POST` | `/v1/chat` | RAG 智能问答 |

### 快速测试

```bash
# 健康检查
curl http://localhost:8000/health

# 上传文档 (表单方式)
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@resume.pdf" \
  -F "user_id=user_001" \
  -F "doc_type=resume"

# 问答
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "query": "根据我的简历，适合什么岗位？"}'
```

## Dify 集成

本项目提供了 `dify_openapi_schema.json`，可直接导入 Dify 平台作为自定义工具使用。

### 配置步骤

1. 启动后端服务，确保 Dify 能访问到服务地址
2. 在 Dify 中进入 **工具 → 自定义工具 → 创建自定义工具**
3. 将 `dify_openapi_schema.json` 的内容粘贴到 Schema 输入框
4. 修改 `servers.url` 为你的实际服务地址：
   - 本地 Docker 环境：`http://host.docker.internal:8000`
   - 局域网部署：`http://<your-ip>:8000`
5. 保存后即可在 Dify 工作流中使用以下两个工具：
   - **ingest_document**：上传并处理文档
   - **career_chat**：职业规划问答

### Dify 工作流示例

```
用户上传文件 → [ingest_document] → 文档入库
用户提问     → [career_chat]     → RAG 检索 + LLM 生成回答
```

### 注意事项

- Dify 不支持 multipart 上传，因此使用 `/v1/ingest_base64` 接口（文件需 base64 编码）
- `user_id` 用于隔离不同用户的文档，确保 Dify 工作流中传入正确的用户标识
- 如果 Dify 运行在 Docker 中，服务地址应使用 `host.docker.internal` 而非 `localhost`

## 核心概念

### 文档类型 (doc_type)

| 值 | 说明 |
|----|------|
| `resume` | 简历，支持 pinned 模式（全文注入） |
| `career_intro` | 职业介绍/岗位 JD |

### 简历模式 (resume_mode)

| 值 | 说明 |
|----|------|
| `rag` | 仅通过向量检索获取相关片段 |
| `pinned` | 将简历全文固定注入到每次问答的上下文中 |
| `hybrid` | 同时使用 pinned 全文 + RAG 检索补充 |

### 检索策略 (retrieval_policy)

| 值 | 说明 |
|----|------|
| `auto` | 根据问题内容自动判断（默认） |
| `resume_first` | 优先检索简历相关内容 |
| `career_first` | 优先检索职业介绍内容 |
| `blended` | 混合检索，不区分文档类型 |

## 处理流程

```
文档上传 (PDF/DOCX/PNG/JPG)
    │
    ▼
文件转换 (PDF→图片, DOCX→文本)
    │
    ▼
VLM OCR (多模态 LLM 提取文字)
    │
    ▼
文本清洗 (合并断行, 去页码)
    │
    ▼
分块 (中文友好分隔符, 800字/块)
    │
    ▼
Embedding (text-embedding-3-large)
    │
    ▼
存入 ChromaDB (带 metadata 过滤)
```
