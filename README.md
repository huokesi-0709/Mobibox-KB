# MoniBox-KB（摩尼灵匣离线知识库）——生成/清洗/编译流水线

本项目用于为 “摩尼灵匣 MoniBox” 构建离线知识库（RAG），支持：
- 使用 DeepSeek 云端模型生成冷启动数据（标签体系 + QA）
- 将 QA 清洗为 60 字左右的 TTS 友好知识片段（chunks）
- 使用 BGE embedding + sqlite-vec 编译入库（rag.db）
- 为后续引入“专家/权威数据”预留来源分级、可追溯、可替换机制

---

## 1. 运行环境

### 1.1 Conda 虚拟环境（推荐）
```bash
conda create -n monibox-kb python=3.10 -y
conda activate monibox-kb
pip install -r requirements.txt
```

### 1.2 关键依赖说明
- `openai`：用于调用 DeepSeek（OpenAI 兼容接口）
- `json5`：用于解析 LLM 常见“不严格 JSON”
- `sentence-transformers`：构建期生成 embedding（BGE）
- `sqlite-vec`：向量检索扩展（最终落库为 sqlite + vec0）

---

## 2. 配置：.env

在项目根目录创建 `.env`（可从 `.env.example` 复制）：

必须配置：
- `DEEPSEEK_API_KEY`

示例：
```env
DEEPSEEK_API_KEY=xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_CHAT=deepseek-chat
DEEPSEEK_MODEL_REASONER=deepseek-reasoner

EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CHUNK_MAX_CHARS=60
CHUNK_MIN_CHARS=15

RAG_DB_PATH=build/rag.db
RUNTIME_PACK_PATH=build/runtime_pack.json
```

> 注意：本项目已在代码中固定从“项目根目录”加载 `.env`，不依赖 PyCharm 工作目录。

---

## 3. 项目目录说明（你只需要理解这三类目录）

### 3.1 `knowledge_src/`（编辑源 & 生成源）
- `00_meta.json`：枚举与基础标签（你手工维护）
- `01_sources.json`：来源注册表（你手工维护）
- `generated/`：脚本生成的中间产物（可随时删、可随时重生成）

### 3.2 `build/`（最终运行产物）
- `rag.db`：SQLite + sqlite-vec（端侧检索用）
- `runtime_pack.json`：运行期需要的配置包（以后会合并协议/意图/事件等）

### 3.3 `scripts/`（你日常只跑这里的脚本）
- `generate_taxonomy.py`：分页生成候选标签体系（并自动规范化）
- `normalize_taxonomy.py`：把候选标签归一化（类别/维度/tag_id 前缀等）
- `generate_qa.py`：分页生成 QA，自动补齐到目标数量
- `qa_to_chunks.py`：QA → 60 字 chunks（TTS 友好）
- `build_pack.py`：chunks → embedding → sqlite-vec 入库

---

## 4. 一键流程（建议按顺序执行）

### Step 1：生成候选标签体系（并自动规范化）
```bash
python scripts/generate_taxonomy.py
```

输出：
- `knowledge_src/generated/02_meta_candidates.json`（原始候选，可能杂乱）
- `knowledge_src/generated/02_meta_candidates.normalized.json`（规范化后，推荐后续使用）

同时会保存调试输出：
- `knowledge_src/generated/_debug_taxonomy_raw_batchXX.txt`

> 如果你看到 generated 目录里有 `_debug_taxonomy_raw_attempt*.txt`，那是旧版本失败残留，对运行无影响，可删。

---

### Step 2：生成 QA（分页补齐）
```bash
python scripts/generate_qa.py
```

输出：
- `knowledge_src/generated/11_qa_synth.json`

调试输出：
- `knowledge_src/generated/_debug_qa_raw_batchXX.txt`

说明：
- 即使模型一次只输出 20 条，脚本也会继续生成，直到补齐目标数量（默认 40，可在脚本里改）

---

### Step 3：QA 转 chunks（60 字切分）
```bash
python scripts/qa_to_chunks.py
```

输出：
- `knowledge_src/generated/12_chunks_synth.json`

---

### Step 4：编译入库（embedding + sqlite-vec）
```bash
python scripts/build_pack.py
```

输出：
- `build/rag.db`
- `build/runtime_pack.json`

---

## 5. 重要原则：哪些文件是“需要你维护的”，哪些是“可随时删的”？

### 你必须维护（长期资产）
- `knowledge_src/00_meta.json`
- `knowledge_src/01_sources.json`

### 脚本自动生成（中间产物，可删可重做）
- `knowledge_src/generated/*`

### 编译产物（可删可重做）
- `build/*`

---

## 6. 常见问题（FAQ）

### 6.1 generated 目录看起来很乱怎么办？
正常。`generated` 是“模型原材料 + debug”目录。
推荐做法：
- 你真正用来后续生成 QA 的，是 `02_meta_candidates.normalized.json`
- debug 文件可随时删，不影响运行

---

### 6.2 为什么 taxonomy/QA 会被截断或 JSON 解析失败？
LLM 输出长度过大或格式不严格是常态，所以：
- taxonomy 已改为分页生成（每批少量）
- JSON 解析使用 json5 兜底，并会自动补齐括号
- 每批 raw 输出都会保存到 debug 文件，便于排查

---

### 6.3 为什么我要求 40 条，模型只输出 20 条？
模型不保证严格遵守数量。新版 `generate_qa.py` 已分页补齐，会继续生成直到凑够目标数。

---

### 6.4 我重新运行脚本会不会覆盖文件？
会。以下文件会被自动覆盖（这是预期行为）：
- `knowledge_src/generated/02_meta_candidates.json`
- `knowledge_src/generated/02_meta_candidates.normalized.json`
- `knowledge_src/generated/11_qa_synth.json`
- `knowledge_src/generated/12_chunks_synth.json`
- `build/rag.db`
- `build/runtime_pack.json`

---

## 7. 下一步（未来扩展）
- 引入“专家/权威数据”作为新来源（source_id 更高权威）
- 在入库时做：指纹去重 + 语义近似去重
- 加入行为协议/传感器事件定义，使硬件动作与知识同源配置化
```

---

## 3) 你现在该怎么继续（最简单的建议）

1) 先不用管 generated 里那些旧 attempt 残留  
2) 直接按我 README 的顺序跑：
- `generate_taxonomy.py`（会产生 normalized 文件）
- `generate_qa.py`（会补齐数量，不再只出 20 条）
- 然后继续 `qa_to_chunks.py`、`build_pack.py`

---

如果你希望我再把“注释太少”这个点彻底解决，我可以把你现在项目里所有脚本都按同一个风格重写：  
- 每个文件开头说明“它干什么 / 输入输出文件 / 你应该什么时候跑它”  
- 每个关键变量旁边写“你只需要改这里”  
你只要回复我：你希望默认 QA 目标数量是 **40** 还是 **100**（用于 README 和脚本默认值统一）。