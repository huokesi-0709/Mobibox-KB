下面按你的要求分两部分做：

1) **更新并“定稿”项目结构 + 给出一份完整 README.md（可直接覆盖你根目录 README）**  
2) **开始优化“标签体系收敛”**：让 `generate_qa.py` 只能产生/保留标准标签（从 `02_meta_candidates.normalized.json` 来），并把不合规标签自动剔除/纠正，保证后续 `--tags` 过滤和协议路由可靠。

---

# Part A：项目结构定稿（你现在该是什么样）

## A1. 最终推荐项目结构（请对照整理）
```
MoniBox-KB/
  README.md
  requirements.txt
  .env
  .env.example
  .gitignore

  build/                         # 构建产物（可删可重建）
    rag.db
    runtime_pack.json

  docs/                          # 文档（长期资产）
    01_knowledgebase_roadmap.md
    02_radxa_deploy_and_hardware_integration.md

  knowledge_src/                 # 编辑源（长期资产）+ 中间产物（可删可重建）
    00_meta.json
    01_sources.json
    generated/
      02_meta_candidates.json
      02_meta_candidates.normalized.json
      11_qa_synth.json
      12_chunks_synth.json
      ...debug...

  models/                        # 本地模型（不建议进git）
    embedding/
      bge-small-zh-v1.5/

  scoring_system/                # 评分系统配置与文档（长期资产）
    policy.json
    README.md

  sql/
    schema.sql

  monibox_kb/                    # 核心库（长期资产）
    __init__.py
    paths.py
    config.py
    deepseek_client.py
    utils_json.py
    text_clean.py
    chunking.py
    dedup.py
    embedding.py
    db_sqlitevec.py
    scoring/
      __init__.py
      rerank.py
    tags/                        # （新增）标签收敛模块
      __init__.py
      registry.py

  scripts/                       # 运行入口（命令行工具）
    __init__.py
    generate_taxonomy.py
    normalize_taxonomy.py
    generate_qa.py
    qa_to_chunks.py
    build_pack.py
    query_demo.py
    rate_chunk.py
    test_sqlitevec.py
    cleanup_generated.py
```

### A2. 你截图里那个 `scripts/knowledge_src/...` 是不是重复错了？
是的，**必须删掉**（或把里面文件搬回根目录 `knowledge_src/`）。

- 正确且唯一的数据目录：`MoniBox-KB/knowledge_src/`
- 不应存在：`MoniBox-KB/scripts/knowledge_src/`

> 原因：早期脚本如果用相对路径，会在 `scripts/` 下误创建 `knowledge_src` 分身。  
> 现在我们已用 `paths.py` 固定到项目根目录，分身目录只会制造混乱。

---

# Part B：更新 README.md（请你直接整文件覆盖）

把下面内容复制到项目根目录 `README.md`（覆盖原文件）：

```markdown
# MoniBox-KB：离线知识库（RAG）构建与端侧运行的工程化流水线

本项目用于为 “摩尼灵匣 MoniBox” 构建离线知识库（RAG）：
- DeepSeek 生成冷启动数据（taxonomy + QA）
- QA 清洗成 60 字 TTS 友好的 chunks
- BGE embedding + sqlite-vec 写入 `build/rag.db`
- 支持评分闭环（评分/启用/停用立即影响检索排序）
- 为 Radxa 端部署与传感器协议联动预留接口

---

## 0. 你需要牢记的两条原则
1) `knowledge_src/generated/` 和 `build/` 都是 **可删可重建** 的产物（调试阶段反复重建很正常）
2) 运行脚本统一用：
   `python -m scripts.xxx`
   （避免 import 路径问题）

---

## 1. 环境安装

### 1.1 conda 环境
```bash
conda create -n monibox-kb python=3.10 -y
conda activate monibox-kb
pip install -r requirements.txt
```

### 1.2 .env
从 `.env.example` 复制为 `.env`，至少配置：
- `DEEPSEEK_API_KEY`
- `EMBEDDING_MODEL=models/embedding/bge-small-zh-v1.5`

---

## 2. 目录说明（哪些是长期资产，哪些可删）

### 长期资产（不要随便删）
- `knowledge_src/00_meta.json`
- `knowledge_src/01_sources.json`
- `sql/schema.sql`
- `docs/`
- `scoring_system/`
- `monibox_kb/`

### 可删可重建
- `knowledge_src/generated/`
- `build/`（尤其 `build/rag.db`）

---

## 3. 一键构建流程（从0到可检索）

### Step 1：生成候选标签体系（taxonomy）
```bash
python -m scripts.generate_taxonomy
```
输出：
- `knowledge_src/generated/02_meta_candidates.json`
- `knowledge_src/generated/02_meta_candidates.normalized.json`

### Step 2：生成 QA（已做“标签收敛校验”）
```bash
python -m scripts.generate_qa
```
输出：
- `knowledge_src/generated/11_qa_synth.json`

### Step 3：QA → chunks（60 字切分、ID 唯一）
```bash
python -m scripts.qa_to_chunks
```
输出：
- `knowledge_src/generated/12_chunks_synth.json`

### Step 4：构建向量库 rag.db
```bash
python -m scripts.build_pack
```
输出：
- `build/rag.db`
- `build/runtime_pack.json`

---

## 4. 检索验证（query_demo）

```bash
python -m scripts.query_demo --q "我好害怕，喘不过气" --topk 5
```

参数说明（完整）：
- `--q`            必填：查询文本
- `--topk`         返回多少条
- `--dimension`    可选：限定维度
- `--risk`         可选：限定风险等级（逗号分隔，如 "中,高"）
- `--tags`         可选：必须包含的标签ID（逗号分隔，如 "psy_panic,act_breath_pacing"）
- `--status`       可选：限定状态（逗号分隔），默认排除“停用”
- `--pool_mult`    可选：候选池倍率（默认 8，用于评分重排）

---

## 5. 评分闭环（rate_chunk + 重排）

### 5.1 打分/启用/停用
```bash
python -m scripts.rate_chunk --display_id "<显示ID>" --score 5 --status 启用
python -m scripts.rate_chunk --display_id "<显示ID>" --status 停用
```

### 5.2 再查询（排序会受评分影响）
```bash
python -m scripts.query_demo --q "我好害怕，喘不过气" --topk 5
```

评分策略配置：
- `scoring_system/policy.json`

---

。