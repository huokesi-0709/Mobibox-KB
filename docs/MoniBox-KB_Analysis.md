# MoniBox-KB 项目深度解析文档

> **生成时间**: 2026年1月28日  
> **生成者**: Gemini CLI Agent  
> **目标读者**: 核心开发者、架构师、系统集成人员

---

## 1. 项目概览 (Executive Summary)

### 1.1 项目背景
**MoniBox (摩尼灵匣)** 是一款面向灾害（如地震）受困人群的**离线救援陪伴终端**。在极端环境下，受困者需要心理安抚、生存指导和状态监测，但云端大模型无法连接，通用端侧模型容易产生幻觉（如错误的急救建议）。

**MoniBox-KB** 是该终端的**大脑构建与运行系统**。它不是一个简单的问答机器人，而是一个**“协议优先、RAG增强、多模态融合”**的工程化流水线。它负责生产高可靠的离线知识库，并在低算力硬件（Radxa Zero 3W）上通过传感器和语音与受困者交互。

### 1.2 核心解决的问题
1.  **离线可用性**: 在无网环境下提供基于 RAG 的智能问答。
2.  **安全性与可控性**: 解决 LLM 幻觉问题。对于高风险场景（如余震、大出血），使用确定性的**Protocol (协议)** 覆盖生成模型，绝不把生死交给概率。
3.  **低延迟与 TTS 友好**: 针对语音交互优化，知识被切分为 **60字短句 (Chunks)**，确保 TTS 播报流畅且关键信息不被遗漏。
4.  **数据冷启动**: 利用 DeepSeek API 自动化生成并清洗知识数据，解决初期缺乏专业救援语料的问题。

### 1.3 当前状态与里程碑
- **已完成**:
    - [x] 基于 DeepSeek 的自动化数据生成流水线 (Taxonomy -> QA -> Chunks)。
    - [x] 基于 `sqlite-vec` 的高性能离线向量检索 (RAG)。
    - [x] 评分与闭环系统 (Scoring System)，支持人工干预排序。
    - [x] 协议引擎 (Protocol Engine) 原型，支持事件与关键词触发。
    - [x] Windows 端 E2E 演示 (ASR -> RAG -> LLM -> TTS)。
- **进行中**:
    - [ ] 标签体系 (Taxonomy) 的严格收敛与标准化。
    - [ ] 硬件层 (Radxa GPIO/UART) 的深度集成。
    - [ ] 真实专业医疗/救援数据源的引入与清洗。

---

## 2. 项目架构 (System Architecture)

项目分为 **构建侧 (PC/Cloud)** 和 **运行侧 (Device/Radxa)** 两大域。

### 2.1 整体架构图

```mermaid
graph TD
    subgraph "构建侧 (PC)"
        Src[Source Data] -->|DeepSeek Gen| Tax[Taxonomy]
        Tax -->|DeepSeek Gen| QA[QA Pairs]
        QA -->|Chunking| Chunks[Atomic Chunks]
        Chunks -->|Embedding| Vectors[Vector Embeddings]
        Vectors -->|Build| DB[(build/rag.db)]
        Scoring[Scoring Policy] -.->|Re-rank| DB
    end

    subgraph "运行侧 (Radxa/Runtime)"
        Sensors[Sensors/IMU] -->|Events| Session
        Mic[Microphone] -->|ASR| Session
        
        Session -->|1. Check| Protocol{Protocol Engine}
        Protocol -- Yes --> Action[Execute Action]
        Protocol -- No --> Router[Router]
        
        Router -->|2. Search| RAG[RagEngine]
        RAG -->|Query + Tags| DB
        DB -->|Context| LLM[LLM (Qwen/Llama)]
        LLM -->|Stream| TTS[TTS Speaker]
    end
```

### 2.2 目录结构深度解析

| 目录/文件 | 类型 | 详细职责 |
| :--- | :--- | :--- |
| **`monibox_kb/`** | **核心库** | 项目的 Python 源码包，包含所有逻辑。 |
| ├── `runtime/` | 运行时 | 包含主循环 `main_cli.py`，会话管理 `session.py`，以及两大引擎：`protocol_engine.py` 和 `rag_engine.py`。 |
| ├── `scoring/` | 算法 | `rerank.py` 实现基于距离、质量分、状态的混合排序算法。 |
| ├── `hw/` | 硬件抽象 | `mock_hw.py` (PC模拟) 和 `windows_hw.py` (Windows实现)，未来会有 `radxa_hw.py`。 |
| ├── `db_sqlitevec.py` | 数据库 | 封装 `sqlite-vec`，处理向量存储与检索。 |
| **`knowledge_src/`** | **数据源** | 知识库的“源代码”。 |
| ├── `generated/` | 临时产物 | 脚本生成的中间文件，可随时删除重建。 |
| ├── `protocols.json` | 规则配置 | 定义高优先级触发规则（如：检测到 Shake -> 触发“防冲击”协议）。 |
| **`scripts/`** | **工具链** | 数据处理流水线的入口。 |
| ├── `generate_*.py` | 生成器 | 调用 DeepSeek 生成标签、QA。 |
| ├── `build_pack.py` | 构建器 | 将 JSON 数据打包进 SQLite 数据库。 |
| ├── `query_demo.py` | 测试 | 命令行测试检索效果。 |
| **`build/`** | **产物** | 最终部署到设备的文件 (`rag.db`, `runtime_pack.json`)。 |
| **`apps/`** | **应用** | `win_e2e_demo.py`: Windows 上的端到端全流程模拟器。 |

---

## 3. 核心工作流详解 (Workflows)

### 3.1 知识库构建流水线 (The Pipeline)

这是将非结构化信息转化为可检索资产的过程。

1.  **分类体系生成 (`generate_taxonomy.py`)**:
    *   利用 LLM 分析需求，生成层级化的标签树 (Dimension -> Domain -> Tag)。
    *   产出: `02_meta_candidates.json`。
2.  **QA合成 (`generate_qa.py`)**:
    *   遍历每个标签，Prompt DeepSeek 生成针对该标签的常见问答对。
    *   **关键点**: 这里会进行“标签收敛”，确保生成的 QA 能够回落到标准标签上。
3.  **切片与清洗 (`qa_to_chunks.py`)**:
    *   将长 QA 对拆解为 **Atomic Chunks** (原子片段)。
    *   **约束**: 长度限制在 60 字左右，语义完整，便于 TTS 一口气读完。
    *   计算去重指纹 (Fingerprint)。
4.  **向量化与构建 (`build_pack.py`)**:
    *   使用 BGE 模型 (`embed_texts`) 计算 Chunk 的向量。
    *   写入 `sqlite-vec` 表 `vec_chunks`。
    *   打包元数据到 `runtime_pack.json`。

### 3.2 运行时交互逻辑 (Runtime Logic)

这是设备启动后的运行逻辑，主要在 `monibox_kb/runtime/session.py` 中编排。

1.  **输入感知**:
    *   **ASR**: 麦克风录音 -> Faster-Whisper 转文字。
    *   **Event**: 传感器 (IMU) 检测到震动/跌落，或系统事件 (Low Battery)。
2.  **协议判定 (`protocol_engine.py`)**:
    *   系统首先检查 `protocols.json`。
    *   **匹配逻辑**: 支持 `any_of` (满足任一), `all_of` (满足所有) 组合。
    *   **触发条件**: 文本包含关键词 (如"救命")、事件匹配 (如"imu_shake")、标签匹配。
    *   **结果**: 如果匹配成功，直接返回预设动作/回复，**跳过 LLM**。这是安全的关键。
3.  **路由与检索 (`rag_engine.py`)**:
    *   如果无协议触发，进入 RAG 流程。
    *   **AutoRouter**: LLM 快速分析用户意图，输出查询关键词和目标标签 (Tags)。
    *   **Search**: 在 `rag.db` 中检索，结合 **语义相似度 (Cosine Distance)** + **质量分 (Quality Score)** + **状态 (Status)** 进行混合重排 (Rerank)。
    *   **多样性**: 强制同一 Group (来源组) 的结果去重，保证回答丰富性。
4.  **生成与播报**:
    *   LLM (Qwen/Llama) 根据检索到的 Top-K 片段生成自然回复。
    *   流式输出给 TTS 模块进行语音合成。

---

## 4. 关键技术点与优化

### 4.1 混合评分与重排 (Scoring & Rerank)
为了让系统越用越聪明，项目引入了 `scoring_system`。
*   **机制**: 每个 Chunk 都有 `quality_score` (默认1.0) 和 `status` (候选/启用/停用)。
*   **优化**: 检索时，`final_distance = distance * f(quality, status)`。
*   **效果**: 被标记为“优质”的回答会更容易浮现，被标记为“停用”的回答即使语义相似度高也会被屏蔽。

### 4.2 SQLite-Vec 的应用
项目放弃了沉重的向量数据库 (如 Milvus/Chroma)，选择了极其轻量的 `sqlite-vec` 插件。
*   **优势**: 单文件 (`rag.db`)，零依赖部署，非常适合嵌入式环境 (Radxa)。
*   **性能**: 支持 SIMD 加速的向量距离计算，满足毫秒级检索需求。

### 4.3 标签收敛 (Tag Convergence)
当前正在解决的重点问题。
*   **问题**: LLM 生成数据时容易“发散”，创造出成百上千个非标标签，导致路由失效。
*   **解法**: 建立 `registry.py` 和归一化脚本，强制所有生成内容映射回预定义的标准标签集。

---

## 5. 未来展望 (Roadmap)

1.  **硬件完全体**: 迁移至 Radxa Zero 3W，驱动 LED 灯带（呼吸灯引导）、OLED 屏幕（显示状态）和 IMU 传感器（自动检测余震）。
2.  **多模态协议**: 协议不仅触发语音，还能触发灯光（如红色闪烁警报）和震动。
3.  **真实数据清洗**: 接入红十字会急救指南、地震局官方手册，替换/增强目前的合成数据。
4.  **端侧模型微调**: 对 Qwen-0.5B 进行 SFT (Supervised Fine-Tuning)，使其更擅长利用检索到的短句进行安抚性对话。

---
