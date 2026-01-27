# MoniBox 向量知识库迁移到 Radxa & 传感器融合执行指南

本文说明：
1) 在 PC 上构建好 rag.db 后，如何迁移到 Radxa Zero 3W
2) Radxa 端运行时模块如何与麦克风/IMU/灯带/屏幕结合，实现“协议优先 + RAG补充”的效果

---

## 1. PC 构建产物清单（需要迁移的东西）

构建完成后，关键文件通常是：

- build/rag.db
  - SQLite + sqlite-vec 向量表（chunks + embedding）
- build/runtime_pack.json
  - 运行期配置（枚举/标签体系/来源注册表；未来会加入协议/意图/传感器事件）
- 模型文件
  - Embedding：BAAI/bge-small-zh-v1.5（Radxa 端需要计算“用户问题”的向量）
  - LLM：Qwen1.5-0.5B-Chat-GGUF（llama.cpp 运行）
  - 可选：ASR/TTS 模型（视你的选型）

建议目录规划（Radxa）：
/opt/monibox/
  data/
    rag.db
    runtime_pack.json
  models/
    bge-small-zh-v1.5/
    qwen1.5-0.5b-chat.gguf
  app/
    monibox_runtime.py
    ...
  logs/

---

## 2. 迁移方式（最简单可靠）

### 2.1 使用 scp/rsync
在 PC 上：
```bash
scp build/rag.db radxa@<ip>:/opt/monibox/data/
scp build/runtime_pack.json radxa@<ip>:/opt/monibox/data/