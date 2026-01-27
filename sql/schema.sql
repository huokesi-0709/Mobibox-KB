-- 元数据表：用于 SQL 过滤
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT NOT NULL UNIQUE,
  text TEXT NOT NULL,

  dimension TEXT NOT NULL,
  topic TEXT,
  risk TEXT NOT NULL,

  source_id TEXT NOT NULL,
  status TEXT NOT NULL,            -- 候选/启用/停用
  quality_score REAL NOT NULL DEFAULT 0,  -- 人工评分 or 自动评分
  fingerprint TEXT NOT NULL,       -- sha256:...

  tts_ok INTEGER NOT NULL DEFAULT 1,
  tts_style TEXT,

  tags_flat TEXT NOT NULL,         -- |scn_eq_trapped|med_crush|
  populations_flat TEXT NOT NULL   -- |adult|elder|
);

CREATE INDEX IF NOT EXISTS idx_chunks_dimension ON chunks(dimension);
CREATE INDEX IF NOT EXISTS idx_chunks_risk ON chunks(risk);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
CREATE INDEX IF NOT EXISTS idx_chunks_fp ON chunks(fingerprint);

-- 向量表（sqlite-vec）
-- 注意：维度需要与你的 embedding 维度一致（bge-small-zh-v1.5 常见为 512）
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
  embedding float[512]
);