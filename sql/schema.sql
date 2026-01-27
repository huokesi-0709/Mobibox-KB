-- 元数据表：用于 SQL 过滤 + 测试调权
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id TEXT NOT NULL UNIQUE,     -- 片段ID（机器主键，必须唯一）
  display_id TEXT,                   -- 显示ID（给测试人员点名用）
  group_id TEXT,                     -- 片段组ID（同一条QA切出来的一组）

  text TEXT NOT NULL,

  dimension TEXT NOT NULL,
  topic TEXT,
  risk TEXT NOT NULL,

  source_id TEXT NOT NULL,
  status TEXT NOT NULL,              -- 候选/启用/停用
  quality_score REAL NOT NULL DEFAULT 0,  -- 测试评分（可用于重排）
  fingerprint TEXT NOT NULL,         -- sha256:...

  tts_ok INTEGER NOT NULL DEFAULT 1,
  tts_style TEXT,

  tags_flat TEXT NOT NULL,           -- |tag1|tag2|
  populations_flat TEXT NOT NULL     -- |成人|哮喘|
);

CREATE INDEX IF NOT EXISTS idx_chunks_dimension ON chunks(dimension);
CREATE INDEX IF NOT EXISTS idx_chunks_risk ON chunks(risk);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
CREATE INDEX IF NOT EXISTS idx_chunks_fp ON chunks(fingerprint);
CREATE INDEX IF NOT EXISTS idx_chunks_display_id ON chunks(display_id);
CREATE INDEX IF NOT EXISTS idx_chunks_group_id ON chunks(group_id);

-- 向量表（sqlite-vec）
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
  embedding float[512]
);