CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  category TEXT NOT NULL
);
-- Python parameterized query:
-- SELECT id,content,category FROM documents
-- ORDER BY embedding <=> %s::vector LIMIT 3;
CREATE INDEX documents_embedding_hnsw
ON documents USING hnsw (embedding vector_cosine_ops);
-- Category filtered:
-- SELECT id,content,category FROM documents WHERE category=%s
-- ORDER BY embedding <=> %s::vector LIMIT 3;
