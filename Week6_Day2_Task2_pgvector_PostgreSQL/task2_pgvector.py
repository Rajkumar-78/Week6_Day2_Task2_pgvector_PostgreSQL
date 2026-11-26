import os
import psycopg
from sentence_transformers import SentenceTransformer

DSN=os.getenv("PG_DSN","host=localhost port=5432 dbname=vectordb user=postgres password=postgres")
MODEL="sentence-transformers/all-MiniLM-L6-v2"
DATA=[
("HR onboarding checklist and new employee orientation","HR"),
("Annual leave policy and vacation request","HR"),
("Performance review and employee feedback","HR"),
("Workplace conduct and grievance process","HR"),
("Finance quarterly budget and revenue forecast","Finance"),
("Expense reimbursement and travel receipt rules","Finance"),
("Invoice processing and accounts payable","Finance"),
("Cost center planning and budget approval","Finance"),
("Sales lead qualification and customer outreach","Sales"),
("Customer contract renewal and account management","Sales"),
("Product demonstration for potential customers","Sales"),
("Sales quota and territory planning","Sales"),
("IT password reset and multi-factor authentication","IT"),
("Laptop provisioning and software access request","IT"),
("Incident response and security escalation","IT"),
("Network monitoring and system uptime","IT"),
("Customer support ticket triage and response","Support"),
("Service level agreement and support coverage","Support"),
("Help center article and frequently asked questions","Support"),
("Escalated complaint resolution and customer care","Support")]
QUERY="How does a new employee get started and request time off?"
def vstr(v): return "["+",".join(f"{float(x):.8f}" for x in v)+"]"
def explain(cur,title,q,force=False):
    print(f"\\n--- EXPLAIN ANALYZE {title} ---")
    cur.execute("SET enable_seqscan = off" if force else "SET enable_seqscan = on")
    cur.execute("""EXPLAIN (ANALYZE, BUFFERS)
      SELECT id, content, category FROM documents
      ORDER BY embedding <=> %s::vector LIMIT 3""",(q,))
    for row in cur.fetchall(): print(row[0])
    cur.execute("RESET enable_seqscan")
def main():
    model=SentenceTransformer(MODEL)
    embeddings=model.encode([t for t,c in DATA],normalize_embeddings=True)
    q=vstr(model.encode(QUERY,normalize_embeddings=True))
    with psycopg.connect(DSN) as conn:
      with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS documents")
        cur.execute("""CREATE TABLE documents(
          id SERIAL PRIMARY KEY, content TEXT NOT NULL,
          embedding vector(384) NOT NULL, category TEXT NOT NULL)""")
        for (content,cat),emb in zip(DATA,embeddings):
          cur.execute("INSERT INTO documents(content,embedding,category) VALUES(%s,%s::vector,%s)",
                      (content,vstr(emb),cat))
        conn.commit()
        print("Inserted rows:",len(DATA))
        explain(cur,"before HNSW index (natural planner)",q)
        cur.execute("CREATE INDEX documents_embedding_hnsw ON documents USING hnsw (embedding vector_cosine_ops)")
        conn.commit()
        explain(cur,"after HNSW index (index plan demonstration)",q,True)
        print("\\n--- 3 nearest documents (parameterized) ---")
        cur.execute("""SELECT id,content,category,1-(embedding <=> %s::vector) AS cosine_similarity
          FROM documents ORDER BY embedding <=> %s::vector LIMIT 3""",(q,q))
        for row in cur.fetchall(): print(row)
        print("\\n--- nearest documents WHERE category = HR ---")
        cur.execute("""SELECT id,content,category,1-(embedding <=> %s::vector) AS cosine_similarity
          FROM documents WHERE category=%s ORDER BY embedding <=> %s::vector LIMIT 3""",(q,"HR",q))
        for row in cur.fetchall(): print(row)
if __name__=="__main__": main()
