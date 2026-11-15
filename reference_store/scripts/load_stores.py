"""
Nạp cases.jsonl vào 2 nơi lưu trữ thật sự được dùng khi chạy:
  - SQLite (db/cases.db)      : "sổ sách" - toàn bộ field, trạng thái verified, dễ SELECT/UPDATE
  - Chroma (vectorstore/)     : chỉ để TRUY HỒI ngữ nghĩa (RAG) - mỗi vector kèm
                                  metadata {"schema": "reference_case.v1", ...}
                                  để khi query, agent lọc đúng loại record cần.

Chạy: python3 reference_store/scripts/load_stores.py
"""
import json
import pathlib
import pickle
import sqlite3

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "reference_store" / "processed" / "cases.jsonl"
DB_PATH = ROOT / "reference_store" / "db" / "cases.db"
VECTOR_DIR = ROOT / "reference_store" / "vectorstore"
VECTORIZER_PATH = VECTOR_DIR / "tfidf_vectorizer.pkl"


class LocalTfidfEmbedding(EmbeddingFunction):
    """THAY THẾ TẠM cho embedding model thật (Voyage/OpenAI/sentence-transformers).
    Sandbox này không có mạng ra ngoài tới Hugging Face/OpenAI nên không tải được
    model embedding thật - dùng TF-IDF chạy 100% local để demo ĐÚNG cơ chế lưu
    trữ + truy hồi của Chroma. Khi triển khai thật, chỉ cần đổi class này thành
    class gọi API embedding thật, mọi chỗ khác (SQLite, schema, agent) giữ nguyên.
    """

    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer

    def __call__(self, input: Documents) -> Embeddings:
        return self.vectorizer.transform(input).toarray().tolist()


def load_cases():
    with open(CASES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_sqlite(cases):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE cases (
            case_id TEXT PRIMARY KEY,
            project TEXT,
            case_type TEXT,
            sentence_id INTEGER,
            sentence_text TEXT,
            model_element_id TEXT,
            model_element_name TEXT,
            code_path TEXT,
            label TEXT,
            verified BOOLEAN,
            source TEXT
        )
    """)
    conn.executemany(
        """INSERT INTO cases VALUES (:case_id,:project,:case_type,:sentence_id,
           :sentence_text,:model_element_id,:model_element_name,:code_path,
           :label,:verified,:source)""",
        [{**c, "label": str(c["label"])} for c in cases],
    )
    conn.commit()
    conn.close()
    print(f"SQLite: đã nạp {len(cases)} case vào {DB_PATH}")


def build_chroma(cases):
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    embeddable = [c for c in cases if c.get("sentence_text")]

    # fit TF-IDF 1 lần trên toàn bộ câu, rồi LƯU LẠI vectorizer đã fit - lúc query
    # sau này phải dùng đúng vectorizer này để câu hỏi mới rơi vào cùng không gian vector
    vectorizer = TfidfVectorizer(max_features=512).fit(
        [c["sentence_text"] for c in embeddable]
    )
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    if "reference_cases" in [c.name for c in client.list_collections()]:
        client.delete_collection("reference_cases")
    collection = client.create_collection(
        "reference_cases", embedding_function=LocalTfidfEmbedding(vectorizer)
    )

    collection.add(
        ids=[c["case_id"] for c in embeddable],
        documents=[c["sentence_text"] for c in embeddable],
        metadatas=[
            {
                "schema": "reference_case.v1",       # <-- tag để lọc đúng loại dữ liệu
                "case_type": c["case_type"],
                "label": str(c["label"]),
                "model_element_name": c["model_element_name"] or "",
                "verified": c["verified"],
            }
            for c in embeddable
        ],
    )
    print(f"Chroma: đã embed {len(embeddable)}/{len(cases)} case vào {VECTOR_DIR}")
    return collection


if __name__ == "__main__":
    cases = load_cases()
    build_sqlite(cases)
    build_chroma(cases)
