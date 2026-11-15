"""
Consistency Checking Agent - bản demo chạy thật (không gọi LLM thật trong sandbox
này, nhưng toàn bộ phần GHÉP DỮ LIỆU + VALIDATE SCHEMA là thật 100%).

Luồng:
  1. Đọc architecture_findings.json (output Architecture Extraction Agent)
  2. Đọc code_findings.json          (output Code Analysis Agent)
  3. Với mỗi câu, query Chroma lấy top-k case tham chiếu gần nhất (RAG)
  4. GHÉP cả 3 nguồn trên thành 1 object, validate theo consistency_input.v1
     -> đây là bước "khớp đầu vào/đầu ra" bạn hỏi: nếu Architecture Extraction
     Agent hoặc Code Analysis Agent đổi field mà không cập nhật schema gốc,
     dòng registry.validate() bên dưới sẽ raise lỗi ngay, không để sai lọt xuống.
  5. (Ở đây thay vì gọi LLM thật) dùng 1 hàm so khớp từ-khoá đơn giản để MÔ PHỎNG
     phán đoán - trong hệ thống thật bước này là 1 lệnh gọi API tới Claude/GPT,
     prompt kèm chính consistency_input đã validate ở bước 4.
  6. Ghi consistency_results.json, validate theo consistency_output.v1

Chạy: python3 agents/consistency_checking_agent.py
"""
import json
import pathlib
import pickle
import sys

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runtime_store" / "runs" / "run_2026-11-14_bigbluebutton"
VECTOR_DIR = ROOT / "reference_store" / "vectorstore"

sys.path.insert(0, str(ROOT))
from agents.schema_registry import SchemaRegistry  # noqa: E402


class LocalTfidfEmbedding(EmbeddingFunction):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def __call__(self, input: Documents) -> Embeddings:
        return self.vectorizer.transform(input).toarray().tolist()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve_reference_cases(collection, sentence_text: str, k: int = 2) -> list[dict]:
    res = collection.query(query_texts=[sentence_text], n_results=k)
    out = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        out.append({
            "case_id": "n/a-from-chroma",
            "project": "bigbluebutton",
            "case_type": meta["case_type"],
            "sentence_id": None,
            "sentence_text": doc,
            "model_element_id": None,
            "model_element_name": meta["model_element_name"],
            "code_path": None,
            "label": 1 if meta["label"] == "1" else (0 if meta["label"] == "0" else "undocumented"),
            "verified": bool(meta["verified"]),
            "source": "ardoco_goldstandard",
        })
    return out


def naive_judge(sentence_text: str, mentioned: list[str], code_elements: list[dict]) -> tuple[str, float, str]:
    """CHỖ NÀY trong hệ thống thật là 1 lệnh gọi LLM (Claude) kèm consistency_input
    đã validate làm prompt. Ở đây dùng so khớp từ khoá đơn giản chỉ để demo luồng
    chạy được đầu-cuối mà không cần gọi API thật."""
    text_lower = sentence_text.lower()
    best = None
    for el in code_elements:
        name_tokens = el["element_name"].lower().replace("-", " ").replace(".", " ").split()
        hit = sum(1 for t in name_tokens if t in text_lower or any(m.lower() in t for m in mentioned))
        path_hit = any(any(m.lower().replace("-", "") in p.lower() for p in el["path"]) for m in mentioned)
        score = hit + (2 if path_hit else 0)
        if best is None or score > best[1]:
            best = (el, score)
    if best is None or best[1] == 0:
        return "no_match", 0.3, "Không tìm thấy phần tử code nào có tên/đường dẫn khớp từ khoá trong câu."
    el, score = best
    confidence = min(0.95, 0.5 + 0.15 * score)
    evidence = f"Tên/đường dẫn code '{el['element_name']}' ({'/'.join(el['path']) or 'no path'}) khớp với từ khoá trong câu."
    label = "match" if confidence >= 0.6 else "no_match"
    return label, round(confidence, 2), evidence


def main():
    registry = SchemaRegistry()

    arch = load_json(RUN_DIR / "architecture_findings.json")
    code = load_json(RUN_DIR / "code_findings.json")
    registry.validate("architecture_finding.v1", arch)
    registry.validate("code_finding.v1", code)
    print("✓ architecture_findings.json khớp schema architecture_finding.v1")
    print("✓ code_findings.json khớp schema code_finding.v1")

    vectorizer = pickle.load(open(VECTOR_DIR / "tfidf_vectorizer.pkl", "rb"))
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.get_collection(
        "reference_cases", embedding_function=LocalTfidfEmbedding(vectorizer)
    )

    retrieved_all = []
    for f in arch["findings"]:
        retrieved_all.extend(retrieve_reference_cases(collection, f["sentence_text"]))

    consistency_input = {
        "run_id": arch["run_id"],
        "project": arch["project"],
        "architecture_findings": arch["findings"],
        "code_elements": code["elements"],
        "retrieved_reference_cases": retrieved_all[:6],
    }
    registry.validate("consistency_input.v1", consistency_input)
    print("✓ consistency_input (ghép từ 2 agent trên + RAG) khớp schema consistency_input.v1")

    with open(RUN_DIR / "consistency_input.json", "w", encoding="utf-8") as f:
        json.dump(consistency_input, f, ensure_ascii=False, indent=2)

    results = []
    for finding in arch["findings"]:
        label, confidence, evidence = naive_judge(
            finding["sentence_text"], finding["mentioned_elements"], code["elements"]
        )
        best_el = None
        text_lower = finding["sentence_text"].lower()
        for el in code["elements"]:
            if any(m.lower() in el["element_name"].lower() or
                   any(m.lower().replace("-", "") in p.lower() for p in el["path"])
                   for m in finding["mentioned_elements"]):
                best_el = el
                break
        results.append({
            "sentence_id": finding["sentence_id"],
            "element_id": best_el["element_id"] if best_el else "NONE",
            "label": label,
            "confidence": confidence,
            "evidence": evidence,
            "verified": False,
        })

    output = {
        "run_id": arch["run_id"],
        "project": arch["project"],
        "produced_by": "consistency_checking_agent",
        "results": results,
    }
    registry.validate("consistency_output.v1", output)
    print("✓ consistency_results.json khớp schema consistency_output.v1")

    with open(RUN_DIR / "consistency_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nĐã ghi {RUN_DIR / 'consistency_input.json'}")
    print(f"Đã ghi {RUN_DIR / 'consistency_results.json'}")


if __name__ == "__main__":
    main()
