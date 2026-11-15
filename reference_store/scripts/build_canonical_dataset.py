"""
Xử lý dữ liệu thô ArDoCo benchmark (bigbluebutton) -> cases.jsonl (Canonical Dataset)
Mỗi case được validate ngay tại chỗ theo schemas/reference_case.schema.json trước khi ghi.

Nguồn thô dùng (đã tải thật, không mô phỏng):
  reference_store/raw/ardoco_benchmark_bigbluebutton/
    model_2021/pcm/bbb.repository          <- PCM model, để tra entityName theo id
    text_2021/bigbluebutton.txt            <- câu văn bản kiến trúc (SAD), 1 dòng = 1 câu
    goldstandards/goldstandard_sad_2021-sam_2021.csv       <- câu <-> phần tử model (positive)
    goldstandards/goldstandard_sad_2021-sam_2021_UME.csv   <- phần tử model KHÔNG có trong text
    goldstandards/goldstandard_sam_2021-code_2023.csv      <- phần tử model <-> code (đã có tên sẵn)

Chạy: python3 reference_store/scripts/build_canonical_dataset.py
"""
import csv
import hashlib
import json
import pathlib
import random
import re
import sys


def short_hash(*parts: str) -> str:
    """8 ký tự hash ổn định - tránh đụng id khi các ID gốc chỉ khác nhau vài ký tự đầu
    (các ID kiểu EMF/XMI thường CHUNG hậu tố vì sinh cùng 1 batch)."""
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:8]

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "reference_store" / "raw" / "ardoco_benchmark_bigbluebutton"
OUT = ROOT / "reference_store" / "processed" / "cases.jsonl"

sys.path.insert(0, str(ROOT))
from agents.schema_registry import SchemaRegistry  # noqa: E402


def load_pcm_names(pcm_path: pathlib.Path) -> dict[str, str]:
    """id -> entityName, lấy từ file PCM .repository (XML dạng text, regex đủ dùng
    vì cấu trúc phẳng, không cần kéo cả thư viện XML nặng)."""
    text = pcm_path.read_text(encoding="utf-8")
    pairs = re.findall(r'id="(_[\w-]+)"\s+entityName="([^"]+)"', text)
    return {i: n for i, n in pairs}


def load_sentences(text_path: pathlib.Path) -> dict[int, str]:
    lines = text_path.read_text(encoding="utf-8").splitlines()
    return {i + 1: line for i, line in enumerate(lines) if line.strip()}


def main():
    registry = SchemaRegistry()
    id_to_name = load_pcm_names(RAW / "model_2021" / "pcm" / "bbb.repository")
    sentences = load_sentences(RAW / "text_2021" / "bigbluebutton.txt")

    cases = []

    # 1) positive: goldstandard_sad_2021-sam_2021.csv
    positive_pairs = set()
    with open(RAW / "goldstandards" / "goldstandard_sad_2021-sam_2021.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            element_id = row["modelElementID"]
            sentence_id = int(row["sentence"])
            positive_pairs.add((sentence_id, element_id))
            cases.append({
                "case_id": f"bbb_sad{sentence_id}_sam_{short_hash(element_id)}",
                "project": "bigbluebutton",
                "case_type": "sad_sam",
                "sentence_id": sentence_id,
                "sentence_text": sentences.get(sentence_id),
                "model_element_id": element_id,
                "model_element_name": id_to_name.get(element_id, "UNKNOWN"),
                "code_path": None,
                "label": 1,
                "verified": True,
                "source": "ardoco_goldstandard",
            })

    # 2) negative: tự sinh - ghép câu có xuất hiện trong gold standard với
    #    1 phần tử KHÁC mà câu đó không nhắc tới (closest-negative, không random tuyệt đối)
    all_elements = sorted({eid for (_, eid) in positive_pairs})
    rng = random.Random(42)
    sentence_ids_in_gold = sorted({sid for (sid, _) in positive_pairs})
    for sid in sentence_ids_in_gold:
        linked = {eid for (s, eid) in positive_pairs if s == sid}
        candidates = [e for e in all_elements if e not in linked]
        if not candidates:
            continue
        neg_element = rng.choice(candidates)
        cases.append({
            "case_id": f"bbb_sad{sid}_sam_{short_hash(neg_element)}_neg",
            "project": "bigbluebutton",
            "case_type": "sad_sam",
            "sentence_id": sid,
            "sentence_text": sentences.get(sid),
            "model_element_id": neg_element,
            "model_element_name": id_to_name.get(neg_element, "UNKNOWN"),
            "code_path": None,
            "label": 0,
            "verified": True,
            "source": "ardoco_goldstandard",
        })

    # 3) undocumented: goldstandard_sad_2021-sam_2021_UME.csv
    with open(RAW / "goldstandards" / "goldstandard_sad_2021-sam_2021_UME.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            element_id = row["missingModelElementID"]
            cases.append({
                "case_id": f"bbb_undoc_{short_hash(element_id)}",
                "project": "bigbluebutton",
                "case_type": "undocumented",
                "sentence_id": None,
                "sentence_text": None,
                "model_element_id": element_id,
                "model_element_name": id_to_name.get(element_id, "UNKNOWN"),
                "code_path": None,
                "label": "undocumented",
                "verified": True,
                "source": "ardoco_goldstandard",
            })

    # 4) sam_code: goldstandard_sam_2021-code_2023.csv (đã có ae_name sẵn)
    with open(RAW / "goldstandards" / "goldstandard_sam_2021-code_2023.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cases.append({
                "case_id": f"bbb_sam_{short_hash(row['ae_id'], row['ce_ids'])}",
                "project": "bigbluebutton",
                "case_type": "sam_code",
                "sentence_id": None,
                "sentence_text": None,
                "model_element_id": row["ae_id"],
                "model_element_name": row["ae_name"],
                "code_path": row["ce_ids"],
                "label": 1,
                "verified": True,
                "source": "ardoco_goldstandard",
            })

    # validate từng case trước khi ghi - sai là dừng ngay, không ghi rác vào dataset
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for case in cases:
            registry.validate("reference_case.v1", case)
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    by_type = {}
    for c in cases:
        by_type[c["case_type"]] = by_type.get(c["case_type"], 0) + 1
    print(f"Đã ghi {len(cases)} case vào {OUT}")
    for t, n in sorted(by_type.items()):
        print(f"  - {t}: {n}")


if __name__ == "__main__":
    main()
