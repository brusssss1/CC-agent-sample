"""
Xuất cases.jsonl (cùng 1 Canonical Dataset đã dùng cho RAG) sang định dạng
hội thoại chuẩn cho fine-tuning (kiểu Together AI / OpenAI), chia train/val/test.
Chỉ lấy case đã verified=True và có sentence_text (loại sam_code vì không phải
câu hỏi dạng "câu <-> phần tử").

Chạy: python3 reference_store/scripts/export_finetune.py
"""
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[2]
CASES = ROOT / "reference_store" / "processed" / "cases.jsonl"
EXPORT_DIR = ROOT / "reference_store" / "exports"


def to_chat_format(case: dict) -> dict:
    if case["case_type"] == "sad_sam":
        user = (f"Câu: \"{case['sentence_text']}\"\n"
                f"Phần tử kiến trúc: \"{case['model_element_name']}\"\n"
                f"Đây có phải liên kết đúng không?")
        answer = "Đúng (match)." if case["label"] == 1 else "Sai (không liên quan)."
    else:  # undocumented
        user = f"Phần tử kiến trúc \"{case['model_element_name']}\" có được mô tả trong tài liệu không?"
        answer = "Không - đây là phần tử chưa được tài liệu hoá (undocumented)."
    return {"messages": [{"role": "user", "content": user},
                          {"role": "assistant", "content": answer}]}


def main():
    cases = [json.loads(l) for l in open(CASES, encoding="utf-8")
              if json.loads(l)["case_type"] in ("sad_sam", "undocumented") and json.loads(l)["verified"]]
    random.Random(42).shuffle(cases)

    n = len(cases)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    splits = {
        "train": cases[:n_train],
        "val": cases[n_train:n_train + n_val],
        "test": cases[n_train + n_val:],
    }

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        path = EXPORT_DIR / f"finetune_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for c in subset:
                f.write(json.dumps(to_chat_format(c), ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)} dòng -> {path}")


if __name__ == "__main__":
    main()
