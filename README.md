# Dự án mẫu: Dataset + Schema + Agent cho hệ Consistency Checking (BigBlueButton)

Đây là bản chạy **thật**, không mô phỏng: toàn bộ dữ liệu lấy từ
`github.com/ardoco/benchmark` (thư mục `bigbluebutton`), xử lý thành dataset,
nạp vào SQLite + Chroma, rồi chạy thử một lượt Consistency Checking Agent
end-to-end, có validate schema ở từng bước.

## Cây thư mục

```
ardoco-consistency-system/
├── schemas/                              ← "file schema" định nghĩa mọi loại dữ liệu
│   ├── reference_case.schema.json         (1 case trong Reference Store)
│   ├── architecture_finding.schema.json   (output Architecture Extraction Agent)
│   ├── code_finding.schema.json           (output Code Analysis Agent)
│   ├── consistency_input.schema.json      (input Consistency Checking Agent -
│   │                                        $ref THẲNG tới 2 schema output trên)
│   └── consistency_output.schema.json     (output Consistency Checking Agent)
│
├── agents/
│   ├── schema_registry.py                 ← nạp + validate theo $id, agent "tra cứu" ở đây
│   └── consistency_checking_agent.py      ← chạy end-to-end: ghép input, RAG, validate, output
│
├── reference_store/                       ← kho tham chiếu ổn định (KHÔNG đổi mỗi lần quét)
│   ├── raw/ardoco_benchmark_bigbluebutton/   (dữ liệu gốc đã tải thật từ GitHub)
│   ├── processed/cases.jsonl                 (Canonical Dataset: 175 case)
│   ├── db/cases.db                           (SQLite - "sổ sách")
│   ├── vectorstore/                          (Chroma - cho RAG)
│   ├── exports/finetune_{train,val,test}.jsonl  (cùng dataset, xuất dạng fine-tune)
│   └── scripts/
│       ├── build_canonical_dataset.py     (CSV gốc -> cases.jsonl, có validate)
│       ├── load_stores.py                 (cases.jsonl -> SQLite + Chroma)
│       └── export_finetune.py             (cases.jsonl -> JSONL fine-tuning)
│
└── runtime_store/                         ← trạng thái CỦA MỘT LẦN QUÉT (đúng sơ đồ LangGraph)
    ├── config/scan_strategy.json
    └── runs/run_2026-11-14_bigbluebutton/
        ├── architecture_findings.json     (do Architecture Extraction Agent ghi)
        ├── code_findings.json             (do Code Analysis Agent ghi)
        ├── consistency_input.json         (bản GHÉP 2 file trên + RAG - tự sinh)
        └── consistency_results.json       (output cuối, Coordinator đọc tiếp)
```

## Chạy lại từ đầu

```bash
cd reference_store/scripts
python3 build_canonical_dataset.py   # CSV gốc -> cases.jsonl (175 case, có validate)
python3 load_stores.py               # -> SQLite + Chroma
python3 export_finetune.py           # -> 3 file JSONL cho fine-tuning

cd ../../
python3 agents/consistency_checking_agent.py   # chạy 1 lượt quét demo, in ra 4 dấu ✓ validate
```

## Trả lời đúng câu hỏi cốt lõi: làm sao để input/output 2 agent "khớp" nhau

Mở `schemas/consistency_input.schema.json` — field `architecture_findings` và
`code_elements` không tự định nghĩa lại cấu trúc, mà **`$ref` thẳng** tới
`architecture_finding.v1#/properties/findings` và `code_finding.v1#/properties/elements`.

Kết quả: nếu một ngày bạn sửa output của Architecture Extraction Agent (thêm/bớt
field) mà quên cập nhật schema gốc, dòng `registry.validate(...)` trong
`consistency_checking_agent.py` sẽ **raise lỗi ngay lập tức**, chỉ rõ field nào sai
ở đâu — không phải đợi chạy sai rồi mới phát hiện.

## Một lưu ý thành thật về bản demo này

- **Embedding**: sandbox này không có mạng ra ngoài tới Hugging Face/OpenAI nên
  không tải được embedding model thật. Tôi thay bằng TF-IDF chạy local
  (`load_stores.py`, class `LocalTfidfEmbedding`). Khi triển khai thật, chỉ cần
  đổi class này sang gọi Voyage/OpenAI embedding API — mọi chỗ khác giữ nguyên.
- **Phán đoán của Consistency Checking Agent**: hàm `naive_judge()` trong
  `consistency_checking_agent.py` chỉ so khớp từ khoá đơn giản để demo luồng
  chạy được đầu-cuối mà không cần gọi API LLM thật (cần API key). Trong hệ thống
  thật, bước này là **một lệnh gọi Claude/GPT**, nhận đúng `consistency_input.json`
  (đã validate) làm prompt, trả về đúng cấu trúc `consistency_output.v1`.
- Kết quả demo cho câu 61–63 ("FreeSWITCH...") khớp đúng với gold standard thật
  (`goldstandard_sam_2021-code_2023.csv`), nhưng câu 67/70 (Kurento/WebRTC-SFU)
  bị match nhầm vào 1 file placeholder rỗng — đây là ví dụ thật cho thấy vì sao
  luôn cần ngưỡng confidence + người review trước khi tin, như đã bàn ở các lượt trước.
