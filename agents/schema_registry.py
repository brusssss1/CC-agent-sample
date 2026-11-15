"""
Schema Registry
================
Đây là câu trả lời trực tiếp cho câu hỏi: "làm sao để khi agent nhìn vào hàng loạt
mẫu schema mà nó gọi ra được đúng dataset đó?"

Cơ chế:
1. Mỗi schema có một `$id` duy nhất (VD "architecture_finding.v1").
2. Registry này nạp TẤT CẢ file .schema.json trong thư mục schemas/ vào một dict
   {$id: schema_dict}, và đăng ký chúng vào một `referencing.Registry` để các
   schema có thể $ref lẫn nhau (xem consistency_input.schema.json).
3. Khi 1 record được lưu vào DB/Chroma, nó LUÔN kèm metadata {"schema": "<$id>"}.
4. Khi 1 agent cần dữ liệu, nó không "đoán" - nó gọi
   `get_schema("architecture_finding.v1")` để biết chính xác field nào cần có,
   và khi retrieve từ Chroma nó lọc `where={"schema": "architecture_finding.v1"}`
   để chỉ nhận đúng loại record đó, dù trong cùng 1 collection có nhiều loại khác.

Chạy thử: python3 agents/schema_registry.py
"""
import json
import pathlib

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_DIR = pathlib.Path(__file__).resolve().parent.parent / "schemas"


class SchemaRegistry:
    def __init__(self, schema_dir: pathlib.Path = SCHEMA_DIR):
        self._schemas: dict[str, dict] = {}
        resources = []
        for path in sorted(schema_dir.glob("*.schema.json")):
            schema = json.loads(path.read_text(encoding="utf-8"))
            schema_id = schema["$id"]
            self._schemas[schema_id] = schema
            resources.append((schema_id, Resource.from_contents(schema)))
        self._registry = Registry().with_resources(resources)

    def list_schema_ids(self) -> list[str]:
        """Agent gọi hàm này để biết hiện có những loại dataset nào."""
        return sorted(self._schemas.keys())

    def get_schema(self, schema_id: str) -> dict:
        if schema_id not in self._schemas:
            raise KeyError(
                f"Không có schema '{schema_id}'. Các schema đang có: {self.list_schema_ids()}"
            )
        return self._schemas[schema_id]

    def validate(self, schema_id: str, data: dict) -> None:
        """Ném lỗi rõ ràng nếu `data` không khớp đúng `schema_id`.
        Đây là bước mà Consistency Checking Agent PHẢI chạy trước khi dùng
        output của 2 agent phía trước - để đảm bảo chúng thực sự "khớp"."""
        schema = self.get_schema(schema_id)
        validator = Draft202012Validator(schema, registry=self._registry)
        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if errors:
            messages = [f"  - tại {list(e.path)}: {e.message}" for e in errors]
            raise ValueError(
                f"Dữ liệu KHÔNG khớp schema '{schema_id}':\n" + "\n".join(messages)
            )


if __name__ == "__main__":
    reg = SchemaRegistry()
    print("Các schema đang đăng ký trong hệ thống:")
    for sid in reg.list_schema_ids():
        print(f"  - {sid}: {reg.get_schema(sid)['title']}")
