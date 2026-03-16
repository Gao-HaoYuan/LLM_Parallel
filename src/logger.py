import json
import os
import time


class JsonlLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, payload: dict):
        payload = dict(payload)
        payload["time"] = time.time()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")