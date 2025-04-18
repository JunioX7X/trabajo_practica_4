# model_registry/versioning.py
import os
import json
import datetime
from typing import Dict, Any


class ModelRegistry:
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.metadata_path = os.path.join(registry_dir, "metadata.json")
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"models": []}

    def register_model(self, model_path: str, metrics: Dict[str, Any],
                       tags: Dict[str, str] = None):
        model_id = f"model_{len(self.metadata['models']) + 1}"
        model_info = {
            "id": model_id,
            "path": model_path,
            "metrics": metrics,
            "tags": tags or {},
            "created_at": datetime.datetime.now().isoformat(),
            "status": "registered"
        }
        self.metadata["models"].append(model_info)
        self._save_metadata()
        return model_id

    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)