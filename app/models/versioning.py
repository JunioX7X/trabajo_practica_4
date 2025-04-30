import os
import json
import datetime
from typing import Dict, Any, List, Optional


class ModelRegistry:
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.metadata_path = os.path.join(registry_dir, "metadata.json")
        self._load_metadata()

    def get_model_version(self, model_id: str) -> Optional[str]:
        model = self.get_model_by_id(model_id)
        if model:
            return model.get("version")
        return None


    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"models": []}

    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(self, model_path: str, metrics: Dict[str, Any],
                       version: str = None, tags: Dict[str, str] = None,
                       status: str = "registered") -> str:
        model_id = f"model_{len(self.metadata['models']) + 1}"
        version = version or f"1.0.{len(self.metadata['models'])}"
        model_info = {
            "id": model_id,
            "path": model_path,
            "metrics": metrics,
            "version": version,
            "tags": tags or {},
            "created_at": datetime.datetime.now().isoformat(),
            "status": status
        }
        self.metadata["models"].append(model_info)
        self._save_metadata()
        return model_id

    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        for model in self.metadata["models"]:
            if model["id"] == model_id:
                return model
        return None

    def list_models(self, status: str = None) -> List[Dict[str, Any]]:
        if status:
            return [m for m in self.metadata["models"] if m["status"] == status]
        return self.metadata["models"]

    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        if not self.metadata["models"]:
            return None
        return self.metadata["models"][-1]

    def set_model_status(self, model_id: str, status: str) -> bool:
        model = self.get_model_by_id(model_id)
        if model:
            model["status"] = status
            self._save_metadata()
            return True
        return False

    def rollback_model(self, target_model_id: str) -> bool:
        model = self.get_model_by_id(target_model_id)
        if model:
            for m in self.metadata["models"]:
                if m["status"] == "production":
                    m["status"] = "archived"
            model["status"] = "production"
            self._save_metadata()
            return True
        return False
if __name__ == "__main__":
    registry = ModelRegistry()
    model_id = "membership_model.joblib"
    version = registry.get_model_version(model_id)

    if version:
        print(f"La versión del modelo '{model_id}' es: {version}")
    else:
        print("Modelo no encontrado.")
