# utils/k8s_config_generator.py
from app.models.schemas import ModelDeploymentConfig
import yaml



def generate_deployment_manifests(config: ModelDeploymentConfig) -> dict:
    """Generate Kubernetes deployment manifests from model config."""
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"model-{config.version}"},
        "spec": {
            "replicas": config.replicas,
            "selector": {"matchLabels": {"app": "membership-model", "version": config.version}},
            "template": {
                "metadata": {"labels": {"app": "membership-model", "version": config.version}},
                "spec": {
                    "containers": [{
                        "name": "model-api",
                        "image": f"${{DOCKER_REGISTRY}}/grocery-membership:{config.version}",
                        "resources": config.resources,
                        "env": [
                            {"name": "MODEL_PATH", "value": config.model_path},
                            {"name": "ENVIRONMENT", "value": config.environment}
                        ]
                    }]
                }
            }
        }
    }

    return deployment