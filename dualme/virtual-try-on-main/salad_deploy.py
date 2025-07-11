#!/usr/bin/env python3
"""
DualMe Virtual Try-On - Salad.com Deployment Script
Script automatico per il deployment dell'applicazione su salad.com
"""

import os
import sys
import json
import subprocess
from pathlib import Path


class SaladDeployer:
    def __init__(self):
        self.config = {
            "image_name": "dualme/virtual-try-on",
            "tag": "latest",
            "container_group_name": "dualme-virtual-tryon",
            "min_replicas": 1,
            "max_replicas": 10,
            "gpu_classes": [
                "rtx3080",
                "rtx3090",
                "rtx4080",
                "rtx4090",
                "a4000",
                "a5000",
            ],
            "ports": [7860, 8000],
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "PYTHONUNBUFFERED": "1",
            },
        }

    def check_requirements(self):
        """Verifica i requisiti per il deployment"""
        print("🔍 Checking deployment requirements...")

        # Check Docker
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            print("✅ Docker is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker is not installed or not accessible")
            return False

        # Check Dockerfile
        if not Path("Dockerfile").exists():
            print("❌ Dockerfile not found")
            return False
        print("✅ Dockerfile found")

        # Check app files
        required_files = ["app_gradio.py", "requirements.txt", "download_models.py"]
        for file in required_files:
            if not Path(file).exists():
                print(f"❌ Required file {file} not found")
                return False
        print("✅ All required files present")

        return True

    def build_docker_image(self):
        """Builda l'immagine Docker"""
        print("🏗️ Building Docker image...")

        image_tag = f"{self.config['image_name']}:{self.config['tag']}"

        try:
            subprocess.run(["docker", "build", "-t", image_tag, "."], check=True)
            print(f"✅ Docker image built: {image_tag}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build Docker image: {e}")
            return False

    def generate_salad_config(self):
        """Genera la configurazione per salad.com"""
        print("📋 Generating Salad.com configuration...")

        config = {
            "name": self.config["container_group_name"],
            "image": f"{self.config['image_name']}:{self.config['tag']}",
            "command": [],
            "environment_variables": self.config["environment_variables"],
            "networking": {
                "ports": [
                    {"name": "gradio", "port": 7860, "protocol": "http"},
                    {"name": "health", "port": 8000, "protocol": "http"},
                ]
            },
            "resources": {
                "cpu": 4,
                "memory": 16384,  # 16GB RAM
                "gpu_classes": self.config["gpu_classes"],
                "storage_amount": 50,  # 50GB storage
            },
            "auto_scaling": {
                "enabled": True,
                "min_replicas": self.config["min_replicas"],
                "max_replicas": self.config["max_replicas"],
                "scale_up_threshold": 80,
                "scale_down_threshold": 20,
            },
            "health_check": {
                "path": "/health",
                "port": 8000,
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "failure_threshold": 3,
                "success_threshold": 1,
            },
        }

        # Save to file
        with open("salad_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("✅ Salad.com configuration saved to salad_config.json")
        return config

    def deploy(self, steps=None):
        """Esegue il deployment completo"""
        if steps is None:
            steps = ["check", "build", "config"]

        print("🚀 Starting DualMe Virtual Try-On deployment process...")
        print("=" * 60)

        if "check" in steps:
            if not self.check_requirements():
                print("❌ Requirements check failed")
                return False

        if "build" in steps:
            if not self.build_docker_image():
                print("❌ Docker build failed")
                return False

        if "config" in steps:
            self.generate_salad_config()

        print("\n🎉 Deployment preparation completed!")
        print("\nNext steps:")
        print("1. Review salad_config.json")
        print("2. Push image to registry")
        print("3. Deploy to Salad.com using the configuration")

        return True


def main():
    deployer = SaladDeployer()
    return deployer.deploy()


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)
