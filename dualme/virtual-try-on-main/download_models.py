#!/usr/bin/env python3
"""
DualMe Virtual Try-On - Model Downloader
Scarica solo i modelli realmente usati dalla pipeline DualMe VTON.
"""

import os
import sys
from huggingface_hub import snapshot_download
from tqdm import tqdm


def create_directories():
    dirs = ["./ckpts", "./ckpts/u2net", "./ckpts/vton_sd", "./data", "./logs"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Directory creata: {dir_path}")


def download_u2net():
    print("📦 Downloading U2Net for garment segmentation...")
    try:
        snapshot_download(
            repo_id="nathan-gs/u2net",
            local_dir="./ckpts/u2net",
            allow_patterns=["*.pth"],
        )
        print("✅ U2Net scaricato con successo!")
        return True
    except Exception as e:
        print(f"❌ Errore download U2Net: {e}")
        return False


def download_vton_sd():
    print(
        "📦 Downloading DualMe VTON generative model (Stable Diffusion Inpainting custom)..."
    )
    try:
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-2-inpainting",
            local_dir="./ckpts/vton_sd",
            allow_patterns=["*.bin", "*.safetensors", "*.yaml"],
        )
        print("✅ Modello generativo scaricato con successo!")
        return True
    except Exception as e:
        print(f"❌ Errore download modello generativo: {e}")
        return False


def main():
    print("🚀 DualMe VTON - Model Downloader")
    print("=" * 50)
    create_directories()
    ok1 = download_u2net()
    ok2 = download_vton_sd()
    if ok1 and ok2:
        print("\n🎉 Tutti i modelli scaricati con successo!")
        return True
    else:
        print(
            "\n⚠️ Alcuni modelli non sono stati scaricati. Controlla la connessione o scarica manualmente."
        )
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Download interrotto dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore imprevisto: {e}")
        sys.exit(1)
