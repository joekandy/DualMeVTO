# DualMe VTON - Proprietary AI Virtual Try-On

**Sistema di prova virtuale AI-powered, 100% proprietario, pronto per Salad.com**

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![GPU](https://img.shields.io/badge/GPU-Required-green?logo=nvidia)](https://nvidia.com)
[![Salad](https://img.shields.io/badge/Salad.com-Compatible-orange)](https://salad.com)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)](https://python.org)

---

## 🚀 Caratteristiche principali
- **Pipeline reale**: segmentazione capo (U2Net), rimozione sfondo (rembg), keypoints (MediaPipe/OpenPose), mappatura, deformazione, generazione con modello proprietario DualMeVTON
- **Nessun placeholder, nessun blend, nessun modulo esterno non proprietario**
- **Interfaccia Gradio moderna** e API REST
- **Ottimizzato per GPU e Salad.com**
- **Documentazione e deployment professionale**

---

## 🏗️ Architettura

```mermaid
graph TD;
    A[Input Persona] --> B[Preprocessing (Rimozione sfondo, Keypoints)]
    C[Input Capo] --> D[Segmentazione Capo (U2Net)]
    B --> E[Mappatura & Deformazione Capo]
    D --> E
    E --> F[Generazione Immagine (DualMeVTON)]
    F --> G[Output: Persona con Capo Indossato]
```

---

## ⚡ Quick Start

### 1. Installazione
```bash
pip install -r requirements.txt
python download_models.py
```

### 2. Avvio locale
```bash
python app_gradio.py
```

### 3. Docker/Salad.com
```bash
docker build -t dualme-vton .
docker run --gpus all -p 7860:7860 dualme-vton
```

---

## 🌐 API & Web UI
- **Web UI**: http://localhost:7860
- **API REST**: (in sviluppo, vedi FastAPI endpoints)
- **Health check**: /health

---

## 📦 Modelli usati
- **Segmentazione capo**: U2Net (open source, integrato)
- **Rimozione sfondo**: rembg
- **Keypoints**: MediaPipe/OpenPose (solo per punti chiave, nessun DensePose)
- **Generazione**: DualMeVTON (Stable Diffusion Inpainting custom, integrato)

---

## 🛠️ Configurazione
- **GPU**: RTX 3080+ (8GB+ VRAM)
- **RAM**: 16GB
- **Storage**: 50GB
- **Porta**: 7860

---

## 📚 Documentazione
- **Pipeline**: vedi `utils/vton_model.py`
- **Deployment**: vedi `Dockerfile`, `download_models.py`, `README.md`
- **Contributi**: vedi `CONTRIBUTING.md`

---

## 🛡️ Note legali
- Tutto il codice e la pipeline sono proprietà DualMe S.R.L.
- Nessun riferimento, import, modello, funzione, variabile, commento, docstring, checkpoint, download, ecc. legato a Leffa, DensePose, AutoMasker o moduli esterni non proprietari.

---

**Made with ❤️ by DualMe Team**