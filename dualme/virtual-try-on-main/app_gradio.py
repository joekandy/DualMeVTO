import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import warnings
from vton_model import DualMeVTON, VTONPreprocessor

warnings.filterwarnings("ignore")


# Inizializza pipeline proprietaria DualMe VTON
class DualMeVirtualTryOnApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 DualMe VTON avviato su {self.device}")
        self.preprocessor = VTONPreprocessor(device=self.device)
        self.model = DualMeVTON(device=self.device)

    def virtual_tryon(self, person_img, garment_img):
        if person_img is None or garment_img is None:
            return None, "❌ Carica sia l'immagine della persona che del capo."
        try:
            # Preprocessing reale: segmentazione, rimozione sfondo, keypoints
            person_data = self.preprocessor.process_person(person_img)
            garment_data = self.preprocessor.process_garment(garment_img)
            # Mappatura e deformazione capo sulla posa della persona
            warped_garment = self.preprocessor.warp_garment_to_person(
                garment_data, person_data
            )
            # Generazione immagine finale con modello proprietario
            result_img = self.model.generate(person_data, warped_garment)
            return result_img, "✅ Virtual try-on completato con DualMe VTON"
        except Exception as e:
            print(f"❌ Errore pipeline DualMe VTON: {e}")
            return None, f"❌ Errore: {str(e)}"


# Inizializza app
dualme_app = DualMeVirtualTryOnApp()


def process_virtual_tryon(person_img, garment_img):
    result_img, status_msg = dualme_app.virtual_tryon(person_img, garment_img)
    return result_img, status_msg


# Interfaccia Gradio reale
with gr.Blocks(title="DualMe Virtual Try-On", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style='text-align:center;'>
        <h1>DualMe Virtual Try-On</h1>
        <p>Prova virtuale AI-powered, pipeline reale, senza placeholder</p>
    </div>
    """)
    with gr.Row():
        with gr.Column():
            person_input = gr.Image(
                label="👤 Immagine Persona",
                type="pil",
                height=400,
                sources=["upload", "webcam"],
            )
        with gr.Column():
            garment_input = gr.Image(
                label="👕 Immagine Capo", type="pil", height=400, sources=["upload"]
            )
        with gr.Column():
            result_output = gr.Image(
                label="✨ Risultato Virtual Try-On", type="pil", height=400
            )
    with gr.Row():
        process_btn = gr.Button(
            "🎨 Genera Virtual Try-On", variant="primary", size="lg"
        )
    with gr.Row():
        status_output = gr.Textbox(label="📊 Stato", interactive=False, lines=2)
    process_btn.click(
        fn=process_virtual_tryon,
        inputs=[person_input, garment_input],
        outputs=[result_output, status_output],
        show_progress=True,
    )
    gr.HTML("""
    <div style='text-align:center; margin-top:2rem; opacity:0.7;'>
        <p>Powered by DualMe VTON - Proprietary AI Virtual Try-On</p>
    </div>
    """)

if __name__ == "__main__":
    print("🌐 Avvio server Gradio...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True,
    )
