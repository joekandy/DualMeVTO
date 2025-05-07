import gradio as gr
from PIL import Image
import torch

# Modello dummy: sostituisci con il tuo modello reale
class DummyModel(torch.nn.Module):
    def forward(self, person_img, cloth_img):
        # Semplice overlay per demo
        return Image.blend(person_img, cloth_img, alpha=0.5)

model = DummyModel()

def virtual_tryon(person, cloth):
    # Qui puoi aggiungere preprocessing o chiamare il tuo modello reale
    output = model(person, cloth)
    return output

demo = gr.Interface(
    fn=virtual_tryon,
    inputs=[
        gr.Image(type="pil", label="Immagine Persona"),
        gr.Image(type="pil", label="Immagine Abito")
    ],
    outputs=gr.Image(type="pil", label="Risultato Try-On"),
    title="DualMe Virtual Try-On API",
    description="Carica una foto di una persona e un capo d'abbigliamento per vedere il risultato del try-on virtuale. Funziona anche come API REST!"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 