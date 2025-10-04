import gradio as gr
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Charger le modèle CodeLLaMA-7B Instruct
model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Fonction pour lire une page web
def lire_page(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text[:2000]  # limite pour éviter d'épuiser la RAM
    except Exception as e:
        return f"Erreur : {e}"

# Fonction principale IA
def ask(question, url="https://www.wikipedia.org"):
    contenu = lire_page(url)
    prompt = f"""Tu es une IA qui peut lire Internet.

Question: {question}
Voici le contenu du site {url} :
{contenu}

Réponds en t'appuyant sur ce texte :
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interface Gradio
iface = gr.Interface(
    fn=ask,
    inputs=[gr.Textbox(label="Ta question"), gr.Textbox(label="Lien du site web")],
    outputs="text",
    title="CodeLLaMA-7B avec Internet"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
