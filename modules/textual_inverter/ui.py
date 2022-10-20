import html

import gradio as gr

from modules.textual_inverter import preprocess, textual_inversion
from modules.diffuser import sd_hijack
from modules.runtime import state


def preprocess(*args):
    preprocess.preprocess(*args)

    return "Preprocessing finished.", ""


def create_embedding(name, initialization_text, nvpt):
    filename = textual_inversion.create_embedding(name, nvpt, init_text=initialization_text)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def train_embedding(*args):

    try:
        sd_hijack.undo_optimizations()

        embedding, filename = textual_inversion.train_embedding(*args)

        res = f"""
Training {'interrupted' if state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()

