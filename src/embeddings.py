"""
Wrapper del modelo multimodal CLIP para generar embeddings de texto e imagen
en el mismo espacio vectorial

CLIP trunca internamente el texto a 77 tokens; para documentos de producto
mas largos esto significa que solo se usa el inicio del texto (titulo +
primeras features), lo cual es aceptable para este caso de uso.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src import config


def _as_tensor(output: torch.Tensor) -> torch.Tensor:
    """
    transformers>=5 hace que CLIPModel.get_text_features/get_image_features
    devuelvan un BaseModelOutputWithPooling (con el vector proyectado en
    .pooler_output) en vez de un tensor plano; en versiones anteriores
    devuelven directamente el tensor. Soporta ambos casos.
    """
    return output.pooler_output if hasattr(output, "pooler_output") else output


class ClipEmbedder:
    def __init__(self, model_name: str = config.CLIP_MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            feats = _as_tensor(self.model.get_text_features(**inputs))
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy())
        return np.concatenate(all_embeds, axis=0).astype("float32")

    @torch.no_grad()
    def encode_images(self, image_paths: list[str], batch_size: int = 16) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            feats = _as_tensor(self.model.get_image_features(**inputs))
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy())
        return np.concatenate(all_embeds, axis=0).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """Embedding de una consulta de texto (misma proyeccion que encode_texts)."""
        return self.encode_texts([query])[0]

    def encode_multimodal_document(self, text: str, image_path: str | None,
                                    text_weight: float = 0.5) -> np.ndarray:
        """
        Combina el embedding de texto y de imagen de un producto en un unico
        vector (promedio ponderado + renormalizado), para indexar un solo
        vector multimodal por documento.
        """
        text_emb = self.encode_texts([text])[0]
        if image_path is None:
            return text_emb
        image_emb = self.encode_images([image_path])[0]
        combined = text_weight * text_emb + (1 - text_weight) * image_emb
        return combined / np.linalg.norm(combined)


if __name__ == "__main__":
    embedder = ClipEmbedder()
    v = embedder.encode_query("wireless bluetooth headphones")
    print("Dim embedding CLIP:", v.shape, "norma:", np.linalg.norm(v))
