import io
import base64
from PIL import Image
import torch
import clip

def load_captions(path="captions.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

CAPTIONS = load_captions("captions.txt")


_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _preprocess = clip.load("ViT-B/32", device=_device)
_text_tokens = clip.tokenize(CAPTIONS).to(_device)


@torch.no_grad()
def caption_from_base64(image_base64: str, top_k: int = 3):
    # base64 cleanup (data:image/png;base64,...)
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    image_bytes = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_input = _preprocess(img).unsqueeze(0).to(_device)

    image_features = _model.encode_image(image_input)
    text_features = _model.encode_text(_text_tokens)

    # cosine similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (image_features @ text_features.T).squeeze(0)  # (num_captions,)
    probs = torch.softmax(sims * 100.0, dim=0)            # sharpened softmax

    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(CAPTIONS)))
    results = [
        {"caption": CAPTIONS[i], "confidence": float(p)}
        for p, i in zip(top_probs.tolist(), top_idx.tolist())
    ]

    return results
