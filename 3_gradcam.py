import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json

# --- Configuracao -------------------------------------------------------------
DATA_DIR    = Path("data/lfw")
OUTPUT_DIR  = Path("outputs")
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A usar dispositivo: {device}")

# --- Carregar modelos ---------------------------------------------------------
mtcnn   = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# --- Carregar dados guardados pelo 2_model.py ---------------------------------
landmarks    = np.load(OUTPUT_DIR / "landmarks.npy",    allow_pickle=True).item()
embeddings   = np.load(OUTPUT_DIR / "embeddings.npy",   allow_pickle=True).item()
pair_details = np.load(OUTPUT_DIR / "pair_details.npy", allow_pickle=True).tolist()
results      = json.load(open(OUTPUT_DIR / "model_results.json"))
THRESHOLD    = results["best_threshold"]

# --- Grad-CAM -----------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, face_tensor, ref_embedding=None):
        """
        Gera mapa Grad-CAM normalizado [0,1] com shape (160, 160).

        CORRECOES aplicadas vs versao anterior:
        1. Se ref_embedding for fornecido, usa similaridade coseno como sinal
           em vez da norma L2 -- o mapa fica orientado a decisao do par,
           produzindo activacoes muito mais focadas e coloridas.
        2. Normalizacao por percentil (p1-p99) em vez de min-max -- evita que
           um unico pixel outlier "esmague" todo o mapa deixando-o azulado.
        3. .detach() antes de .cpu().numpy() para evitar o RuntimeError de grad.
        """
        self.model.zero_grad()
        face_tensor = face_tensor.to(device).requires_grad_(True)

        embedding = self.model(face_tensor)   # (1, 512)

        if ref_embedding is not None:
            score = F.cosine_similarity(embedding, ref_embedding.to(device))
        else:
            score = embedding.norm(p=2, dim=1)

        score.backward()

        grads = self.gradients
        acts  = self.activations

        if grads is None or acts is None:
            return np.zeros((160, 160), dtype=np.float32)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * acts).sum(dim=1).squeeze()
        cam     = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_np = cv2.resize(cam_np, (160, 160))

        # Normalizacao robusta por percentil
        p1, p99 = np.percentile(cam_np, 1), np.percentile(cam_np, 99)
        if p99 - p1 > 1e-8:
            cam_np = np.clip((cam_np - p1) / (p99 - p1), 0, 1)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


# Tentar repeat_3[-1] (mais rico em features espaciais); fallback para block8
try:
    target_layer = facenet.repeat_3[-1]
    print("Camada alvo: repeat_3[-1]")
except (AttributeError, IndexError):
    target_layer = facenet.block8
    print("Camada alvo: block8 (fallback)")

gradcam = GradCAM(facenet, target_layer)

# --- Funcoes auxiliares -------------------------------------------------------

def get_face_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    t   = mtcnn(img)
    return t.unsqueeze(0) if t is not None else None


def landmarks_to_boxes(lm, img_w=160, img_h=160):
    le = np.array(lm["left_eye"])
    re = np.array(lm["right_eye"])
    no = np.array(lm["nose"])
    ml = np.array(lm["mouth_left"])
    mr = np.array(lm["mouth_right"])

    iod = np.linalg.norm(re - le)
    pad = iod * 0.35

    def box(cx, cy, half_w, half_h):
        x1 = max(0,     int(cx - half_w))
        y1 = max(0,     int(cy - half_h))
        x2 = min(img_w, int(cx + half_w))
        y2 = min(img_h, int(cy + half_h))
        return (x1, y1, x2, y2)

    mid_mouth = (ml + mr) / 2
    return {
        "olho_esq": box(le[0], le[1], pad * 1.1, pad * 0.8),
        "olho_dir": box(re[0], re[1], pad * 1.1, pad * 0.8),
        "nariz":    box(no[0], no[1], pad * 0.9, pad * 1.0),
        "boca":     box(mid_mouth[0], mid_mouth[1], iod * 0.45, pad * 0.9),
        "testa":    box((le[0]+re[0])/2, le[1] - iod * 0.6, iod * 0.7, pad * 0.8),
    }


def overlay_heatmap(face_rgb, cam, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * face_rgb.astype(float) + alpha * heatmap.astype(float)
    return np.uint8(np.clip(blended, 0, 255))


def remap_landmarks(lm_orig, img_orig_size, crop_size=160, margin=20):
    pts    = np.array(list(lm_orig.values()))
    cx, cy = pts.mean(axis=0)
    iod_orig  = np.linalg.norm(
        np.array(lm_orig["left_eye"]) - np.array(lm_orig["right_eye"])
    )
    crop_orig = iod_orig * 2.5 + 2 * margin
    scale     = crop_size / crop_orig
    remapped  = {}
    for k, (x, y) in lm_orig.items():
        remapped[k] = [(x - cx + crop_orig / 2) * scale,
                       (y - cy + crop_orig / 2) * scale]
    return remapped


def get_ref_embedding(key, pair_details):
    """Devolve o embedding do par desta imagem como tensor, ou None."""
    for d in pair_details:
        if d["img1"] == key:
            partner = d["img2"]
        elif d["img2"] == key:
            partner = d["img1"]
        else:
            continue
        if partner in embeddings:
            return torch.tensor(embeddings[partner]).unsqueeze(0)
    return None


def visualize_gradcam(img_path, save_path, pair_score=None, pair_label=None):
    key = f"{img_path.parent.name}/{img_path.name}"

    face_tensor = get_face_tensor(img_path)
    if face_tensor is None:
        print(f"  Sem face: {key}")
        return

    # Grad-CAM orientado ao par (se disponivel)
    ref_emb = get_ref_embedding(key, pair_details)
    cam     = gradcam(face_tensor, ref_embedding=ref_emb)

    # Face como array RGB
    face_np = face_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    face_np = ((face_np - face_np.min()) /
               (face_np.max() - face_np.min()) * 255).astype(np.uint8)

    if key in landmarks:
        img_orig_size = Image.open(img_path).size
        lm_remapped   = remap_landmarks(landmarks[key], img_orig_size)
        boxes         = landmarks_to_boxes(lm_remapped)
    else:
        lm_remapped = None
        boxes       = {}

    heatmap_overlay = overlay_heatmap(face_np, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    region_colors = {
        "olho_esq": "cyan", "olho_dir": "cyan",
        "nariz": "yellow", "boca": "lime", "testa": "magenta",
    }

    # Painel 1 -- face + regioes
    axes[0].imshow(face_np)
    axes[0].set_title("Face + Regioes", fontsize=10)
    if lm_remapped:
        for pt_name, (x, y) in lm_remapped.items():
            axes[0].plot(x, y, "ro", markersize=4)
    for region, (x1, y1, x2, y2) in boxes.items():
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5, edgecolor=region_colors[region], facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-2, region, color=region_colors[region],
                     fontsize=6, va="bottom")
    axes[0].axis("off")

    # Painel 2 -- Grad-CAM
    axes[1].imshow(heatmap_overlay)
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[1].axis("off")

    # Painel 3 -- Grad-CAM + regioes
    axes[2].imshow(heatmap_overlay)
    for region, (x1, y1, x2, y2) in boxes.items():
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5, edgecolor=region_colors[region], facecolor="none"
        )
        axes[2].add_patch(rect)
    axes[2].set_title("Grad-CAM + Regioes", fontsize=10)
    axes[2].axis("off")

    title = key
    if pair_score is not None:
        decision = "Correto" if (
            (pair_score >= THRESHOLD and pair_label == 1) or
            (pair_score <  THRESHOLD and pair_label == 0)
        ) else "Erro"
        title += f"  |  score={pair_score:.3f}  |  {decision}"
    fig.suptitle(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# --- Selecionar imagens representativas ---------------------------------------
categories = {
    "true_positive":  [],
    "true_negative":  [],
    "false_negative": [],
    "false_positive": [],
}

for d in pair_details:
    s, lbl = d["score"], d["label"]
    pred   = 1 if s >= THRESHOLD else 0
    if   lbl == 1 and pred == 1: categories["true_positive"].append(d)
    elif lbl == 0 and pred == 0: categories["true_negative"].append(d)
    elif lbl == 1 and pred == 0: categories["false_negative"].append(d)
    elif lbl == 0 and pred == 1: categories["false_positive"].append(d)

N = 5
print("\nDistribuicao dos pares:")
for cat, items in categories.items():
    print(f"  {cat}: {len(items)}")

processed = set()

for cat, items in categories.items():
    cat_dir = GRADCAM_DIR / cat
    cat_dir.mkdir(exist_ok=True)

    for d in items[:N]:
        for key in (d["img1"], d["img2"]):
            if key in processed:
                continue
            processed.add(key)

            person, fname = key.split("/")
            img_path  = DATA_DIR / person / fname
            save_name = f"{person}_{fname.replace('.jpg','')}.png"
            save_path = cat_dir / save_name

            print(f"  A gerar Grad-CAM: {key}  [{cat}]")
            visualize_gradcam(img_path, save_path,
                              pair_score=d["score"], pair_label=d["label"])

print(f"\nGrad-CAMs guardados em {GRADCAM_DIR}/")
print("  true_positive/   -- mesma pessoa, bem reconhecida")
print("  true_negative/   -- pessoas diferentes, bem separadas")
print("  false_negative/  -- mesma pessoa NAO reconhecida  <- casos de falha")
print("  false_positive/  -- pessoas diferentes confundidas <- casos de falha")