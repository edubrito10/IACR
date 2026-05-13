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

# ─── Configuração ─────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/lfw")
OUTPUT_DIR = Path("outputs")
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A usar dispositivo: {device}")

# ─── Carregar modelos ─────────────────────────────────────────────────────────
mtcnn   = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ─── Carregar dados guardados pelo 2_model.py ─────────────────────────────────
landmarks  = np.load(OUTPUT_DIR / "landmarks.npy",  allow_pickle=True).item()
pair_details = np.load(OUTPUT_DIR / "pair_details.npy", allow_pickle=True).tolist()
results    = json.load(open(OUTPUT_DIR / "model_results.json"))
THRESHOLD  = results["best_threshold"]

# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
# O FaceNet (InceptionResnetV1) tem um bloco final chamado 'block8'.
# Registamos hooks nessa camada para capturar gradientes e ativações.

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model    = model
        self.gradients = None
        self.activations = None

        # Forward hook → guarda ativações
        target_layer.register_forward_hook(self._save_activation)
        # Backward hook → guarda gradientes
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Dado um tensor de face [1, 3, 160, 160], devolve o mapa Grad-CAM
        normalizado em [0,1] com shape (160, 160).

        Estratégia: usamos a norma L2 do embedding como sinal de score
        (maximiza a "confiança" da representação aprendida).
        """
        self.model.zero_grad()
        face_tensor = face_tensor.to(device).requires_grad_(True)

        embedding = self.model(face_tensor)          # (1, 512)
        score     = embedding.norm(p=2, dim=1)       # escalar — norma do embedding
        score.backward()

        # Gradientes: (1, C, H, W) → pesos por canal
        grads = self.gradients                       # (1, C, H, W)
        acts  = self.activations                     # (1, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * acts).sum(dim=1).squeeze()  # (H, W)
        cam     = F.relu(cam)

        # Redimensionar para 160×160 e normalizar
        cam_np  = cam.cpu().numpy()
        cam_np  = cv2.resize(cam_np, (160, 160))
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)

        return cam_np


# Camada alvo: último bloco residual antes do pooling global
target_layer = facenet.block8
gradcam      = GradCAM(facenet, target_layer)

# ─── Funções auxiliares ───────────────────────────────────────────────────────

def get_face_tensor(img_path: Path):
    """Devolve o tensor da face recortada [1,3,160,160] ou None."""
    img = Image.open(img_path).convert("RGB")
    t   = mtcnn(img)
    return t.unsqueeze(0) if t is not None else None


def landmarks_to_boxes(lm: dict, img_w=160, img_h=160) -> dict:
    """
    Converte os 5 landmarks faciais em bounding boxes aproximadas
    para 5 regiões semânticas, em coordenadas da face recortada (160×160).

    Como o MTCNN devolve landmarks na imagem original e depois faz crop+resize,
    precisamos de re-mapear. Usamos offsets heurísticos baseados na estrutura
    anatómica típica de uma face detetada pelo MTCNN com margin=20.

    Regiões:
        olho_esq, olho_dir  → elipse em torno de cada olho
        nariz               → área central abaixo dos olhos
        boca                → área abaixo do nariz
        testa               → área acima dos olhos
    """
    le = np.array(lm["left_eye"])
    re = np.array(lm["right_eye"])
    no = np.array(lm["nose"])
    ml = np.array(lm["mouth_left"])
    mr = np.array(lm["mouth_right"])

    # Distância interocular — usada como unidade de escala
    iod = np.linalg.norm(re - le)
    pad = iod * 0.35   # padding em torno de cada ponto

    def box(cx, cy, half_w, half_h):
        """Bounding box clampada ao tamanho da imagem."""
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


def overlay_heatmap(face_rgb: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    """Sobrepõe o mapa de calor (colormap JET) na face original."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * face_rgb.astype(float) + alpha * heatmap.astype(float)
    return np.uint8(np.clip(blended, 0, 255))


def remap_landmarks(lm_orig: dict, img_orig_size, crop_size=160, margin=20) -> dict:
    """
    O MTCNN deteta landmarks na imagem original e depois recorta+redimensiona
    a face para crop_size×crop_size. Este método aproxima as coordenadas
    remapeadas para o espaço da face recortada.

    Nota: sem acesso à bounding box exata do MTCNN esta é uma aproximação
    heurística. Para maior precisão usa mtcnn.detect() + crop manual.
    """
    W, H = img_orig_size

    # Centro aproximado da face (média dos 5 pontos)
    pts = np.array(list(lm_orig.values()))
    cx, cy = pts.mean(axis=0)

    # Tamanho do crop original estimado a partir da distância interocular
    iod_orig = np.linalg.norm(
        np.array(lm_orig["left_eye"]) - np.array(lm_orig["right_eye"])
    )
    # MTCNN usa iod * ~2.5 como tamanho de crop antes de margin
    crop_orig = iod_orig * 2.5 + 2 * margin
    scale = crop_size / crop_orig

    remapped = {}
    for k, (x, y) in lm_orig.items():
        nx = (x - cx + crop_orig / 2) * scale
        ny = (y - cy + crop_orig / 2) * scale
        remapped[k] = [nx, ny]
    return remapped


def visualize_gradcam(img_path: Path, save_path: Path, pair_score: float = None,
                      pair_label: int = None):
    """
    Gera figura com 3 painéis:
        1. Face original com landmarks e bounding boxes das regiões
        2. Mapa Grad-CAM sobreposto
        3. Grad-CAM com contornos das regiões
    """
    key = f"{img_path.parent.name}/{img_path.name}"

    face_tensor = get_face_tensor(img_path)
    if face_tensor is None:
        print(f"  ⚠ Sem face: {key}")
        return

    # Grad-CAM
    cam = gradcam(face_tensor)

    # Face como array RGB 160×160
    face_np = face_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    face_np = ((face_np - face_np.min()) /
               (face_np.max() - face_np.min()) * 255).astype(np.uint8)

    # Landmarks remapeados para o espaço 160×160
    if key in landmarks:
        img_orig_size = Image.open(img_path).size
        lm_remapped   = remap_landmarks(landmarks[key], img_orig_size)
        boxes         = landmarks_to_boxes(lm_remapped)
    else:
        lm_remapped = None
        boxes       = {}

    heatmap_overlay = overlay_heatmap(face_np, cam)

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    region_colors = {
        "olho_esq": "cyan",
        "olho_dir": "cyan",
        "nariz":    "yellow",
        "boca":     "lime",
        "testa":    "magenta",
    }

    # Painel 1 — face + landmarks + boxes
    axes[0].imshow(face_np)
    axes[0].set_title("Face + Regiões", fontsize=10)
    if lm_remapped:
        for pt_name, (x, y) in lm_remapped.items():
            axes[0].plot(x, y, "ro", markersize=4)
    for region, (x1, y1, x2, y2) in boxes.items():
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=region_colors[region], facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 2, region, color=region_colors[region],
                     fontsize=6, va="bottom")
    axes[0].axis("off")

    # Painel 2 — Grad-CAM puro
    axes[1].imshow(heatmap_overlay)
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[1].axis("off")

    # Painel 3 — Grad-CAM + contornos das regiões
    axes[2].imshow(heatmap_overlay)
    for region, (x1, y1, x2, y2) in boxes.items():
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=region_colors[region], facecolor="none"
        )
        axes[2].add_patch(rect)
    axes[2].set_title("Grad-CAM + Regiões", fontsize=10)
    axes[2].axis("off")

    # Título geral com info do par (se disponível)
    title = key
    if pair_score is not None:
        decision = "✓ Correto" if (
            (pair_score >= THRESHOLD and pair_label == 1) or
            (pair_score < THRESHOLD  and pair_label == 0)
        ) else "✗ Erro"
        title += f"  |  score={pair_score:.3f}  |  {decision}"
    fig.suptitle(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─── Selecionar imagens representativas ───────────────────────────────────────
# Gerar Grad-CAM para:
#   • 5 pares corretos (positivos bem classificados)
#   • 5 pares corretos (negativos bem classificados)
#   • 5 falsos negativos (mesma pessoa mas não reconhecida)
#   • 5 falsos positivos (pessoas diferentes mas reconhecidas como iguais)

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

N = 5  # imagens por categoria
print("\nDistribuição dos pares:")
for cat, items in categories.items():
    print(f"  {cat}: {len(items)}")

processed = set()

for cat, items in categories.items():
    cat_dir = GRADCAM_DIR / cat
    cat_dir.mkdir(exist_ok=True)
    sample  = items[:N]

    for d in sample:
        for key in (d["img1"], d["img2"]):
            if key in processed:
                continue
            processed.add(key)

            person, fname = key.split("/")
            img_path      = DATA_DIR / person / fname
            save_name     = f"{person}_{fname.replace('.jpg','')}.png"
            save_path     = cat_dir / save_name

            print(f"  A gerar Grad-CAM: {key}  [{cat}]")
            visualize_gradcam(
                img_path, save_path,
                pair_score=d["score"],
                pair_label=d["label"]
            )

print(f"\nGrad-CAMs guardados em {GRADCAM_DIR}/")
print("Estrutura:")
print("  gradcam/true_positive/   — pares mesma pessoa, bem classificados")
print("  gradcam/true_negative/   — pares pessoas diferentes, bem classificados")
print("  gradcam/false_negative/  — mesma pessoa não reconhecida  ← casos de falha")
print("  gradcam/false_positive/  — pessoas diferentes confundidas ← casos de falha")