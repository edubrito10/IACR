import os
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
from itertools import combinations
import random

# ─── Configuração ────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/lfw")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A usar dispositivo: {device}")

# ─── Carregar modelos ─────────────────────────────────────────────────────────
print("A carregar MTCNN e FaceNet...")
# return_prob=False por omissão; landmarks=True ativa os pontos faciais
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    device=device,
)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ─── Extrair embeddings + landmarks ──────────────────────────────────────────
def get_embedding_and_landmarks(img_path: Path):
    """
    Deteta a face, extrai o embedding e os landmarks faciais.

    O MTCNN devolve landmarks como array (5, 2) com os pontos:
        0 → olho esquerdo
        1 → olho direito
        2 → nariz
        3 → canto esquerdo da boca
        4 → canto direito da boca
    Coordenadas em píxeis relativos à imagem ORIGINAL (antes do crop).

    Devolve:
        embedding  : np.ndarray (512,)  ou None
        landmarks  : dict com chaves 'left_eye', 'right_eye', 'nose',
                     'mouth_left', 'mouth_right'  ou None
        img_size   : (width, height) da imagem original
    """
    img = Image.open(img_path).convert("RGB")
    img_size = img.size  # (W, H)

    # detect() devolve boxes, probs, landmarks  (landmarks shape: (1, 5, 2))
    boxes, probs, lm = mtcnn.detect(img, landmarks=True)

    if boxes is None or lm is None:
        return None, None, img_size

    # Usar a face com maior probabilidade (índice 0 já é a melhor)
    face_tensor = mtcnn(img)
    if face_tensor is None:
        return None, None, img_size

    with torch.no_grad():
        emb = facenet(face_tensor.unsqueeze(0).to(device))

    # lm[0] → landmarks da primeira (melhor) face detetada, shape (5, 2)
    pts = lm[0]
    landmarks = {
        "left_eye":    pts[0].tolist(),
        "right_eye":   pts[1].tolist(),
        "nose":        pts[2].tolist(),
        "mouth_left":  pts[3].tolist(),
        "mouth_right": pts[4].tolist(),
    }

    return emb.squeeze().cpu().numpy(), landmarks, img_size


print("A extrair embeddings e landmarks...")
embeddings = {}   # { "pessoa/0.jpg": np.ndarray }
landmarks  = {}   # { "pessoa/0.jpg": dict }
img_sizes  = {}   # { "pessoa/0.jpg": (W, H) }
failed     = []

for person_dir in sorted(DATA_DIR.iterdir()):
    if not person_dir.is_dir():
        continue
    for img_path in sorted(person_dir.glob("*.jpg")):
        key = f"{person_dir.name}/{img_path.name}"
        emb, lm, sz = get_embedding_and_landmarks(img_path)
        if emb is not None:
            embeddings[key] = emb
            landmarks[key]  = lm
            img_sizes[key]  = sz
        else:
            failed.append(key)
            print(f"  ⚠ Face não detetada: {key}")

print(f"\nEmbeddings extraídos: {len(embeddings)}  |  Falhas: {len(failed)}")

# ─── Construir pares ──────────────────────────────────────────────────────────
people = {}
for key in embeddings:
    name = key.split("/")[0]
    people.setdefault(name, []).append(key)

positive_pairs = []
for name, keys in people.items():
    if len(keys) >= 2:
        positive_pairs += list(combinations(keys, 2))

random.seed(42)
negative_pairs = []
names_list = [n for n, k in people.items() if len(k) >= 1]
while len(negative_pairs) < len(positive_pairs):
    n1, n2 = random.sample(names_list, 2)
    k1 = random.choice(people[n1])
    k2 = random.choice(people[n2])
    negative_pairs.append((k1, k2))

print(f"Pares positivos: {len(positive_pairs)}  |  Pares negativos: {len(negative_pairs)}")

# ─── Similaridade coseno ──────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

scores  = []
labels  = []
details = []

for k1, k2 in positive_pairs:
    s = cosine_similarity(embeddings[k1], embeddings[k2])
    scores.append(s); labels.append(1)
    details.append({"img1": k1, "img2": k2, "score": s, "label": 1})

for k1, k2 in negative_pairs:
    s = cosine_similarity(embeddings[k1], embeddings[k2])
    scores.append(s); labels.append(0)
    details.append({"img1": k1, "img2": k2, "score": s, "label": 0})

# ─── Curva ROC + AUC ─────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)
print(f"\nAUC: {roc_auc:.4f}")

best_idx    = np.argmax(tpr - fpr)
best_thresh = float(thresholds[best_idx])
print(f"Melhor threshold: {best_thresh:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"FaceNet (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Curva ROC – FaceNet no LFW"); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()
print("Curva ROC guardada em outputs/roc_curve.png")

# ─── Guardar tudo ─────────────────────────────────────────────────────────────
results = {
    "auc": roc_auc,
    "best_threshold": best_thresh,
    "n_positive_pairs": len(positive_pairs),
    "n_negative_pairs": len(negative_pairs),
    "n_failed_detections": len(failed),
    "failed_images": failed,
}
with open(OUTPUT_DIR / "model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Embeddings, landmarks e detalhes dos pares — usados nos scripts seguintes
np.save(OUTPUT_DIR / "embeddings.npy",    embeddings)
np.save(OUTPUT_DIR / "landmarks.npy",     landmarks)
np.save(OUTPUT_DIR / "img_sizes.npy",     img_sizes)
np.save(OUTPUT_DIR / "pair_details.npy",  details)

print("\nFicheiros guardados em outputs/:")
print("  embeddings.npy   — embeddings de cada imagem")
print("  landmarks.npy    — 5 pontos faciais por imagem")
print("  img_sizes.npy    — dimensões das imagens originais")
print("  pair_details.npy — pares com scores e labels")
print("  roc_curve.png    — curva ROC")
print("  model_results.json — métricas gerais")