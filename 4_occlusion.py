import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# --- Configuracao -------------------------------------------------------------
DATA_DIR     = Path("data/lfw")
OUTPUT_DIR   = Path("outputs")
OCCLUSION_DIR = OUTPUT_DIR / "occlusion"
OCCLUSION_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A usar dispositivo: {device}")

# --- Carregar modelos ---------------------------------------------------------
mtcnn   = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# --- Carregar dados do 2_model.py ---------------------------------------------
landmarks    = np.load(OUTPUT_DIR / "landmarks.npy",    allow_pickle=True).item()
embeddings   = np.load(OUTPUT_DIR / "embeddings.npy",   allow_pickle=True).item()
pair_details = np.load(OUTPUT_DIR / "pair_details.npy", allow_pickle=True).tolist()
results      = json.load(open(OUTPUT_DIR / "model_results.json"))
THRESHOLD    = results["best_threshold"]

# --- Funcoes auxiliares -------------------------------------------------------

def get_face_tensor(img_path):                                          # deteta e recorta a face usando o MTCNN
    img = Image.open(img_path).convert("RGB")
    t   = mtcnn(img)
    return t.unsqueeze(0).to(device) if t is not None else None


def cosine_similarity(a, b):                                                    #calcula a similaridade coseno entre dois embeddings
    return float(F.cosine_similarity(a, b).item())


def get_embedding_tensor(face_tensor):
    with torch.no_grad():
        return facenet(face_tensor)


def remap_landmarks(lm_orig, img_orig_size, crop_size=160, margin=20):   #converte os landmarks da imagem original para as coordenadas da face recortada
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


def landmarks_to_boxes(lm, img_w=160, img_h=160):
    le = np.array(lm["left_eye"])
    re = np.array(lm["right_eye"])
    no = np.array(lm["nose"])
    ml = np.array(lm["mouth_left"])
    mr = np.array(lm["mouth_right"])

    iod = np.linalg.norm(re - le)       #distancia interocular. Usada como referencia para o tamanho das regioes
    pad = iod * 0.35                    #tamanho base para as regioes de 35% da distancia interocular

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


def occlude_region(face_tensor, box, fill_value=0.0):   #fill_value preenche com zeros(preto)
    
    occluded = face_tensor.clone()         #fazer copia para nao modificar o tensor original (.clone())
    x1, y1, x2, y2 = box
    occluded[0, :, y1:y2, x1:x2] = fill_value    # seleciona a região a tapar em todos os canais RGB simultaneamente
    return occluded


def analyze_pair_occlusion(key1, key2, label, score_base):

    img_path1 = DATA_DIR / key1.split("/")[0] / key1.split("/")[1]
    img_path2 = DATA_DIR / key2.split("/")[0] / key2.split("/")[1]

    face1 = get_face_tensor(img_path1)
    face2 = get_face_tensor(img_path2)

    if face1 is None or face2 is None:
        return None

    emb2 = get_embedding_tensor(face2)

    # Landmarks da img1
    if key1 not in landmarks:
        return None

    img_orig_size = Image.open(img_path1).size
    lm_remapped   = remap_landmarks(landmarks[key1], img_orig_size)
    boxes         = landmarks_to_boxes(lm_remapped)

    # Score base (sem oclusao) -- deve ser igual ao guardado em pair_details
    emb1_base  = get_embedding_tensor(face1)
    score_base_calc = cosine_similarity(emb1_base, emb2)

    # Score por regiao tapada
    #para cadaregiao, tapa-a e recalcula o score
    region_drops = {}
    for region, box in boxes.items():
        face1_occ   = occlude_region(face1, box)
        emb1_occ    = get_embedding_tensor(face1_occ)
        score_occ   = cosine_similarity(emb1_occ, emb2)
        drop        = score_base_calc - score_occ   # queda no score
        region_drops[region] = {                                    #guarda os resultados para esta regiao 
            "score_base":     round(score_base_calc, 4),
            "score_occluded": round(score_occ,       4),
            "drop":           round(drop,            4),
        }

    return region_drops


# --- Selecionar pares para analise --------------------------------------------
# Analisar pares positivos (mesma pessoa) -- os mais informativos
# para perceber quais as regioes mais importantes para o reconhecimento

positive_pairs = [d for d in pair_details if d["label"] == 1]
negative_pairs = [d for d in pair_details if d["label"] == 0]

# Usar os primeiros 20 pares de cada tipo
N = 20
selected_pos = positive_pairs[:N]
selected_neg = negative_pairs[:N]

print(f"A analisar {len(selected_pos)} pares positivos e {len(selected_neg)} negativos...")

# --- Calcular quedas por regiao -----------------------------------------------
REGIONS = ["olho_esq", "olho_dir", "nariz", "boca", "testa"]

# Acumular quedas medias por regiao
drops_pos = {r: [] for r in REGIONS}
drops_neg = {r: [] for r in REGIONS}
all_results = []

for d in selected_pos:                          #calcula as quedas para cada par positivo e acumula os resultados para fazer media
    res = analyze_pair_occlusion(d["img1"], d["img2"], d["label"], d["score"])
    if res is None:
        continue
    for r in REGIONS:
        drops_pos[r].append(res[r]["drop"])
    all_results.append({
        "img1": d["img1"], "img2": d["img2"],
        "label": d["label"], "score_base": d["score"],
        "regions": res
    })

for d in selected_neg:                                          #calcula as quedas para cada par negativo e acumula os resultados para fazer media
    res = analyze_pair_occlusion(d["img1"], d["img2"], d["label"], d["score"])
    if res is None:
        continue
    for r in REGIONS:
        drops_neg[r].append(res[r]["drop"])
    all_results.append({
        "img1": d["img1"], "img2": d["img2"],
        "label": d["label"], "score_base": d["score"],
        "regions": res
    })

# Medias
mean_drops_pos = {r: round(float(np.mean(v)), 4) for r, v in drops_pos.items() if v}
mean_drops_neg = {r: round(float(np.mean(v)), 4) for r, v in drops_neg.items() if v}

print("\nQueda media no score por regiao tapada (pares positivos):")
for r, v in sorted(mean_drops_pos.items(), key=lambda x: -x[1]):
    print(f"  {r:<12}: {v:+.4f}")

print("\nQueda media no score por regiao tapada (pares negativos):")
for r, v in sorted(mean_drops_neg.items(), key=lambda x: -x[1]):
    print(f"  {r:<12}: {v:+.4f}")

# --- Grafico de barras --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ["cyan", "cyan", "yellow", "lime", "magenta"]
region_labels = ["olho_esq", "olho_dir", "nariz", "boca", "testa"]

for ax, drops, title in zip(
    axes,
    [mean_drops_pos, mean_drops_neg],
    ["Pares Positivos (mesma pessoa)", "Pares Negativos (pessoas diferentes)"]
):
    values = [drops.get(r, 0) for r in region_labels]
    bars   = ax.bar(region_labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Regiao tapada")
    ax.set_ylabel("Queda no score de similaridade")
    ax.set_ylim(min(min(values) - 0.02, -0.02), max(max(values) + 0.02, 0.02))

    # Valor em cima de cada barra
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.008,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Importancia de cada regiao facial para o FaceNet\n(queda no score ao tapar cada regiao)", fontsize=12)
plt.tight_layout()
plt.savefig(OCCLUSION_DIR / "region_importance.png", dpi=150)
plt.close()
print("\nGrafico guardado em outputs/occlusion/region_importance.png")






# --- Visualizacao de exemplos -------------------------------------------------
# Gerar imagem com os 5 paineis de oclusao para os primeiros 3 pares positivos

def visualize_occlusion(pair_data, save_path):
    
    key1 = pair_data["img1"]
    img_path1 = DATA_DIR / key1.split("/")[0] / key1.split("/")[1]

    face1 = get_face_tensor(img_path1)
    if face1 is None:
        return

    img_orig_size = Image.open(img_path1).size
    lm_remapped   = remap_landmarks(landmarks[key1], img_orig_size)
    boxes         = landmarks_to_boxes(lm_remapped)

    def tensor_to_img(t):
        arr = t.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return (arr * 255).astype(np.uint8)

    region_colors_map = {
        "olho_esq": "cyan", "olho_dir": "cyan",
        "nariz": "yellow", "boca": "lime", "testa": "magenta"
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    # Painel 0 -- face original com boxes
    face_img = tensor_to_img(face1)
    axes[0].imshow(face_img)
    axes[0].set_title(f"Original\nscore={pair_data['score_base']:.3f}", fontsize=9)
    for region, (x1, y1, x2, y2) in boxes.items():
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5, edgecolor=region_colors_map[region], facecolor="none"
        )
        axes[0].add_patch(rect)
    axes[0].axis("off")

    # Paineis 1-4 -- cada regiao tapada
    for i, region in enumerate(REGIONS[:5], start=1):
        face_occ = occlude_region(face1, boxes[region])
        occ_img  = tensor_to_img(face_occ)
        drop     = pair_data["regions"][region]["drop"]
        score_occ = pair_data["regions"][region]["score_occluded"]

        axes[i].imshow(occ_img)
        axes[i].set_title(
            f"Sem {region}\nscore={score_occ:.3f}  (queda={drop:+.3f})",
            fontsize=9,
            color="red" if drop > 0.05 else "black"
        )
        # Destacar a regiao tapada
        x1, y1, x2, y2 = boxes[region]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="red", facecolor="red", alpha=0.5
        )
        axes[i].add_patch(rect)
        axes[i].axis("off")

    fig.suptitle(f"{key1}  vs  {pair_data['img2']}\nlabel={'mesma pessoa' if pair_data['label']==1 else 'pessoas diferentes'}",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


print("\nA gerar visualizacoes de oclusao...")
vis_dir = OCCLUSION_DIR / "visualizacoes"
vis_dir.mkdir(exist_ok=True)

for i, d in enumerate(all_results[:6]):
    label_str = "pos" if d["label"] == 1 else "neg"
    save_path = vis_dir / f"{label_str}{i:02d}{d['img1'].replace('/', '_')}.png"
    print(f"  {d['img1']} vs {d['img2']}")
    visualize_occlusion(d, save_path)

# --- Guardar resultados em TXT ------------------------------------------------
with open(OCCLUSION_DIR / "resumo_oclusao.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("ANALISE DE OCLUSAO POR REGIAO FACIAL\n")
    f.write("=" * 60 + "\n\n")

    f.write("[INTERPRETACAO]\n")
    f.write("  Queda positiva -> tapar esta regiao reduziu o score (regiao importante)\n")
    f.write("  Queda negativa -> tapar esta regiao aumentou o score (regiao confunde)\n\n")

    f.write("[MEDIA POR REGIAO - PARES POSITIVOS (mesma pessoa)]\n")
    for r, v in sorted(mean_drops_pos.items(), key=lambda x: -x[1]):
        barra = "#" * int(abs(v) * 100)
        f.write(f"  {r:<12}: {v:+.4f}  {barra}\n")

    f.write("\n[MEDIA POR REGIAO - PARES NEGATIVOS (pessoas diferentes)]\n")
    for r, v in sorted(mean_drops_neg.items(), key=lambda x: -x[1]):
        barra = "#" * int(abs(v) * 100)
        f.write(f"  {r:<12}: {v:+.4f}  {barra}\n")

    f.write("\n[DETALHE POR PAR]\n")
    for d in all_results:
        label_str = "mesma pessoa" if d["label"] == 1 else "pessoas diferentes"
        f.write(f"\n  {d['img1']} vs {d['img2']}  [{label_str}]  score_base={d['score_base']:.4f}\n")
        for r in REGIONS:
            info = d["regions"][r]
            f.write(f"    {r:<12}: base={info['score_base']:.4f}  "
                    f"occluded={info['score_occluded']:.4f}  "
                    f"queda={info['drop']:+.4f}\n")

print("\nResultados guardados em outputs/occlusion/:")
print("  region_importance.png  -- grafico de barras por regiao")
print("  visualizacoes/         -- imagens com cada regiao tapada")
print("  resumo_oclusao.txt     -- tabela completa legivel")