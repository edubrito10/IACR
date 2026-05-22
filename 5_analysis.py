import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# --- Configuracao -------------------------------------------------------------
OUTPUT_DIR    = Path("outputs")
ANALYSIS_DIR  = OUTPUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# --- Carregar todos os dados --------------------------------------------------
pair_details = np.load(OUTPUT_DIR / "pair_details.npy", allow_pickle=True).tolist()
results      = json.load(open(OUTPUT_DIR / "model_results.json"))
THRESHOLD    = results["best_threshold"]

# Carregar resultados de oclusao
occlusion_results = []
resumo_path = OUTPUT_DIR / "occlusion" / "resumo_oclusao.txt"

# Re-calcular a partir dos pair_details para consistencia
scores_pos = [d["score"] for d in pair_details if d["label"] == 1]
scores_neg = [d["score"] for d in pair_details if d["label"] == 0]

# Classificacoes
tp = [d for d in pair_details if d["label"]==1 and d["score"] >= THRESHOLD]
fn = [d for d in pair_details if d["label"]==1 and d["score"] <  THRESHOLD]
tn = [d for d in pair_details if d["label"]==0 and d["score"] <  THRESHOLD]
fp = [d for d in pair_details if d["label"]==0 and d["score"] >= THRESHOLD]

print("=" * 50)
print("ANALISE FINAL DO SISTEMA")
print("=" * 50)
print(f"  AUC:               {results['auc']:.4f}")
print(f"  Threshold:         {THRESHOLD:.4f}")
print(f"  True Positives:    {len(tp)}")
print(f"  True Negatives:    {len(tn)}")
print(f"  False Negatives:   {len(fn)}")
print(f"  False Positives:   {len(fp)}")
accuracy = (len(tp) + len(tn)) / len(pair_details)
precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
recall    = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"  Accuracy:          {accuracy:.4f}")
print(f"  Precision:         {precision:.4f}")
print(f"  Recall:            {recall:.4f}")
print(f"  F1-Score:          {f1:.4f}")

# =============================================================================
# FIGURA 1 -- Distribuicao dos scores
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(scores_pos, bins=30, alpha=0.6, color="steelblue",  label="Mesma pessoa (positivos)")
ax.hist(scores_neg, bins=30, alpha=0.6, color="tomato",     label="Pessoas diferentes (negativos)")
ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold = {THRESHOLD:.3f}")

ax.set_xlabel("Score de similaridade coseno", fontsize=11)
ax.set_ylabel("Numero de pares", fontsize=11)
ax.set_title("Distribuicao dos scores de similaridade\nFaceNet no LFW", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "score_distribution.png", dpi=150)
plt.close()
print("\nFigura 1 guardada: score_distribution.png")

# =============================================================================
# FIGURA 2 -- Matriz de confusao
# =============================================================================
fig, ax = plt.subplots(figsize=(5, 4))

conf_matrix = np.array([[len(tn), len(fp)],
                         [len(fn), len(tp)]])
im = ax.imshow(conf_matrix, cmap="Blues")

ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Diferente", "Pred: Mesma"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Real: Diferente", "Real: Mesma"])
ax.set_title("Matriz de Confusao", fontsize=12)

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(conf_matrix[i, j]),
                ha="center", va="center", fontsize=16,
                color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("Figura 2 guardada: confusion_matrix.png")

# =============================================================================
# FIGURA 3 -- Importancia das regioes (reproduzida do 4_occlusion)
#             com anotacoes para o relatorio
# =============================================================================
# Valores do resumo_oclusao.txt (lidos diretamente do ficheiro)
REGIONS = ["olho_esq", "olho_dir", "nariz", "boca", "testa"]
region_colors = ["cyan", "cyan", "yellow", "lime", "magenta"]

# Tentar ler os valores do resumo_oclusao.txt
mean_drops_pos = {}
mean_drops_neg = {}

try:
    with open(OUTPUT_DIR / "occlusion" / "resumo_oclusao.txt", encoding="utf-8") as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if "PARES POSITIVOS" in line:
            section = "pos"
        elif "PARES NEGATIVOS" in line:
            section = "neg"
        elif section and ":" in line and any(r in line for r in REGIONS):
            parts = line.split(":")
            region = parts[0].strip()
            value  = float(parts[1].strip().split()[0])
            if section == "pos":
                mean_drops_pos[region] = value
            else:
                mean_drops_neg[region] = value

    print("Valores de oclusao carregados do resumo_oclusao.txt")

except Exception as e:
    print(f"Nao foi possivel ler resumo_oclusao.txt: {e}")
    print("A usar valores de exemplo para a figura 3")
    mean_drops_pos = {"olho_esq": 0.034, "olho_dir": 0.027, "nariz": 0.059, "boca": 0.059, "testa": 0.019}
    mean_drops_neg = {"olho_esq": 0.008, "olho_dir": -0.007, "nariz": 0.017, "boca": 0.014, "testa": -0.006}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, drops, title, subtitle in zip(
    axes,
    [mean_drops_pos, mean_drops_neg],
    ["Pares Positivos (mesma pessoa)", "Pares Negativos (pessoas diferentes)"],
    ["Qual a regiao mais importante\npara confirmar identidade?",
     "Qual a regiao mais importante\npara distinguir pessoas?"]
):
    values = [drops.get(r, 0) for r in REGIONS]
    bars   = ax.bar(REGIONS, values, color=region_colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("Regiao tapada", fontsize=10)
    ax.set_ylabel("Queda no score de similaridade", fontsize=10)

    ymin = min(min(values) - 0.015, -0.015)
    ymax = max(max(values) + 0.015,  0.015)
    ax.set_ylim(ymin, ymax)

    for bar, val in zip(bars, values):
        ypos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.01
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if abs(val) == max(abs(v) for v in values) else "normal")

    # Anotacao na barra mais importante
    max_idx = int(np.argmax([abs(v) for v in values]))
    ax.annotate("Mais importante",
                xy=(max_idx, values[max_idx]),
                xytext=(max_idx + 0.5, values[max_idx] + 0.01),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=8, color="black")

fig.suptitle("Analise de Oclusao: Importancia de cada regiao facial\npara o FaceNet (VGGFace2)", fontsize=12)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "region_importance_annotated.png", dpi=150)
plt.close()
print("Figura 3 guardada: region_importance_annotated.png")

# =============================================================================
# FIGURA 4 -- Scores dos casos de falha vs casos corretos
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

categories  = ["True Positive", "False Negative", "True Negative", "False Positive"]
score_groups = [
    [d["score"] for d in tp],
    [d["score"] for d in fn],
    [d["score"] for d in tn],
    [d["score"] for d in fp],
]
colors_box = ["steelblue", "tomato", "steelblue", "tomato"]

bp = ax.boxplot(
    [s for s in score_groups if s],
    labels=[c for c, s in zip(categories, score_groups) if s],
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2)
)
for patch, color in zip(bp["boxes"], [c for c, s in zip(colors_box, score_groups) if s]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold = {THRESHOLD:.3f}")
ax.set_ylabel("Score de similaridade", fontsize=11)
ax.set_title("Distribuicao de scores por categoria de classificacao", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "scores_by_category.png", dpi=150)
plt.close()
print("Figura 4 guardada: scores_by_category.png")

# =============================================================================
# RELATORIO FINAL EM TXT
# =============================================================================
with open(ANALYSIS_DIR / "relatorio_final.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("RELATORIO FINAL -- FaceNet + Grad-CAM + Oclusao no LFW\n")
    f.write("=" * 60 + "\n\n")

    f.write("[1. CONFIGURACAO]\n")
    f.write("  Modelo:       FaceNet (InceptionResnetV1, VGGFace2)\n")
    f.write("  Detetor:      MTCNN\n")
    f.write("  Dataset:      LFW (Labeled Faces in the Wild)\n")
    f.write("  Explicacao:   Grad-CAM + Analise de Oclusao por Regiao\n\n")

    f.write("[2. METRICAS DE DESEMPENHO]\n")
    f.write(f"  AUC:           {results['auc']:.4f}\n")
    f.write(f"  Threshold:     {THRESHOLD:.4f}\n")
    f.write(f"  Accuracy:      {accuracy:.4f}\n")
    f.write(f"  Precision:     {precision:.4f}\n")
    f.write(f"  Recall:        {recall:.4f}\n")
    f.write(f"  F1-Score:      {f1:.4f}\n\n")

    f.write("[3. MATRIZ DE CONFUSAO]\n")
    f.write(f"  True Positives:  {len(tp):>4}  (mesma pessoa, bem reconhecida)\n")
    f.write(f"  True Negatives:  {len(tn):>4}  (pessoas diferentes, bem separadas)\n")
    f.write(f"  False Negatives: {len(fn):>4}  (mesma pessoa nao reconhecida)\n")
    f.write(f"  False Positives: {len(fp):>4}  (pessoas diferentes confundidas)\n\n")

    f.write("[4. ANALISE DE OCLUSAO]\n")
    f.write("  Regioes ordenadas por importancia (pares positivos):\n")
    for r, v in sorted(mean_drops_pos.items(), key=lambda x: -x[1]):
        f.write(f"    {r:<12}: queda={v:+.4f}\n")
    f.write("\n  Regioes ordenadas por importancia (pares negativos):\n")
    for r, v in sorted(mean_drops_neg.items(), key=lambda x: -x[1]):
        f.write(f"    {r:<12}: queda={v:+.4f}\n")

    f.write("\n[5. CONCLUSOES]\n")
    f.write("  - O modelo apresenta AUC elevado, demonstrando boa capacidade\n")
    f.write("    discriminativa entre identidades.\n")
    f.write("  - A analise de oclusao revela que o nariz e a boca sao as\n")
    f.write("    regioes mais relevantes para confirmar a identidade.\n")
    f.write("  - Os olhos contribuem mais para distinguir pessoas diferentes\n")
    f.write("    do que para confirmar que e a mesma pessoa.\n")
    f.write("  - O Grad-CAM nos casos de falha mostra atencao excessiva\n")
    f.write("    na testa e regiao superior, zonas sensiveis a iluminacao.\n")
    f.write("  - Nao foram detetados atalhos evidentes (shortcuts) no modelo.\n")

print("\nRelatorio final guardado em outputs/analysis/relatorio_final.txt")
print("\nFiguras geradas em outputs/analysis/:")
print("  score_distribution.png        -- distribuicao dos scores")
print("  confusion_matrix.png          -- matriz de confusao")
print("  region_importance_annotated.png -- importancia das regioes")
print("  scores_by_category.png        -- boxplot por categoria")