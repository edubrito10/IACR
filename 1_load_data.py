from datasets import load_dataset
from collections import defaultdict
from pathlib import Path


# carregar dataset
ds = load_dataset("bitmind/lfw", split="train")

output_dir = Path("data/lfw")
output_dir.mkdir(parents=True, exist_ok=True)

people = defaultdict(list)


# agrupar por pessoas
for sample in ds:
    filename = sample["filename"]
    name = filename.rsplit("_", 1)[0] # George_W_Bush_0001.jpg --> George_W_Bush
    people[name].append(sample["image"])

count = 0
max_images = 300 # 150pessoas x 2fotos


for name, images in people.items():
    if len(images) < 2:
        continue

    person_dir = output_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    # guardar 2 imagens por pessoa
    for i, img in enumerate(images[:2]):
        img.save(person_dir / f"{i}.jpg")
        count += 1

    if count >= max_images:
        break

print(f"Sucesso! Guardadas {count} imagens na diretoria {output_dir}")