from datasets import load_dataset
from collections import defaultdict
from pathlib import Path

print("A carregar o dataset LFW da Hugging Face...")
# O primeiro download pode demorar um pouco
ds = load_dataset("bitmind/lfw", split="train")

output_dir = Path("data/lfw")
output_dir.mkdir(parents=True, exist_ok=True)

people = defaultdict(list)

print("A organizar as imagens por pessoa...")
# Agrupar imagens por pessoa
for sample in ds:
    filename = sample["filename"]
    # Isolar o nome: 'George_W_Bush_0001.jpg' passa a 'George_W_Bush'
    name = filename.rsplit("_", 1)[0]
    people[name].append(sample["image"])

count = 0
max_images = 300 # 150 pessoas x 2 fotos

print("A extrair e a guardar as imagens fisicamente no disco...")
for name, images in people.items():
    # Ignorar pessoas que só têm 1 fotografia (não servem para criar pares da mesma pessoa)
    if len(images) < 2:
        continue

    person_dir = output_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Guardar exatamente 2 imagens por pessoa
    for i, img in enumerate(images[:2]):
        img.save(person_dir / f"{i}.jpg")
        count += 1

    if count >= max_images:
        break

print(f"Sucesso! Guardadas {count} imagens na diretoria {output_dir}")