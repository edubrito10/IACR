# Trabalho Prático de Explicabilidade em Verificação Facial (FaceNet) 

**Unidade Curricular:** Inteligência Artificial Confiável e Responsável  
**Instituição:** Universidade da Beira Interior (UBI) - Mestrado em Engenharia Informática  
**Ano Letivo:** 2025/2026  
**Autores:** Eduardo Brito (M15384) e Rodrigo Sousa (M15782)  

---

## Sobre o Projeto

Este repositório contém o código-fonte desenvolvido para a auditoria de um sistema de verificação facial baseado no modelo **FaceNet** (InceptionResnetV1 pré-treinado no VGGFace2). 

O objetivo principal deste trabalho é garantir a explicabilidade das decisões da rede (IA Responsável), investigando se a confirmação de identidade assenta em biometria fidedigna ou em "atalhos" contextuais espúrios. A avaliação foi realizada sobre o *dataset* **LFW (Labeled Faces in the Wild)** e recorre a duas metodologias complementares:

1. **Explicabilidade Visual:** Implementação de um **Grad-CAM Adaptado**, orientado diretamente à Similaridade Coseno entre o par de imagens.
2. **Validação Quantitativa:** Oclusão semântica e sistemática de 5 regiões faciais (olho esquerdo, olho direito, nariz, boca e testa), extraídas geometricamente através do detetor **MTCNN**.

---

## Estrutura do Repositório

A arquitetura de software foi desenhada de forma modular e sequencial. A execução (pipeline experimental) deve seguir a ordem numérica dos ficheiros:

* `1_load_data.py`: Filtragem, transferência e balanceamento do *dataset* LFW (150 pares positivos e 150 pares negativos).
* `2_model.py`: Deteção facial, extração de *landmarks* (MTCNN), geração de *embeddings* (FaceNet) e determinação do *threshold* ótimo pela curva ROC.
* `3_gradcam.py`: Geração e adaptação dos mapas de calor (explicabilidade visual).
* `4_occlusion.py`: Mascaramento dinâmico por preenchimento de tensores a zeros e recálculo das quedas de *score*.
* `5_analysis.py`: Cálculo das métricas de desempenho final (AUC, Matriz de Confusão) e exportação dos gráficos quantitativos.
* `outputs/`: Diretoria gerada automaticamente durante a execução, onde são guardados os tensores (`.npy`), métricas (`.json`) e visualizações.

---

## Instalação e Execução

### Pré-requisitos
Certifique-se de que tem o Python 3.8+ instalado. Instale as dependências necessárias através do comando:


pip install torch torchvision facenet-pytorch numpy pandas matplotlib scikit-learn opencv-python