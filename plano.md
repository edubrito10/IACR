# Plano de Desenvolvimento de Código - Trabalho Prático de IACR

Este documento mapeia todas as etapas de desenvolvimento de software necessárias para concluir o projeto de Explicabilidade de Decisão usando o modelo FaceNet e o dataset LFW.

## 1. Configuração do Ambiente
- [ ] Criar o ambiente virtual (`venv`).
- [ ] Instalar as bibliotecas necessárias: `deepface`, `opencv-python`, `mediapipe`, `numpy`, `matplotlib`, `scikit-learn` e `pandas`.
- [ ] Criar ficheiro `requirements.txt` para garantir que todos no grupo usam as mesmas versões.

## 2. Preparação e Exploração de Dados (Script: `1_load_data.py`)
- [ ] Fazer o download do *dataset* LFW (via `scikit-learn` ou ficheiros raw).
- [ ] Criar uma função para selecionar pares de imagens da mesma pessoa (para testar os "verdadeiros positivos").
- [ ] Criar uma função para selecionar pares de pessoas diferentes (para testar os "verdadeiros negativos").

## 3. Pré-processamento e Landmarks (Script: `2_face_landmarks.py`)
- [ ] Integrar o MediaPipe Face Mesh.
- [ ] Criar uma função que recebe uma imagem e devolve as coordenadas exatas (polígonos) de 4 regiões chave:
  - Olho Esquerdo e Olho Direito
  - Nariz
  - Boca
  - Contorno Facial (opcional, mas recomendado)
- [ ] Garantir que o script lida com imagens onde o MediaPipe não consegue detetar o rosto (tratamento de erros).

## 4. Oclusão Semântica (Script: `3_occlusion_generator.py`)
- [ ] Desenvolver funções usando OpenCV (`cv2.fillPoly`) para aplicar máscaras nas imagens baseando-se nas coordenadas do script anterior.
- [ ] Criar variantes de oclusão:
  - Oclusão total (polígono preto).
  - Oclusão por desfoque (*Gaussian Blur* intenso) - muitas vezes o desfoque é menos disruptivo para a rede neuronal do que um quadrado preto.

## 5. Avaliação Experimental e Inferência (Script: `4_evaluate_facenet.py`)
- [ ] Instanciar o modelo FaceNet usando a biblioteca DeepFace.
- [ ] **Experiência Base:** Passar os pares de imagens originais (sem máscaras) e guardar a distância/similaridade e a decisão final.
- [ ] **Experiências de Explicabilidade:** Iterar sobre os mesmos pares, mas aplicando oclusão numa região de cada vez (ex: só tapar olhos, depois só nariz, depois só boca).
- [ ] Guardar todos os resultados (Imagem A, Imagem B, Tipo de Oclusão, Distância Obtida, Decisão) num ficheiro estruturado (ex: `resultados.csv`).

## 6. Análise de Resultados e Métricas (Script/Notebook: `5_data_analysis.ipynb`)
- [ ] Ler o ficheiro `resultados.csv` usando o Pandas.
- [ ] Calcular a queda de precisão (*accuracy drop*) para cada região facial ocultada.
- [ ] Isolar os "Casos de Falha": identificar pares onde o FaceNet acertou na versão original, mas falhou redondamente com a oclusão.
- [ ] Gerar gráficos (barras, *boxplots*) usando o `matplotlib` ou `seaborn` para exportar para o relatório final e para a apresentação.