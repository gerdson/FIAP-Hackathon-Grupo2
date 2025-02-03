from ultralytics import YOLO
import cv2
import torch

MODELO_CAMINHO = "./runs/detect/train/weights/best.pt"
modelo = YOLO(MODELO_CAMINHO)

frame = cv2.imread("assets/faca.png") # Substitua pelo caminho da sua imagem de teste
caixa = [50, 50, 200, 200] # Exemplo de caixa, ajuste conforme necessário
x1, y1, x2, y2 = map(int, caixa)
roi = frame[y1:y2, x1:x2]

indice_camada_teste = [10] # Ou outro índice para testar
resultados_embed = modelo.predict(roi, verbose=True, augment=False, embed=indice_camada_teste)

print(f"Tipo de resultados_embed: {type(resultados_embed)}")
if resultados_embed and isinstance(resultados_embed[0], torch.Tensor):
    embedding = resultados_embed[0]
    print(f"Embeddings extraídos com sucesso da camada {indice_camada_teste}:")
    print(f"Shape do embedding: {embedding.shape}")
else:
    print(f"Falha ao extrair embeddings com camada {indice_camada_teste} ou resultados não são Tensor.")