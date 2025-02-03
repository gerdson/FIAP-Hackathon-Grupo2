import cv2
import time
import torch
import numpy as np
import gradio as gr
from collections import defaultdict, deque
from ultralytics import YOLO
from typing import Tuple, Any

# ====================================================
# CONFIGURAÇÕES GLOBAIS
# ====================================================
CONFIG_CLASSES = {
    0: {"nome": "cortante", "cooldown": 30, "cor": (0, 0, 255)},
    1: {"nome": "arma", "cooldown": 15, "cor": (255, 0, 0)},
}
IOU_THRESHOLD = 0.3
SIMILARIDADE_THRESHOLD = 0.85  # Similaridade mínima entre embeddings
HISTORICO_EMBEDDINGS = 5  # Número de embeddings armazenados por ID
MODELO_CAMINHO = "./runs/detect/train/weights/best.pt"
MODELO_DETECCAO = YOLO(MODELO_CAMINHO) # Carregar modelo para detecção (separado)
MODELO_EMBEDDING = YOLO(MODELO_CAMINHO) # Carregar modelo para embeddings (separado)
FONTE_WEBCAM = 0  # 0 para webcam padrão


# ====================================================
# FUNÇÕES AUXILIARES
# ====================================================
def inicializar_modelo() -> YOLO: # Função inicializar_modelo não é mais necessária
    return MODELO_DETECCAO # Retornar o modelo de detecção pré-carregado


def calcular_iou(caixa1: np.ndarray, caixa2: np.ndarray) -> float:
    """Calcula a intersecção sobre união entre duas caixas delimitadoras"""
    x1_1, y1_1, x2_1, y2_1 = caixa1
    x1_2, y1_2, x2_2, y2_2 = caixa2

    # Calcula área de interseção
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    intersecao = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_total = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - intersecao

    return intersecao / area_total if area_total != 0 else 0


def extrair_embedding_defeito(modelo: YOLO, frame: np.ndarray, caixa: np.ndarray) -> torch.Tensor:
    """Extrai vetor de características da região do objeto detectado"""
    x1, y1, x2, y2 = map(int, caixa)
    roi = frame[y1:y2, x1:x2]

    # Fallback para ROIs inválidas
    if roi.size == 0:
        return torch.zeros(512)

    # Extrai embeddings usando o parâmetro embed=True
    resultados = modelo.predict(roi, verbose=False, augment=False, embed=True)
    return torch.from_numpy(resultados[0].embeddings.flatten())

def extrair_embedding(modelo: YOLO, frame: np.ndarray, caixa: np.ndarray) -> torch.Tensor:

    print(f"Tipo de modelo dentro de extrair_embedding: {type(modelo)}")
    print(f"Tipo de frame dentro de extrair_embedding: {type(frame)}")
    print(f"Tipo de caixa dentro de extrair_embedding: {type(caixa)}")

    """Extrai vetor de características da região do objeto detectado"""
    x1, y1, x2, y2 = map(int, caixa)
    roi = frame[y1:y2, x1:x2]

    # Fallback para ROIs inválidas
    if roi.size == 0:
        return torch.zeros(512)

    # Use embed=[10] (or other index)
    indice_camada_teste = [10]
    resultados = MODELO_EMBEDDING.predict(roi, verbose=False, augment=False, embed=indice_camada_teste)
    #resultados = modelo.predict(roi, verbose=False, augment=False)

    # Check if resultados[0] is a Tensor and use it directly
    if isinstance(resultados[0], torch.Tensor):  # <--- Check if it's a Tensor
        embedding = resultados[0]  # <--- Use resultados[0] directly as embedding
        print(f"Embeddings extraídos com sucesso da camada {indice_camada_teste}:")
        print(f"Shape do embedding: {embedding.shape}")  # Verify shape
        return embedding.flatten()  # Flatten the embedding Tensor
    else:
        print(f"Falha ao extrair embeddings com camada {indice_camada_teste} ou resultados não são Tensor.")
        return torch.zeros(512)  # Fallback


def verificar_similaridade(embedding_atual: torch.Tensor, historico: deque) -> bool:
    """Compara embedding atual com histórico usando similaridade de cosseno"""
    if not historico:
        return False

    similaridades = [
        torch.nn.functional.cosine_similarity(embedding_atual, emb, dim=0).item()
        for emb in historico
    ]
    return max(similaridades) > SIMILARIDADE_THRESHOLD


def enviar_notificacao(classe_id: int, objeto_id: int) -> None:
    """Emite alerta visual no console para objetos detectados"""
    nome_classe = CONFIG_CLASSES[classe_id]["nome"].upper()
    print(f"\033[91m[ALERTA] {nome_classe} detectado! (ID: {objeto_id})\033[0m")


# ====================================================
# MÓDULO DE PROCESSAMENTO DE IMAGEM
# ====================================================
def processar_imagem(caminho_imagem: str) -> Tuple[np.ndarray, str]:
    """Processa imagem estática e retorna resultado anotado"""
    modelo = inicializar_modelo()
    frame = cv2.imread(caminho_imagem)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = Image.open(caminho_imagem)
    resultados = modelo.predict(frame, verbose=False)

    mensagem = "Nenhum objeto detectado"
    if len(resultados[0].boxes.cls) > 0:
        mensagem = "Objeto(s) perigoso(s) detectado(s)!"
        frame = resultados[0].plot()

    return frame, mensagem


# ====================================================
# MÓDULO DE PROCESSAMENTO DE VÍDEO/WEBCAM
# ====================================================
def processar_video(fonte_video: Any, webcam: bool = False):
    modelo = inicializar_modelo()
    #print(f"Modelo: {modelo.model}") # Imprimir a estrutura completa do modelo

    cap = cv2.VideoCapture(FONTE_WEBCAM if webcam else fonte_video)

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            break

        resultados_detecção = modelo.predict(frame, verbose=False) # Detecção (sem rastreamento)

        if resultados_detecção and resultados_detecção[0].boxes: # Verificar se detecções foram feitas
            caixas = resultados_detecção[0].boxes.xyxy.cpu().numpy()
            for caixa in caixas: # Iterar sobre as caixas detectadas
                embedding = extrair_embedding(modelo, frame, caixa) # Chamar extrair_embedding
                print(f"Shape do embedding extraído: {embedding.shape}") # Imprimir shape do embedding
                print(f"Embedding (primeiros 10 valores): {embedding[:10]}") # Imprimir alguns valores do embedding (para verificar se não são zeros)

        frame_anotado = resultados_detecção[0].plot() if resultados_detecção and resultados_detecção[0].plot() is not None else frame # Anotar o frame (se houver detecções)
        yield frame_anotado, "" # Retornar frame anotado (sem mensagem de status)

    cap.release()

def processar_video_completa(fonte_video: Any, webcam: bool = False):
    """Processa fluxo de vídeo com 4 camadas de verificação"""
    modelo = inicializar_modelo()

    # Estado global para rastreamento
    estado = {
        "ids_notificados": defaultdict(dict),  # {classe: {id: último_tempo}}
        "deteccoes_recentes": defaultdict(list),  # {classe: [(caixa, tempo)]}
        "historico_embeddings": defaultdict(lambda: deque(maxlen=HISTORICO_EMBEDDINGS))
    }

    cap = cv2.VideoCapture(FONTE_WEBCAM if webcam else fonte_video)

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            break

        print(f"Shape do frame: {frame.shape}") 
        
        tempo_atual = time.time()
        resultados = modelo.track(frame, persist=True, verbose=False)
        #resultados = modelo.track(frame, persist=True, verbose=False, tracker='bytetrack.yaml') # Ou 'botsort.yaml'
        mensagem_status = ""

        if resultados[0].boxes.id is not None:
            caixas = resultados[0].boxes.xyxy.cpu().numpy()
            ids = resultados[0].boxes.id.cpu().numpy().astype(int)
            classes = resultados[0].boxes.cls.cpu().numpy().astype(int)

            for caixa, obj_id, classe in zip(caixas, ids, classes):
                if classe not in CONFIG_CLASSES:
                    continue

                # 1ª Verificação: Cooldown por ID
                ultima_notificacao = estado["ids_notificados"][classe].get(obj_id, 0)
                if (tempo_atual - ultima_notificacao) < CONFIG_CLASSES[classe]["cooldown"]:
                    continue

                # 2ª Verificação: Sobreposição Espacial
                sobreposicao = any(
                    calcular_iou(caixa, caixa_antiga) > IOU_THRESHOLD
                    and (tempo_atual - tempo_antigo) < CONFIG_CLASSES[classe]["cooldown"]
                    for caixa_antiga, tempo_antigo in estado["deteccoes_recentes"][classe]
                )

                # 3ª Verificação: Similaridade de Embeddings
                embedding = extrair_embedding(modelo, frame, caixa)
                similaridade = verificar_similaridade(
                    embedding,
                    estado["historico_embeddings"][obj_id]
                )

                # 4ª Verificação: Decisão Final
                if not sobreposicao and not similaridade:
                #if not sobreposicao:
                    enviar_notificacao(classe, obj_id)
                    mensagem_status = f"{CONFIG_CLASSES[classe]['nome'].upper()} detectado!"

                    # Atualizar estado
                    estado["ids_notificados"][classe][obj_id] = tempo_atual
                    estado["deteccoes_recentes"][classe].append((caixa, tempo_atual))
                    estado["historico_embeddings"][obj_id].append(embedding)

        # Manutenção do estado
        for classe in CONFIG_CLASSES:
            # Limpar IDs inativos
            estado["ids_notificados"][classe] = {
                id: tempo for id, tempo in estado["ids_notificados"][classe].items()
                if (tempo_atual - tempo) < CONFIG_CLASSES[classe]["cooldown"]
            }

            # Limpar detecções antigas
            estado["deteccoes_recentes"][classe] = [
                (c, t) for c, t in estado["deteccoes_recentes"][classe]
                if (tempo_atual - t) < CONFIG_CLASSES[classe]["cooldown"]
            ]

        # Gerar saída
        frame_anotado = resultados[0].plot()
        yield frame_anotado, mensagem_status

    cap.release()


# ====================================================
# INTERFACE GRÁFICA (GRADIO)
# ====================================================
with gr.Blocks(title="Sistema de Segurança Avançado") as interface:
    gr.Markdown("# 🔪🔫 Sistema de Detecção de Objetos Perigosos")

    with gr.Tabs():
        # Aba de Imagem
        with gr.Tab("📷 Imagem"):
            entrada_imagem = gr.Image(type="filepath", label="Carregar Imagem", sources=['upload'])
            botao_imagem = gr.Button("Analisar")
            saida_imagem = gr.Image(label="Resultado")
            texto_imagem = gr.Textbox(label="Status da Detecção")

        # Aba de Vídeo
        with gr.Tab("🎥 Vídeo"):
            entrada_video = gr.Video(label="Carregar Vídeo", sources=['upload'])
            botao_video = gr.Button("Iniciar Análise")
            saida_video = gr.Image(label="Visualização em Tempo Real")
            texto_video = gr.Textbox(label="Alertas de Segurança")

        # Aba de Webcam
        #with gr.Tab("🌐 Webcam"):
        #    entrada_webcam = gr.Webcam(label="Transmissão ao Vivo")
        #    botao_webcam = gr.Button("Iniciar Monitoramento")
        #    saida_webcam = gr.Image(label="Visualização")
        #    texto_webcam = gr.Textbox(label="Alertas")

    # Conexões de Eventos
    botao_imagem.click(
        processar_imagem,
        inputs=entrada_imagem,
        outputs=[saida_imagem, texto_imagem]
    )

    botao_video.click(
        processar_video,
        inputs=entrada_video,
        outputs=[saida_video, texto_video]
    )

    #botao_webcam.click(
    #    lambda: processar_video(None, True),
    #    outputs=[saida_webcam, texto_webcam]
    #)

interface.launch(share=False)