import cv2
from PIL import Image
import time
import torch
import numpy as np
import streamlit as st  # Alteração: Importar Streamlit
from collections import defaultdict, deque
from ultralytics import YOLO
from typing import Tuple, Any

# ====================================================
# CONFIGURAÇÕES GLOBAIS
# ====================================================
CONFIG_CLASSES = {
    0: {"nome": "cortante", "cooldown": 30, "cor": (0, 0, 255)}
}
#CONFIG_CLASSES = {
#    0: {"nome": "knife", "cooldown": 30, "cor": (0, 0, 255)},
#    1: {"nome": "scissors", "cooldown": 30, "cor": (255, 0, 0)},
#}
IOU_THRESHOLD = 0.3
SIMILARIDADE_THRESHOLD = 0.85  # Similaridade mínima entre embeddings
HISTORICO_EMBEDDINGS = 5  # Número de embeddings armazenados por ID
MODELO_CAMINHO = "./runs/detect/train8/weights/best.pt"
#MODELO_CAMINHO = "yolov8m-worldv2.pt"
FONTE_WEBCAM = 0  # 0 para webcam padrão

# Carregar modelos separados para detecção/rastreamento e embeddings
MODELO_DETECCAO = YOLO(MODELO_CAMINHO)
MODELO_EMBEDDING = YOLO(MODELO_CAMINHO)
#MODELO_DETECCAO.set_classes(["knife", "scissors"])
#MODELO_EMBEDDING.set_classes(["knife", "scissors"])
# MODELO_DETECCAO.set_classes(["sharp object"])
# MODELO_EMBEDDING.set_classes(["sharp object"])


# ====================================================
# FUNÇÕES AUXILIARES
# ====================================================
def inicializar_modelo() -> YOLO:
    """Retorna o modelo YOLO pré-carregado para detecção"""
    return MODELO_DETECCAO  # Retorna o modelo de detecção pré-carregado


def calcular_iou(caixa1: np.ndarray, caixa2: np.ndarray) -> float:
    """Calcula a intersecção sobre união entre duas caixas delimitadoras"""
    x1_1, y1_1, x2_1, y2_1 = caixa1
    x1_2, y1_2, x2_2, y2_2 = caixa2

    # Calcula área de interseção
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_2, y2_2)

    intersecao = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_total = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - intersecao

    return intersecao / area_total if area_total != 0 else 0

    
def extrair_embedding(modelo_embedding: YOLO, frame: np.ndarray, caixa: np.ndarray) -> torch.Tensor: # Usar modelo_embedding como argumento
    """Extrai vetor de características da região do objeto detectado"""
    x1, y1, x2, y2 = map(int, caixa)
    roi = frame[y1:y2, x1:x2]

    # Fallback para ROIs inválidas
    if roi.size == 0:
        return torch.zeros(512)

    # Extrai embeddings usando MODELO_EMBEDDING e embed=[10] (ou outro índice de camada válido)
    resultados = MODELO_EMBEDDING.predict(roi, verbose=False, augment=False, embed=[10]) # Usar MODELO_EMBEDDING aqui

    if isinstance(resultados[0], torch.Tensor): # Verificar se resultados[0] é um Tensor
        embedding = resultados[0] # Usar resultados[0] diretamente como embedding
        #print(f"Embeddings extraídos com sucesso da camada [10]:") # Mensagem de sucesso (opcional)
        #print(f"Shape do embedding: {embedding.shape}") # Imprimir shape do embedding (opcional)
        return embedding.flatten()
    else:
        #print(f"Falha ao extrair embeddings com camada [10] ou resultados não são Tensor.") # Mensagem de falha (opcional)
        return torch.zeros(512) # Retornar zeros como fallback


def verificar_similaridade(embedding_atual: torch.Tensor, historico: deque) -> bool:
    """Compara embedding atual com histórico usando similaridade de cosseno"""

    #print("Inicio verificar_similaridade")

    if not historico:
        return False

    similaridades = [
        torch.nn.functional.cosine_similarity(embedding_atual, emb, dim=0).item()
        for emb in historico
    ]

    #print("Fim verificar_similaridade")

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
    resultados = modelo.predict(frame, verbose=False)

    mensagem = "Nenhum objeto detectado"
    if len(resultados[0].boxes.cls) > 0:
        mensagem = "Objeto(s) perigoso(s) detectado(s)!"
        frame_anotado = resultados[0].plot()
        return frame_anotado, mensagem
    else:
        return frame, mensagem


# ====================================================
# MÓDULO DE PROCESSAMENTO DE VÍDEO/WEBCAM
# ====================================================
def processar_video(fonte_video: Any, webcam: bool = False):
    """Processa fluxo de vídeo com 4 camadas de verificação"""
    modelo_detecao = inicializar_modelo()  # Usar modelo_detecao para detecção/rastreamento

    # Estado global para rastreamento
    estado = {
        "ids_notificados": defaultdict(dict),  # {classe: {id: último_tempo}}
        "deteccoes_recentes": defaultdict(list),  # {classe: [(caixa, tempo)]}
        "historico_embeddings": defaultdict(lambda: deque(maxlen=HISTORICO_EMBEDDINGS)),
    }

    cap = cv2.VideoCapture(FONTE_WEBCAM if webcam else fonte_video)
    print("Ligou webcam...")

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            break

        tempo_atual = time.time()
        resultados = modelo_detecao.track(
            frame, persist=True, verbose=False, tracker="bytetrack.yaml"
        )  # Usar ByteTrack e modelo_detecao

        mensagem_status = ""

        if (
            resultados
            and isinstance(resultados, list)
            and resultados[0] is not None
            and hasattr(resultados[0], "boxes")
        ):  # Verificação adicional antes de acessar .boxes
            if (
                resultados[0].boxes is not None and resultados[0].boxes.id is not None
            ):  # Verificação adicional para boxes e boxes.id
                caixas = resultados[0].boxes.xyxy.cpu().numpy()
                ids = resultados[0].boxes.id.cpu().numpy().astype(int)
                classes = resultados[0].boxes.cls.cpu().numpy().astype(int)

                # ... (o resto do seu código de processamento dentro do loop for) ...
                for caixa, obj_id, classe in zip(caixas, ids, classes):
                    if classe not in CONFIG_CLASSES:
                        continue

                    # 1ª Verificação: Cooldown por ID
                    #print("Primeira verificacao...")
                    ultima_notificacao = estado["ids_notificados"][classe].get(obj_id, 0)
                    if ((tempo_atual - ultima_notificacao) < CONFIG_CLASSES[classe]["cooldown"]):
                        continue

                    # 2ª Verificação: Sobreposição Espacial
                    #print("Segunda verificacao...")
                    sobreposicao = any(
                        calcular_iou(caixa, caixa_antiga) > IOU_THRESHOLD
                        and (tempo_atual - tempo_antigo)
                        < CONFIG_CLASSES[classe]["cooldown"]
                        for caixa_antiga, tempo_antigo in estado["deteccoes_recentes"][
                            classe
                        ]
                    )

                    # 3ª Verificação: Similaridade de Embeddings
                    #print("Terceira verificacao...")
                    embedding = extrair_embedding(
                        MODELO_EMBEDDING, frame, caixa
                    )  # Usar MODELO_EMBEDDING para extrair embeddings
                    similaridade = verificar_similaridade(
                        embedding, estado["historico_embeddings"][obj_id]
                    )

                    # 4ª Verificação: Decisão Final
                    #print("Quarta verificacao...")
                    if not sobreposicao and not similaridade:
                    #if not sobreposicao:
                        enviar_notificacao(classe, obj_id)
                        mensagem_status = (
                            f"{CONFIG_CLASSES[classe]['nome'].upper()} detectado!"
                        )

                        # Atualizar estado
                        estado["ids_notificados"][classe][obj_id] = tempo_atual
                        estado["deteccoes_recentes"][classe].append((caixa, tempo_atual))
                        #estado["historico_embeddings"][obj_id].append(embedding)
            else:
                mensagem_status = "Nenhum objeto perigoso detectado."

        else:
            mensagem_status = "Nenhum objeto perigoso detectado."

        # Manutenção do estado
        for classe in CONFIG_CLASSES:
            # Limpar IDs inativos
            estado["ids_notificados"][classe] = {
                id: tempo
                for id, tempo in estado["ids_notificados"][classe].items()
                if (tempo_atual - tempo) < CONFIG_CLASSES[classe]["cooldown"]
            }

            # Limpar detecções antigas
            estado["deteccoes_recentes"][classe] = [
                (c, t)
                for c, t in estado["deteccoes_recentes"][classe]
                if (tempo_atual - t) < CONFIG_CLASSES[classe]["cooldown"]
            ]

        frame_anotado = (
            resultados[0].plot()
            if resultados
            and isinstance(resultados, list)
            and resultados[0] is not None
            and hasattr(resultados[0], "plot")
            and resultados[0].plot() is not None
            else frame
        )  # Anotar o frame (se resultados e plot() não forem None)
        frame_anotado_rgb = cv2.cvtColor(
            frame_anotado, cv2.COLOR_BGR2RGB
        )  # Converter para RGB para Streamlit
        yield frame_anotado_rgb, mensagem_status

    cap.release()


# ====================================================
# INTERFACE GRÁFICA (STREAMLIT)
# ====================================================
st.set_page_config(page_title="Sistema de Segurança Avançado")  # Alteração: page_title
st.markdown("# 🔪🔫 Sistema de Detecção de Objetos Perigosos")  # Alteração: Markdown

tab_imagem, tab_video, tab_webcam = st.tabs(
    ["📷 Imagem", "🎥 Vídeo", "🌐 Webcam (Tempo Real)"]
)  # Alteração: Tabs - Adicionada aba Webcam

# Aba de Imagem
with tab_imagem:
    entrada_imagem = st.file_uploader(
        "Carregar Imagem", type=["png", "jpg", "jpeg"]
    )  # Alteração: File uploader
    botao_imagem = st.button("Analisar Imagem")  # Alteração: Botão
    saida_imagem = st.empty()  # Alteração: Placeholder para imagem
    texto_imagem = st.empty()  # Alteração: Placeholder para texto

    if botao_imagem:
        if entrada_imagem is not None:
            caminho_temporario_imagem = "temp_image.jpg"
            with open(caminho_temporario_imagem, "wb") as f:
                f.write(entrada_imagem.read())
            frame_processado, mensagem_detecao = processar_imagem(
                caminho_temporario_imagem
            )
            saida_imagem.image(
                frame_processado,
                caption="Resultado da Análise",
                use_container_width=True,
            )  # Alteração: Exibir imagem, use_container_width
            texto_imagem.text(mensagem_detecao)  # Alteração: Exibir texto
        else:
            texto_imagem.text("Por favor, carregue uma imagem.")

# Aba de Vídeo
with tab_video:
    entrada_video = st.file_uploader(
        "Carregar Vídeo", type=["mp4", "avi", "mov"]
    )  # Alteração: Video uploader
    botao_video = st.button("Iniciar Análise de Vídeo")  # Alteração: Botão
    saida_video = st.empty()  # Alteração: Placeholder para vídeo/imagens em tempo real
    texto_video = st.empty()  # Alteração: Placeholder para texto de alerta

    if botao_video:
        if entrada_video is not None:
            caminho_temporario_video = "temp_video." + entrada_video.name.split(".")[-1]
            with open(caminho_temporario_video, "wb") as f:
                f.write(entrada_video.read())

            video_stream = processar_video(caminho_temporario_video)
            for frame_anotado, mensagem_status in video_stream:
                saida_video.image(
                    frame_anotado,
                    channels="RGB",
                    caption="Visualização em Tempo Real",
                    use_container_width=True,
                )  # Alteração: Exibir frame de vídeo, use_container_width
                texto_video.text(mensagem_status)  # Alteração: Exibir texto de alerta
        else:
            texto_video.text("Por favor, carregue um vídeo.")

# Aba da Webcam (Tempo Real)
with tab_webcam:
    status_webcam = st.empty()  # Placeholder para status da webcam
    saida_webcam = st.empty()  # Placeholder para vídeo da webcam
    texto_webcam = st.empty()  # Placeholder para alertas da webcam
    botao_iniciar_webcam = st.button("Iniciar Webcam")
    botao_parar_webcam = st.button("Parar Webcam")
    webcam_processando = False  # Variável de controle

    if botao_iniciar_webcam:
        webcam_processando = True
        video_stream_webcam = processar_video(None, webcam=True)
        status_webcam.text("Webcam iniciada. Detectando...")
        while webcam_processando:
            try:
                frame_webcam, mensagem_webcam = next(video_stream_webcam)
                saida_webcam.image(
                    frame_webcam,
                    channels="RGB",
                    caption="Webcam em Tempo Real",
                    use_container_width=True,
                )  # Alteração: Exibir frame da webcam, use_container_width
                texto_webcam.text(mensagem_webcam)
            except StopIteration:
                status_webcam.text("Webcam finalizada.")
                webcam_processando = False
                break
            except Exception as e:
                status_webcam.error(f"Erro ao processar webcam: {e}")
                webcam_processando = False
                break

    if botao_parar_webcam:
        webcam_processando = False
        status_webcam.text("Webcam parada.")