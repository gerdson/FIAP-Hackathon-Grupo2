import cv2
import time
import torch
import numpy as np
import streamlit as st
from collections import defaultdict, deque
from ultralytics import YOLO
from typing import Tuple, Any
import smtplib
import email.message
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

# ====================================================
# CONFIGURAÇÕES GLOBAIS
# ====================================================
CONFIG_CLASSES = {
    0: {"nome": "cortante", "cooldown": 30, "cor": (0, 0, 255)}
}

CONFIANCA_MINIMA = 0.3
SIMILARIDADE_THRESHOLD = 0.95  # Similaridade mínima entre embeddings
HISTORICO_EMBEDDINGS = 10  # Número de embeddings armazenados por ID
MODELO_CAMINHO = "./modelo/best.pt"
FONTE_WEBCAM = 0  # 0 para webcam padrão

# Carregar modelos separados para detecção/rastreamento e embeddings
MODELO_DETECCAO = YOLO(MODELO_CAMINHO)
MODELO_EMBEDDING = YOLO(MODELO_CAMINHO)

#imprime camadas do modelo
#for nome, modulo in MODELO_EMBEDDING.model.named_modules():
#    print(f"Camada: {nome}, Tipo: {type(modulo)}")


# Dados notificacao por email
ENVIAR_EMAIL = False
EMAIL_REMETENTE = os.environ.get('EMAIL_REMETENTE')
EMAIL_DESTINATARIO = os.environ.get('EMAIL_DESTINATARIO')
EMAIL_SENHA = os.environ.get('EMAIL_SENHA')


# ====================================================
# FUNÇÕES AUXILIARES
# ====================================================
def inicializar_modelo() -> YOLO:
    """Retorna o modelo YOLO pré-carregado para detecção"""
    return MODELO_DETECCAO  # Retorna o modelo de detecção pré-carregado


def extrair_embedding(modelo_embedding: YOLO, frame: np.ndarray, caixa: np.ndarray) -> torch.Tensor: # Usar modelo_embedding como argumento
    """Extrai vetor de embeddings da região do objeto detectado"""
    x1, y1, x2, y2 = map(int, caixa)
    roi = frame[y1:y2, x1:x2]

    # Fallback para ROIs inválidas
    if roi.size == 0:
        return torch.zeros(512)

    # Extrai embeddings usando índice de camada anterior da saida)
    resultados = MODELO_EMBEDDING.predict(roi, verbose=False, augment=False, embed=[9, 10, 19, 22]) # Usar MODELO_EMBEDDING aqui
    
    if isinstance(resultados[0], torch.Tensor): # Verificar se resultados[0] é um Tensor
        embedding = resultados[0] # Usar resultados[0] diretamente como embedding
        return embedding.flatten()
    else:
        return torch.zeros(512) # Retornar zeros como fallback


def verificar_similaridade(embedding_atual: torch.Tensor, historico: deque) -> bool:
    """Compara embedding atual com histórico usando similaridade de cosseno"""

    if not historico:
        return False

    similaridades = [
        torch.nn.functional.cosine_similarity(embedding_atual, emb, dim=0).item()
        for emb in historico
    ]

    soma = sum(similaridades) # Soma todos os elementos da lista
    quantidade = len(similaridades) # Obtém o número de elementos na lista
    media = soma / quantidade # Calcula a média

    print(f"media das similaridades: {media}")
    #print(f"similaridade: {max(similaridades)}")

    #return max(similaridades) > SIMILARIDADE_THRESHOLD
    return media > SIMILARIDADE_THRESHOLD

def enviar_email(nome_classe: str):
    corpo_email = f"[ALERTA] {nome_classe} detectado!"

    msg = email.message.Message()
    msg['Subject'] = "Alerta de Objeto Cortante"
    msg['From'] = EMAIL_REMETENTE
    msg['To'] = EMAIL_DESTINATARIO
    password = EMAIL_SENHA
    msg.add_header('Content-Type', 'text/html')
    msg.set_payload(corpo_email)

    s = smtplib.SMTP('smtp.gmail.com: 587')
    s.starttls()
    # Login e Credenciais para enviar o email
    s.login(msg['From'], password)
    s.sendmail(msg['From'], [msg['To']], msg.as_string().encode('utf-8'))
    print('Email enviado!')


def enviar_notificacao(classe_id: int) -> None:
    """Emite alerta visual no console para objetos detectados"""

    nome_classe = CONFIG_CLASSES[classe_id]["nome"].upper()

    if(ENVIAR_EMAIL):
        enviar_email(nome_classe)

    print(f"\033[91m[ALERTA] {nome_classe} detectado!\033[0m")

# ====================================================
# MÓDULO DE PROCESSAMENTO DE IMAGEM
# ====================================================
def processar_imagem(caminho_imagem: str) -> Tuple[np.ndarray, str]:
    """Processa imagem estática e retorna resultado anotado"""
    modelo = inicializar_modelo()
    resultados = modelo.predict(caminho_imagem, verbose=False)

    mensagem = "Nenhum objeto detectado"
    if len(resultados[0].boxes.cls) > 0:
        mensagem = "Objeto(s) perigoso(s) detectado(s)!"
        frame_anotado = resultados[0].plot()
        frame_anotado = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
        return frame_anotado, mensagem
    else:
        frame = cv2.imread(caminho_imagem)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, mensagem


# ====================================================
# MÓDULO DE PROCESSAMENTO DE VÍDEO/WEBCAM
# ====================================================
def processar_video(fonte_video: Any, webcam: bool = False):
    """Processa fluxo de vídeo com 4 camadas de verificação"""
    modelo_detecao = inicializar_modelo()  # Usar modelo_detecao para detecção

    # Estado global para detecção
    estado = {
        "ultima_notificacao": defaultdict(float),  # {classe: último_tempo}
        "deteccoes_recentes": defaultdict(list),  # {classe: [(caixa, tempo)]}
        "historico_embeddings": defaultdict(lambda: deque(maxlen=HISTORICO_EMBEDDINGS)), # {classe: deque[embeddings]}
    }

    cap = cv2.VideoCapture(FONTE_WEBCAM if webcam else fonte_video)
    
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            break

        tempo_atual = time.time()
        resultados = modelo_detecao.predict(frame, verbose=False) # Usar modelo_detecao para detecção
        mensagem_status = ""

        if (resultados and isinstance(resultados, list)
                and resultados[0] is not None
                and hasattr(resultados[0], "boxes")):
            if (resultados[0].boxes is not None):

                caixas = resultados[0].boxes.xyxy.cpu().numpy()
                classes = resultados[0].boxes.cls.cpu().numpy().astype(int)

                for caixa, classe in zip(caixas, classes):
                    if classe not in CONFIG_CLASSES:
                        continue

                    # Verificação: Cooldown por Classe, Similaridade de Embeddings e Confianca
                    ultima_notificacao = estado["ultima_notificacao"][classe]
                    condicao_cooldown = ((tempo_atual - ultima_notificacao) < CONFIG_CLASSES[classe]["cooldown"])

                    #print(f"caixa: {caixa}") 
                    embedding = extrair_embedding(MODELO_EMBEDDING, frame, caixa)  # Usar MODELO_EMBEDDING para extrair embeddings
                    similaridade = verificar_similaridade(embedding, estado["historico_embeddings"][classe])

                    if (not similaridade and (resultados[0].boxes.conf >= CONFIANCA_MINIMA).all()) or (not condicao_cooldown):
                        enviar_notificacao(classe)
                        mensagem_status = (f"{CONFIG_CLASSES[classe]['nome'].upper()} detectado!")

                        # Atualizar estado
                        estado["ultima_notificacao"][classe] = tempo_atual
                        estado["deteccoes_recentes"][classe].append((caixa, tempo_atual))
                        estado["historico_embeddings"][classe].append(embedding)
                        if len(estado["historico_embeddings"][classe]) > HISTORICO_EMBEDDINGS:
                            estado["historico_embeddings"][classe].popleft() # Mantem o tamanho máximo do histórico
            else:
                mensagem_status = "Nenhum objeto perigoso detectado."

        else:
            mensagem_status = "Nenhum objeto perigoso detectado."

        # Manutenção do estado
        for classe in CONFIG_CLASSES:
            # Limpar detecções antigas
            estado["deteccoes_recentes"][classe] = [
                (c, t)
                for c, t in estado["deteccoes_recentes"][classe]
                if (tempo_atual - t) < CONFIG_CLASSES[classe]["cooldown"]
            ]

        if (resultados[0].boxes.conf >= CONFIANCA_MINIMA).all():

            frame_anotado = (
                resultados[0].plot()
                if resultados
                and isinstance(resultados, list)
                and resultados[0] is not None
                and hasattr(resultados[0], "plot")
                and resultados[0].plot() is not None
                else frame
            )  # Anotar o frame (se resultados e plot() não forem None)
            frame_anotado_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)  # Converter para RGB para Streamlit
            yield frame_anotado_rgb, mensagem_status

        else:
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             yield frame, mensagem_status

    cap.release()


# ====================================================
# INTERFACE GRÁFICA (STREAMLIT)
# ====================================================
st.set_page_config(page_title="Sistema de Segurança Avançado")
st.markdown("# 🔪🔫 Sistema de Detecção de Objetos Cortantes")

tab_imagem, tab_video, tab_webcam = st.tabs(
    ["📷 Imagem", "🎥 Vídeo", "🌐 Webcam (Tempo Real)"]
)

# Aba de Imagem
with tab_imagem:
    entrada_imagem = st.file_uploader(
        "Carregar Imagem", type=["png", "jpg", "jpeg"]
    )
    botao_imagem = st.button("Analisar Imagem")
    saida_imagem = st.empty()
    texto_imagem = st.empty()

    if botao_imagem:
        if entrada_imagem is not None:
            caminho_temporario_imagem = "temp_image.jpg"
            with open(caminho_temporario_imagem, "wb") as f:
                f.write(entrada_imagem.read())
            frame_processado, mensagem_detecao = processar_imagem(caminho_temporario_imagem)
            saida_imagem.image(
                frame_processado,
                caption="Resultado da Análise",
                use_container_width=True,
            )
            texto_imagem.text(mensagem_detecao)
        else:
            texto_imagem.text("Por favor, carregue uma imagem.")

# Aba de Vídeo
with tab_video:
    entrada_video = st.file_uploader(
        "Carregar Vídeo", type=["mp4", "avi", "mov"]
    )
    botao_video = st.button("Iniciar Análise de Vídeo")
    saida_video = st.empty()
    texto_video = st.empty()

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
                )
                texto_video.text(mensagem_status)
        else:
            texto_video.text("Por favor, carregue um vídeo.")

# Aba da Webcam (Tempo Real)
with tab_webcam:
    status_webcam = st.empty()
    saida_webcam = st.empty()
    texto_webcam = st.empty()
    botao_iniciar_webcam = st.button("Iniciar Webcam")
    botao_parar_webcam = st.button("Parar Webcam")
    webcam_processando = False

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
                )
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