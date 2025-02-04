import gradio as gr
from gradio_webrtc import WebRTC
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt") 

#test_image = cv2.imread("./assets/faca.png")   # Substitua pelo caminho real da imagem

def detection(image, conf_threshold=0.3):
    """
    Realiza a detecção de objetos em um frame de vídeo usando YOLOv8.

    Args:
        image (np.ndarray): Imagem do frame de vídeo.
        conf_threshold (float): Limiar de confiança para detecção de objetos.

    Returns:
        np.ndarray: Imagem com os objetos detectados desenhados.
    """
    if image is None:
        print("Imagem recebida é None")
        return image
    print(f"Imagem recebida na função detection. Formato: {image.shape}")
    return image # Retorna a imagem original

with gr.Blocks() as demo:
    image = WebRTC(label="Stream", mode="send-receive", modality="video")
    conf_threshold = gr.Slider(
        label="Confidence Threshold",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        value=0.1,
    )
    image.stream(
        fn=detection,
        inputs=[image, conf_threshold],
        outputs=[image]
    )
    
    # Teste da inferência (apenas para debug)
    test_image = cv2.imread("./assets/faca.png")  # Substitua pelo caminho real da imagem
    if test_image is not None:
        test_results = model(test_image)
        print(f"Tipo de test_results: {type(test_results)}") # Check do tipo
        
        # Acesso correto aos resultados de inferência
        if isinstance(test_results, list) and len(test_results) > 0:
            test_results_obj = test_results[0]  # Pega o objeto Results da lista
            print(f"Quantidade de detecções (teste): {len(test_results_obj.boxes.data)}")  # Corrigido
            if len(test_results_obj.boxes.data) > 0:
                print("Teste do Modelo: Detecção realizada com sucesso!")
                print(f"Exemplo de detecção (teste): {test_results_obj.boxes.data[0]}") # Corrigido
            else:
                print("Teste do Modelo: Nenhuma detecção encontrada.")
        else:
            print("Teste do Modelo: Nenhum resultado encontrado.")
    else:
        print("Teste do Modelo: Não foi possível carregar a imagem de teste.")


if __name__ == "__main__":
    demo.launch()