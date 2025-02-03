from ultralytics import YOLO
import cv2

MODELO_CAMINHO = "./runs/detect/train/weights/best.pt"
modelo = YOLO(MODELO_CAMINHO)
FONTE_WEBCAM = 0  # Ou caminho para um vídeo

cap = cv2.VideoCapture(FONTE_WEBCAM)
sucesso, frame = cap.read()
cap.release()

if sucesso:
    resultados_track = modelo.track(frame, persist=True, verbose=True)
    print(f"Type de resultados_track: {type(resultados_track)}")

    if resultados_track: # Verificar se a lista não está vazia
        resultados_0 = resultados_track[0]
        print(f"Type de resultados_track[0]: {type(resultados_0)}")
        try:
            print(f"Type de resultados_track[0].boxes: {type(resultados_0.boxes)}") # Tentar acessar .boxes
        except AttributeError as e:
            print(f"Erro ao acessar .boxes: {e}") # Imprimir erro se falhar
    else:
        print("resultados_track está vazio.")
else:
    print("Falha ao ler o frame da fonte de vídeo.")