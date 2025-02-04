import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam funcionando!")
    cap.release()
else:
    print("Erro: Webcam não detectada")