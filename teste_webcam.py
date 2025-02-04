import gradio as gr
import cv2
import numpy as np

def processar_imagem(imagem_np):
    print(type(imagem_np))
    # imagem_np ja eh um array numpy
    if imagem_np is None:
        print("Erro: Nenhuma imagem capturada da webcam.")
        return None  # Ou retornar uma imagem padrão, se desejar

    if not isinstance(imagem_np, np.ndarray):
        print("Erro: imagem não é um array numpy")
        return None

    # imagem_np já é um array numpy que você pode usar com cv2
    imagem_cv2 = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2BGR) # Converter RGB para BGR
    cv2.imshow("Imagem da Webcam", imagem_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return imagem_np # Retornar uma imagem para o gradio mostrar

entrada_webcam = gr.Image(label="Webcam", sources=['webcam'], mirror_webcam=False, streaming=True)
iface = gr.Interface(fn=processar_imagem, inputs=entrada_webcam, outputs=entrada_webcam)
iface.launch()