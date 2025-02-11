# FIAP-Hackathon-Grupo2

\`\`\`markdown

# Sistema de Detecção de Objetos Cortantes com YOLO e Streamlit

Este projeto implementa um sistema de detecção de objetos cortantes (como facas e armas de fogo) utilizando o modelo YOLOv8 e a biblioteca Streamlit para criar uma interface web interativa. O sistema é capaz de processar imagens estáticas, vídeos e fluxos de vídeo em tempo real da webcam, emitindo alertas visuais e por e-mail quando objetos perigosos são detectados.

## 🚀 Funcionalidades

  * **Detecção de Objetos Cortantes:** Identifica objetos pré-definidos como "cortantes" em imagens e vídeos.
  * **Rastreamento de Objetos:** Utiliza o ByteTrack para rastrear objetos em vídeos, melhorando a precisão da detecção ao longo do tempo.
  * **Interface Web Interativa:**  Desenvolvido com Streamlit, oferece uma interface simples e intuitiva para carregar imagens, vídeos e iniciar a detecção via webcam.
  * **Alertas Visuais:**  Destaca visualmente os objetos detectados na interface e emite alertas no console.
  * **Notificações por E-mail:**  Envia e-mails de alerta quando um objeto perigoso é detectado (configurável).
  * **Múltiplas Camadas de Verificação:** Implementa quatro camadas de verificação (Cooldown por ID, Sobreposição Espacial, Similaridade de Embeddings e Confiança) para reduzir falsos positivos.
  * **Histórico de Embeddings:** Mantém um histórico de embeddings para cada objeto rastreado, melhorando a identificação consistente ao longo do tempo.

## ⚙️ Pré-requisitos

Antes de executar o código, você precisará ter instalado os seguintes softwares e bibliotecas:

  * **Python:** Versão 3.7 ou superior ([https://www.python.org/downloads/](https://www.google.com/url?sa=E&source=gmail&q=https://www.python.org/downloads/))
  * **pip:** Gerenciador de pacotes do Python (geralmente incluído com o Python)
  * **OpenCV (cv2):** Biblioteca para processamento de imagens ([https://pypi.org/project/opencv-python/](https://www.google.com/url?sa=E&source=gmail&q=https://pypi.org/project/opencv-python/))
  * **PyTorch:** Framework de aprendizado de máquina ([https://pytorch.org/get-started/locally/](https://www.google.com/url?sa=E&source=gmail&q=https://pytorch.org/get-started/locally/))
  * **torchvision:** Pacote complementar ao PyTorch para visão computacional ([https://pytorch.org/vision/stable/](https://www.google.com/url?sa=E&source=gmail&q=https://pytorch.org/vision/stable/))
  * **NumPy:** Biblioteca para computação numérica ([https://numpy.org/install/](https://www.google.com/url?sa=E&source=gmail&q=https://numpy.org/install/))
  * **Streamlit:** Framework para criar aplicativos web interativos ([https://streamlit.io/](https://www.google.com/url?sa=E&source=gmail&q=https://streamlit.io/))
  * **Ultralytics (YOLO11):** Biblioteca para utilizar modelos YOLO ([https://github.com/ultralytics/ultralytics](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/ultralytics/ultralytics))
  * **python-dotenv:** Para carregar variáveis de ambiente de um arquivo `.env` ([https://pypi.org/project/python-dotenv/](https://www.google.com/url?sa=E&source=gmail&q=https://pypi.org/project/python-dotenv/))

## 🛠️ Instalação

Siga os passos abaixo para configurar o ambiente e executar o projeto:

1.  **Clone o repositório:**

    ```bash
    git clone [URL_DO_REPOSITORIO]
    cd [NOME_DO_REPOSITORIO]
    ```

    *(Substitua `[URL_DO_REPOSITORIO]` pela URL do repositório no GitHub e `[NOME_DO_REPOSITORIO]` pelo nome da pasta que será criada).*

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python -m venv venv
    ```

      * Para ativar o ambiente virtual:\*
          * **No Linux/macOS:**
            ```bash
            source venv/bin/activate
            ```
          * **No Windows:**
            ```bash
            venv\Scripts\activate
            ```

3.  **Instale as dependências Python:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as variáveis de ambiente para e-mail:**

      * Crie um arquivo chamado `.env` na raiz do projeto.

      * Adicione as seguintes variáveis com suas respectivas informações de e-mail:

        ```
        EMAIL_REMETENTE="[endereço de e-mail removido]"
        EMAIL_DESTINATARIO="[endereço de e-mail removido]"
        EMAIL_SENHA="sua_senha_de_email"
        ```

          * **Importante:** Para utilizar o Gmail, pode ser necessário configurar o acesso a "aplicativos menos seguros" na sua conta Google ou gerar uma "Senha de app" se a autenticação de dois fatores estiver ativada. **Tenha cuidado com a segurança das suas credenciais de e-mail.**

5.  **Arquivos de Modelo YOLO:**

      * Certifique-se de ter o arquivo de modelo YOLO treinado (`best.pt`) e coloque-o na pasta `modelo` dentro do diretório do projeto. Se a pasta `modelo` não existir, crie-a.
      * Caso não possua um modelo treinado, você precisará treinar um modelo YOLO11 para detecção de objetos cortantes ou utilizar um modelo pré-treinado adequado.

## 🚀 Execução

Para executar o sistema, siga as instruções abaixo:

1.  **Execute o aplicativo Streamlit:**

    ```bash
    streamlit run app.py
    ```

    *(Certifique-se de que o arquivo principal do código Python se chama `app.py` ou ajuste o comando conforme o nome do seu arquivo).*

2.  **Acesse a interface web:**

      * O Streamlit irá exibir um URL no terminal (geralmente `http://localhost:8501`). Abra este URL no seu navegador web.

3.  **Utilize as abas na interface web:**

      * **📷 Imagem:**

          * Carregue uma imagem estática (PNG, JPG, JPEG) utilizando o botão "Carregar Imagem".
          * Clique em "Analisar Imagem" para processar a imagem.
          * O resultado da análise (imagem anotada e mensagem de detecção) será exibido abaixo.

      * **🎥 Vídeo:**

          * Carregue um arquivo de vídeo (MP4, AVI, MOV) utilizando o botão "Carregar Vídeo".
          * Clique em "Iniciar Análise de Vídeo" para começar a análise.
          * O vídeo será processado frame a frame, e a visualização em tempo real com as detecções e mensagens de status serão exibidas.

      * **🌐 Webcam (Tempo Real):**

          * Clique em "Iniciar Webcam" para iniciar a detecção em tempo real utilizando a webcam padrão do seu computador.
          * A visualização da webcam com as detecções e mensagens de status serão exibidas.
          * Clique em "Parar Webcam" para interromper o processamento da webcam.

## ⚙️ Configurações Adicionais

Você pode ajustar as seguintes configurações diretamente no código `app.py`:

  * **Configurações:** Recomenda-se fortemente que o sistema primeiro seja testado sem enviar e-mails, pois dependendo dos valores configurados nas variáveis abaixo, o sistema pode enviar diversos e-mails e com isso o google bloquear sua conta.
  * **`CONFIG_CLASSES`:** Define as classes de objetos a serem detectadas, o tempo de cooldown para notificações e a cor das caixas delimitadoras.
  * **`CONFIANCA_MINIMA`:**  Define o nível mínimo de confiança para que uma detecção seja considerada válida.
  * **`IOU_THRESHOLD`:** Limiar de Intersecção sobre União (IoU) para determinar sobreposição espacial entre detecções.
  * **`SIMILARIDADE_THRESHOLD`:** Limiar de similaridade de cosseno para comparação de embeddings.
  * **`HISTORICO_EMBEDDINGS`:** Número de embeddings históricos armazenados por ID de objeto para verificação de similaridade.
  * **`MODELO_CAMINHO`:** Caminho para o arquivo do modelo YOLO (`best.pt`).
  * **`FONTE_WEBCAM`:** Índice da webcam a ser utilizada (geralmente `0` para a webcam padrão).

## 📧 Configuração de Notificações por E-mail (Opcional)

Para habilitar as notificações por e-mail, certifique-se de que:

1.  **Variável:** Você configurou corretamente a variável `ENVIAR_EMAIL` com valor `True`.
2.  **Variáveis de Ambiente:** Você configurou corretamente as variáveis `EMAIL_REMETENTE`, `EMAIL_DESTINATARIO` e `EMAIL_SENHA` no arquivo `.env`.
3.  **Permissões de E-mail:** Se estiver usando o Gmail, verifique as configurações de segurança da sua conta Google e habilite o acesso para "aplicativos menos seguros" ou configure uma "Senha de app".


## 📂 Estrutura de Diretórios

A estrutura de diretórios esperada para o projeto é a seguinte:

```
nome-do-projeto/
├── app.py           # Arquivo principal do aplicativo Streamlit
├── modelo/          # Pasta para arquivos de modelo
│   └── best.pt      # Arquivo do modelo YOLO treinado (exemplo)
├── .env             # Arquivo para variáveis de ambiente (credenciais de e-mail)
├── requirements.txt # Arquivo com as dependências Python
└── README.md        # Este arquivo README
```

## 📝 Observações

  * **Desempenho:** O desempenho do sistema pode variar dependendo do hardware, da complexidade do modelo YOLO e da resolução do vídeo/imagem.
  * **Precisão:** A precisão da detecção depende da qualidade do modelo YOLO treinado e dos dados de treinamento utilizados.
  * **Segurança:** Tenha cuidado ao configurar as credenciais de e-mail e ao habilitar recursos de segurança menos rigorosos na sua conta de e-mail.

## 🤝 Contribuições

Contribuições são bem-vindas\! Sinta-se à vontade para abrir issues e pull requests para melhorias, correções de bugs ou novas funcionalidades.

-----

**Este README.md fornece instruções detalhadas para executar o código do sistema de detecção de objetos cortantes. Certifique-se de seguir todos os passos de instalação e configuração para garantir o funcionamento correto do sistema.**

```
```