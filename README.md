## Leitor de Libras (A–Z) – Instruções de Execução

Projeto para treinar um classificador de letras A–Z a partir de imagens e realizar inferência ao vivo via webcam com MediaPipe + ResNet18.

### Requisitos

- Python 3.10+ (recomendado 3.11)
- Windows 10/11 (testado) – funciona também em Linux/macOS com pequenas adaptações
- Webcam para a inferência ao vivo

### Estrutura do projeto

```
leitor-libras/
  artifacts/            # onde ficam modelo e arquivos gerados
  data/sign_es/         # dataset organizado por pastas A/ ... Z/
  scripts/
    train_sign_alphabet.py
    live_inference.py
    plot_confusion.py
  venv/                 # ambiente virtual (opcional, já presente)
```

### Preparação do ambiente

No Windows PowerShell (pwsh):

1. Entrar na pasta do projeto

```powershell
cd "C:\Users\Cleiton\Desktop\pos-ia\leitor-libras"
```

2. Ativar o ambiente virtual existente (recomendado)

```powershell
./venv/Scripts/Activate.ps1
```

Se aparecer erro de política de execução, rode o PowerShell como Administrador e execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. Instalar dependências (CPU)

```powershell
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python mediapipe scikit-learn matplotlib pillow numpy
```

Para GPU (opcional, se tiver CUDA compatível), instale PyTorch conforme o site oficial e pule a linha com `--index-url cpu`.

### Dataset

O script de treino espera o dataset em `data/sign_es/` com subpastas A/ ... Z/ contendo imagens (`.png`, `.jpg`, ...). Exemplo:

```
data/sign_es/
  A/ 0.png, 1.png, ...
  B/ 0.png, 1.png, ...
  ...
  Z/ 0.png, 1.png, ...
```

### 1) Treinar o modelo

Gera `artifacts/best_resnet18.pt`, `artifacts/classes.json` e (após avaliação) `artifacts/confusion_matrix.npy`.

```powershell
python scripts/train_sign_alphabet.py
```

Parâmetros principais (editar no código, se desejar):

- `DATA_DIR`: caminho do dataset (padrão: `data/sign_es`)
- `EPOCHS`, `BATCH_SIZE`, `LR`, `VAL_SPLIT`, `IMG_SIZE`

Notas (Windows): o script já usa `num_workers=0` no `DataLoader` e `multiprocessing.freeze_support()` no main.

### 2) Inferência ao vivo (webcam)

Requer `artifacts/best_resnet18.pt` e `artifacts/classes.json` gerados no treino.

```powershell
python scripts/live_inference.py
```

Dicas:

- Se a câmera não abrir, troque o índice no `cv2.VideoCapture(0, ...)` para 1 ou 2.
- Pressione ESC para sair.
- A janela mostra o bounding box da mão, a letra prevista e a predição suavizada por votação.

### 3) Visualizar a matriz de confusão

Após o treino, para ver a matriz salva em `artifacts/confusion_matrix.npy`:

```powershell
python scripts/plot_confusion.py
```

### Solução de problemas

- Erro ao importar `mediapipe` no Windows: garanta `pip install mediapipe` após ativar a `venv`.
- GPU não utilizada: verifique se a instalação do PyTorch está com CUDA e se `torch.cuda.is_available()` retorna True.
- Webcam não detectada: tente outro índice (1/2) e feche outros apps que usem a câmera.
- Falta de arquivos em `artifacts/`: rode primeiro o treino para gerar `best_resnet18.pt` e `classes.json`.

### Licença

Uso acadêmico/experimental. Ajuste conforme sua necessidade.
