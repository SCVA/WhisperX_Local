# WhisperX Local (GPU)

Transcripción local de audio en español usando GPU NVIDIA.
Requiere python 3.13

## Archivos principales

- `whisperx_local.py`: script principal.
- `requirements-gpu.txt`: dependencias Python.
- `outputs/`: carpeta de salida.

## Instalación (Python global, sin venv)

```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128
python -m pip install -r requirements-gpu.txt
```

## Uso rápido

### 1) TXT sin hablantes (más rápido)

```powershell
python whisperx_local.py --audio-file "claustro26mayo2026.wav" --output-dir outputs --model large-v3 --language es --preset fast --compute-type float16 --output-format txt --log-progress
```

### 2) DOCX con hablantes (diarización)

```powershell
python whisperx_local.py --audio-file "claustro26mayo2026.wav" --output-dir outputs --model large-v3 --language es --preset fast --compute-type float16 --diarize --hf-token "TU_TOKEN_HF" --diarization-device cuda --output-format docx --docx-title "Relatoria Claustro" --log-progress
```

## Opciones útiles

- Número fijo de hablantes:

```powershell
--num-speakers 6
```

- Rango de hablantes:

```powershell
--min-speakers 4 --max-speakers 10
```
