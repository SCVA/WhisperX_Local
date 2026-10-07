# WhisperX Local

Transcripcion local de audio en espanol usando GPU NVIDIA o CPU. Requiere Python 3.13.

## Archivos principales

- `whisperx_local.py`: script principal.
- `requirements-gpu.txt`: dependencias Python.
- `outputs/`: carpeta de salida.

## Instalacion (Python global, sin venv)

```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128
python -m pip install -r requirements-gpu.txt
```

## Uso con GPU

### TXT sin hablantes

```powershell
python whisperx_local.py --audio-file "claustro26mayo2026.wav" --output-dir outputs --device cuda --model large-v3 --language es --preset fast --output-format txt --log-progress
```

### DOCX con hablantes

```powershell
python whisperx_local.py --audio-file "claustro26mayo2026.wav" --output-dir outputs --device cuda --model large-v3 --language es --preset fast --diarize --hf-token "TU_TOKEN_HF" --output-format docx --docx-title "Relatoria Claustro" --log-progress
```

## Uso con CPU

### TXT sin hablantes

```powershell
python whisperx_local.py --audio-file "claustro26mayo2026.wav" --output-dir outputs --device cpu --model large-v3 --language es --preset fast --output-format txt --log-progress
```

### DOCX con hablantes

```powershell
python whisperx_local.py --audio-file "claustro26mayo2026.wav" --output-dir outputs --device cpu --model large-v3 --language es --preset fast --diarize --hf-token "TU_TOKEN_HF" --output-format docx --docx-title "Relatoria Claustro" --log-progress
```

## Opciones de hablantes

Numero fijo:

```powershell
--num-speakers 6
```

Rango:

```powershell
--min-speakers 4 --max-speakers 10
```
