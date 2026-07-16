# Audio ya estandarizado y listo para procesar.
import subprocess # Para ejecutar comandos externos, como ffmpeg.
from pathlib import Path
import numpy as np
import soundfile as sf

SAMPLE_RATE_OBJETIVO = 16000
CANALES_OBJETIVO = 1


class AudioConversionError(Exception):
    """Se lanza cuando FFmpeg falla al convertir el archivo."""

# Convierte cualquier archivo de audio soportado a WAV PCM mono de 16 kHz usando FFmpeg desde Python (subprocess).
def convertir_a_wav_pcm(ruta_entrada: Path, ruta_salida: Path) -> Path:
    comando = [
        "ffmpeg",
        "-y",  # sobrescribe si ya existe
        "-i", str(ruta_entrada),
        "-ac", str(CANALES_OBJETIVO),          # forzar mono
        "-ar", str(SAMPLE_RATE_OBJETIVO),      # forzar 16 kHz
        "-sample_fmt", "s16",                  # PCM 16 bits
        str(ruta_salida),
    ]

    resultado = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if resultado.returncode != 0:
        error_ffmpeg = resultado.stderr.decode(errors="ignore")
        raise AudioConversionError(
            f"FFmpeg no pudo convertir el archivo '{ruta_entrada.name}'.\n"
            f"Detalle: {error_ffmpeg}"
        )

    return ruta_salida


# Cargar un archivo WAV PCM ya estandarizado y entregarlo como arreglo
def cargar_muestras_pcm(ruta_wav: Path) -> tuple[np.ndarray, int]:
    muestras, sample_rate = sf.read(ruta_wav, dtype="float32")
    if muestras.ndim > 1:
        muestras = muestras.mean(axis=1) #monocanal
    return muestras, sample_rate

# Orquesta el flujo completo de preprocesamiento: archivo original -> WAV PCM mono 16kHz -> muestras cargadas
def procesar_audio(ruta_entrada: Path, ruta_salida_wav: Path) -> dict:
    convertir_a_wav_pcm(ruta_entrada, ruta_salida_wav)
    muestras, sample_rate = cargar_muestras_pcm(ruta_salida_wav)

    duracion_segundos = len(muestras) / sample_rate

    return {
        "sample_rate": sample_rate,
        "canales": CANALES_OBJETIVO,
        "num_muestras": int(len(muestras)),
        "duracion_segundos": round(duracion_segundos, 3),
        "wav_path": str(ruta_salida_wav),
    }
