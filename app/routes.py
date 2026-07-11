"""
routes.py

Define los endpoints de la API.

Flujo actual de /benford-features:
    1. Recibe el archivo de audio + el parámetro "feature".
    2. Valida su extensión.
    3. Lo guarda en /uploads.
    4. Lo convierte a WAV PCM mono 16kHz en /temp.
    5. Carga las muestras.
    6. Según la feature solicitada:
       - "fft": Extrae las magnitudes FFT (ventaneo + Hann + FFT real).
       - "psd": Estima la Power Spectral Density por ventana.
       - "stft": Calcula el espectrograma STFT (magnitudes normalizadas).
       - "logmel": Log-Mel Espectrograma (128 bandas Mel en dB).
       - "mel": Mel Espectrograma en potencia lineal (sin log).
       - "mfcc": Coeficientes Cepstrales en las Frecuencias de Mel (13 coefs).
    7. Aplica la Ley de Benford sobre los valores extraídos.
    8. Calcula las métricas de desviación (χ², MAD, KL, JS) entre lo
       observado y Benford.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.audio import AudioConversionError, procesar_audio
from app.benford import analizar_benford
from app.fft import extraer_magnitudes_fft
from app.logmel import extraer_logmel
from app.mel import extraer_mel
from app.mfcc import extraer_mfcc
from app.metrics import comparar_con_benford
from app.psd import extraer_psd
from app.stft import extraer_stft
from app.utils import generar_nombre_unico, validar_extension

router = APIRouter()

# Carpetas de trabajo (relativas a la raíz del proyecto).
UPLOADS_DIR = Path("uploads")
TEMP_DIR = Path("temp")

UPLOADS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Representaciones soportadas hasta ahora. El resto (mfcc, etc.)
# se irá agregando en etapas posteriores, reutilizando este mismo flujo.
FEATURES_IMPLEMENTADAS = {"fft", "psd", "stft", "logmel", "mel", "mfcc"}


@router.post("/benford-features")
async def benford_features(file: UploadFile = File(...), feature: str = Form("fft")):
    """
    Recibe un archivo de audio y la representación a analizar,
    lo convierte a WAV PCM mono 16kHz y extrae los valores
    correspondientes para el análisis de Benford.
    """
    feature = feature.lower().strip()

    if feature not in FEATURES_IMPLEMENTADAS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"La representación '{feature}' todavía no está implementada. "
                f"Disponibles por ahora: {sorted(FEATURES_IMPLEMENTADAS)}"
            ),
        )

    # 1. Validar extensión
    try:
        extension = validar_extension(file.filename)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    # 2. Guardar el archivo original en /uploads con nombre único
    nombre_original = generar_nombre_unico(extension)
    ruta_entrada = UPLOADS_DIR / nombre_original

    contenido = await file.read()
    if not contenido:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    ruta_entrada.write_bytes(contenido)

    # 3. Definir ruta de salida del WAV convertido en /temp
    nombre_wav = generar_nombre_unico(".wav")
    ruta_salida_wav = TEMP_DIR / nombre_wav

    # 4. Convertir y cargar el audio
    try:
        info_audio = procesar_audio(ruta_entrada, ruta_salida_wav)
    except AudioConversionError as error:
        raise HTTPException(status_code=422, detail=str(error))

    respuesta = {
        "status": "ok",
        "archivo_original": file.filename,
        "feature": feature,
        "audio_convertido": info_audio,
    }

    # 5. Cargar las muestras del WAV convertido
    muestras, sr = sf.read(ruta_salida_wav, dtype="float32")

    # 6. Extraer la representación pedida
    if feature == "fft":
        try:
            resultado_fft = extraer_magnitudes_fft(muestras)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        valores = resultado_fft["magnitudes"]

        # Guardar en disco (son demasiadas para devolverlas en JSON).
        nombre_npy = generar_nombre_unico(".npy")
        ruta_valores = TEMP_DIR / nombre_npy
        np.save(ruta_valores, valores)

        respuesta["fft_info"] = {
            "num_ventanas": resultado_fft["num_ventanas"],
            "bins_por_ventana": resultado_fft["bins_por_ventana"],
            "total_magnitudes": resultado_fft["total_magnitudes"],
            "magnitudes_stats": {
                "min": float(valores.min()),
                "max": float(valores.max()),
                "mean": float(valores.mean()),
            },
            "magnitudes_path": str(ruta_valores),
        }

    elif feature == "psd":
        try:
            resultado_psd = extraer_psd(muestras, sr)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        valores = resultado_psd["valores_psd"]

        respuesta["psd_info"] = {
            "num_ventanas": resultado_psd["num_ventanas"],
            "bins_por_ventana": resultado_psd["bins_por_ventana"],
            "total_valores": resultado_psd["total_valores"],
            "psd_stats": {
                "min": float(valores.min()),
                "max": float(valores.max()),
                "mean": float(valores.mean()),
            },
        }

    elif feature == "stft":
        try:
            resultado_stft = extraer_stft(muestras, sr)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        valores = resultado_stft["valores_stft"]

        respuesta["stft_info"] = {
            "num_ventanas": resultado_stft["num_ventanas"],
            "bins_por_ventana": resultado_stft["bins_por_ventana"],
            "total_valores": resultado_stft["total_valores"],
            "stft_stats": {
                "min": float(valores.min()),
                "max": float(valores.max()),
                "mean": float(valores.mean()),
            },
        }

    elif feature == "logmel":
        try:
            resultado_logmel = extraer_logmel(muestras, sr)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        valores = resultado_logmel["valores_logmel"]

        respuesta["logmel_info"] = {
            "num_bandas_mel": resultado_logmel["num_bandas_mel"],
            "num_frames": resultado_logmel["num_frames"],
            "total_valores": resultado_logmel["total_valores"],
            "logmel_stats": {
                "min": float(valores.min()),
                "max": float(valores.max()),
                "mean": float(valores.mean()),
            },
        }

    elif feature == "mel":
        try:
            resultado_mel = extraer_mel(muestras, sr)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        valores = resultado_mel["valores_mel"]

        respuesta["mel_info"] = {
            "num_bandas_mel": resultado_mel["num_bandas_mel"],
            "num_frames": resultado_mel["num_frames"],
            "total_valores": resultado_mel["total_valores"],
            "mel_stats": {
                "min": float(valores.min()),
                "max": float(valores.max()),
                "mean": float(valores.mean()),
            },
        }

    elif feature == "mfcc":
        try:
            resultado_mfcc = extraer_mfcc(muestras, sr)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

        valores = resultado_mfcc["valores_mfcc"]

        respuesta["mfcc_info"] = {
            "num_coeficientes": resultado_mfcc["num_coeficientes"],
            "num_frames": resultado_mfcc["num_frames"],
            "total_valores": resultado_mfcc["total_valores"],
            "offset_aplicado": resultado_mfcc["offset_aplicado"],
            "mfcc_stats": {
                "min": float(valores.min()),
                "max": float(valores.max()),
                "mean": float(valores.mean()),
            },
        }

    # 7. Aplicar la Ley de Benford sobre los valores extraídos
    try:
        resultado_benford = analizar_benford(valores)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    respuesta["benford"] = resultado_benford

    # 8. Calcular las métricas de desviación respecto a Benford
    respuesta["metricas"] = comparar_con_benford(resultado_benford)

    return respuesta