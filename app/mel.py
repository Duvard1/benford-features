"""
mel.py

Responsabilidad única de este módulo:
    Tomar las muestras PCM de un audio ya estandarizado (mono, 16kHz)
    y calcular el Mel Espectrograma en escala de potencia lineal,
    devolviendo todos los valores como un vector 1D.

        Audio (muestras PCM)
            ↓
        STFT (via librosa)
            ↓
        Banco de filtros Mel (128 bandas)
            ↓
        Mel Spectrogram (potencia)
            ↓
        Vector 1D (sin transformaciones)

    Este módulo es el complemento experimental de logmel.py.
    La diferencia clave: aquí NO se aplica power_to_db().

    logmel.py: mel → power_to_db → dB → desplazamiento → Benford
    mel.py:    mel → directamente → Benford

    Esto permite aislar experimentalmente el efecto de la
    transformación logarítmica sobre la distribución de primeros
    dígitos de Benford. Si las métricas (χ², MAD, KL, JS) difieren
    significativamente entre "mel" y "logmel", la responsable es la
    compresión logarítmica; si no, el efecto viene de los filtros Mel.
"""

import numpy as np
import librosa

from app.fft import FRAME_SIZE, HOP_LENGTH

N_MELS = 128


def extraer_mel(muestras: np.ndarray, sample_rate: int,
                n_fft: int = FRAME_SIZE,
                hop_length: int = HOP_LENGTH,
                n_mels: int = N_MELS) -> dict:
    """
    Calcula el Mel Espectrograma en escala de potencia lineal.

    Los valores ya son positivos (potencia = |STFT|² × filtros Mel)
    y no requieren ninguna transformación adicional para Benford.

    Parámetros
    ----------
    muestras : np.ndarray 1D
        Señal de audio mono, ya estandarizada a 16kHz.
    sample_rate : int
        Frecuencia de muestreo (16000 Hz en este proyecto).
    n_fft : int
        Tamaño de la FFT / ventana (default: 1024).
    hop_length : int
        Salto entre ventanas (default: 512).
    n_mels : int
        Número de bandas Mel (default: 128).

    Retorna
    -------
    dict con:
        valores_mel: np.ndarray 1D (float32) con los valores de potencia
                     del Mel Spectrogram. Siempre >= 0.
        num_bandas_mel: número de bandas Mel (128).
        num_frames: número de frames temporales.
        total_valores: longitud total del arreglo.
    """
    if len(muestras) < n_fft:
        raise ValueError(
            f"El audio tiene {len(muestras)} muestras, "
            f"insuficientes para calcular el Mel Spectrogram con n_fft={n_fft}."
        )

    # center=False para no añadir padding, consistente con los demás módulos.
    mel_spec = librosa.feature.melspectrogram(
        y=muestras,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        n_mels=n_mels,
        center=False,
    )

    # mel_spec tiene forma (n_mels, num_frames), valores en potencia lineal.
    # No se aplica power_to_db: los valores se usan tal cual.
    num_bandas_mel, num_frames = mel_spec.shape

    valores_mel = mel_spec.T.ravel().astype(np.float32)

    return {
        "valores_mel": valores_mel,
        "num_bandas_mel": int(num_bandas_mel),
        "num_frames": int(num_frames),
        "total_valores": int(valores_mel.size),
    }
