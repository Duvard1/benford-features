"""
logmel.py

Responsabilidad única de este módulo:
    Tomar las muestras PCM de un audio ya estandarizado (mono, 16kHz)
    y calcular el Log-Mel Espectrograma, devolviendo todos los valores
    como un vector 1D listo para el análisis de Benford.

        Audio (muestras PCM)
            ↓
        STFT (via librosa)
            ↓
        Banco de filtros Mel (128 bandas)
            ↓
        Mel Spectrogram (potencia)
            ↓
        Escala logarítmica (dB) con librosa.power_to_db
            ↓
        Desplazamiento a valores positivos
            ↓
        Vector 1D

    Preparación de valores para Benford
    ------------------------------------
    librosa.power_to_db produce valores en dB, típicamente en el rango
    [-80, 0] dB (o ligeramente positivos). La Ley de Benford solo se
    aplica a valores estrictamente positivos (benford.py filtra con > 0).

    La escala en dB es relativa: el punto "0 dB" depende del valor de
    referencia elegido (ref=1.0 por defecto en librosa). Cambiar esa
    referencia simplemente desplaza toda la escala por una constante:

        10·log10(S/ref1) = 10·log10(S/ref2) + 10·log10(ref2/ref1)

    Por tanto, desplazar todos los valores sumando |min| + ε es
    matemáticamente equivalente a haber elegido una referencia menor.
    Esto no altera las diferencias relativas entre valores (que son lo
    que importa para la distribución del primer dígito), solo traslada
    la escala para que todos sean positivos.
"""

import numpy as np
import librosa

from app.fft import FRAME_SIZE, HOP_LENGTH

N_MELS = 128


def extraer_logmel(muestras: np.ndarray, sample_rate: int,
                   n_fft: int = FRAME_SIZE,
                   hop_length: int = HOP_LENGTH,
                   n_mels: int = N_MELS) -> dict:
    """
    Calcula el Log-Mel Espectrograma y devuelve todos los valores
    como un arreglo 1D positivo para el análisis de Benford.

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
        valores_logmel: np.ndarray 1D (float32) con los valores del
                        Log-Mel desplazados a positivos. Siempre > 0.
        num_bandas_mel: número de bandas Mel (128).
        num_frames: número de frames temporales.
        total_valores: longitud total del arreglo.
    """
    if len(muestras) < n_fft:
        raise ValueError(
            f"El audio tiene {len(muestras)} muestras, "
            f"insuficientes para calcular el Log-Mel con n_fft={n_fft}."
        )

    # 1. Mel Spectrogram (potencia): STFT → |S|² → filtros Mel
    # center=False para no añadir padding y que el número de frames
    # sea consistente con los otros módulos (fft, psd, stft).
    mel_spec = librosa.feature.melspectrogram(
        y=muestras,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        n_mels=n_mels,
        center=False,
    )

    # 2. Conversión a escala logarítmica (dB)
    # power_to_db: 10 * log10(S / ref), con piso en -80 dB por defecto.
    log_mel_db = librosa.power_to_db(mel_spec, ref=1.0)

    # 3. Desplazar a valores positivos para Benford.
    # Sumar |min| + 1 para que el mínimo quede en 1.0 (no en ~0).
    # Esto equivale a elegir una referencia de potencia menor;
    # no altera las diferencias relativas entre valores.
    min_val = log_mel_db.min()
    if min_val <= 0:
        log_mel_db = log_mel_db + (abs(min_val) + 1.0)

    # log_mel_db tiene forma (n_mels, num_frames)
    num_bandas_mel, num_frames = log_mel_db.shape

    valores_logmel = log_mel_db.T.ravel().astype(np.float32)

    return {
        "valores_logmel": valores_logmel,
        "num_bandas_mel": int(num_bandas_mel),
        "num_frames": int(num_frames),
        "total_valores": int(valores_logmel.size),
    }
