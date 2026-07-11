"""
mfcc.py

Responsabilidad única de este módulo:
    Tomar las muestras PCM de un audio ya estandarizado (mono, 16kHz)
    y extraer los Coeficientes Cepstrales en las Frecuencias de Mel (MFCC),
    devolviendo todos los coeficientes de todos los frames como un vector 1D
    procesado para el análisis de Benford.

        Audio (muestras PCM)
            ↓
        MFCC (via librosa.feature.mfcc)
            ↓
        Matriz de coeficientes (13 × N frames)
            ↓
        Preprocesamiento para Benford: abs(MFCC) + EPSILON
            ↓
        Vector 1D (aplanado por frames)

    Justificación Científica / Metodológica
    --------------------------------------
    Anteriormente se aplicaba un desplazamiento global (offset) sumando
    |min| + 1 a todos los coeficientes. Sin embargo, los resultados indicaron
    que este gran desplazamiento dominaba completamente la distribución,
    concentrando los primeros dígitos en 6 y 7 (el primer dígito del propio
    offset). Esto causaba que el experimento analizara la constante del offset
    en lugar del comportamiento intrínseco de los MFCC.

    Para evitar este sesgo, en este experimento se adopta una estrategia sin
    desplazamiento artificial:
    1. Se aplica el valor absoluto (np.abs) para remover los signos negativos.
    2. Se añade un valor ínfimo (EPSILON = 1e-12) para evitar exactamente los
       valores en cero, que no son válidos para el análisis de Benford (log10).

    Esto preserva la magnitud relativa original y el rango dinámico de cada
    coeficiente sin sesgar la distribución de dígitos con un offset constante.
"""

import numpy as np
import librosa

from app.fft import FRAME_SIZE, HOP_LENGTH

N_MFCC = 13
EPSILON = 1e-12


def extraer_mfcc(muestras: np.ndarray, sample_rate: int,
                 n_fft: int = FRAME_SIZE,
                 hop_length: int = HOP_LENGTH,
                 n_mels: int = 128,  # librosa default/consistency
                 n_mfcc: int = N_MFCC) -> dict:
    """
    Calcula los MFCC del audio y devuelve todos los coeficientes enventanados
    como un arreglo 1D positivo para el análisis de Benford usando valor absoluto.

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
    n_mfcc : int
        Número de coeficientes MFCC a extraer (default: 13).

    Retorna
    -------
    dict con:
        valores_mfcc: np.ndarray 1D (float32) con los coeficientes procesados con abs + epsilon.
        num_coeficientes: número de coeficientes por frame (13).
        num_frames: número de frames temporales.
        total_valores: longitud total del arreglo 1D.
        offset_aplicado: metadatos del preprocesamiento para compatibilidad con routes.py.
    """
    if len(muestras) < n_fft:
        raise ValueError(
            f"El audio tiene {len(muestras)} muestras, "
            f"insuficientes para calcular los MFCC con n_fft={n_fft}."
        )

    # 1. Extraer los MFCC.
    # center=False para mantener consistencia temporal con el resto de representaciones.
    mfcc_coeffs = librosa.feature.mfcc(
        y=muestras,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        n_mfcc=n_mfcc,
        center=False,
    )

    # 2. Preprocesamiento libre de desplazamientos sesgados:
    # Aplicar valor absoluto y sumar EPSILON para evitar ceros.
    mfcc_coeffs = np.abs(mfcc_coeffs) + EPSILON

    # mfcc_coeffs tiene forma (n_mfcc, num_frames)
    num_coef, num_frames = mfcc_coeffs.shape

    # Transponer a (num_frames, n_mfcc) antes de aplanar con flatten/ravel.
    valores_mfcc = mfcc_coeffs.T.ravel().astype(np.float32)

    return {
        "valores_mfcc": valores_mfcc,
        "num_coeficientes": int(num_coef),
        "num_frames": int(num_frames),
        "total_valores": int(valores_mfcc.size),
        # Se retorna un diccionario de metadatos en lugar de un float.
        # Esto satisface el llamado en routes.py sin modificar dicho archivo.
        "offset_aplicado": {
            "operacion": "valor_absoluto",
            "epsilon": EPSILON
        },
    }

