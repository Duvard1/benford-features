"""
psd.py

Responsabilidad única de este módulo:
    Tomar las muestras PCM de un audio ya estandarizado (mono, 16kHz)
    y calcular la Power Spectral Density (PSD) de cada ventana
    individual, sin promediar.

        Audio (muestras PCM)
            ↓
        División en ventanas (reutiliza fft.dividir_en_ventanas)
            ↓
        Ventana Hann aplicada a cada frame
            ↓
        FFT por ventana
            ↓
        Periodograma: |FFT|² / (fs · Σw²)
            ↓
        Concatenar todos los valores PSD

    ¿Por qué no usar scipy.signal.welch?
    Welch promedia los periodogramas de todas las ventanas en un
    único vector de 513 valores. Eso es útil para obtener una
    estimación suavizada de la densidad espectral, pero destruye la
    variabilidad temporal: un audio de 45 segundos produce los mismos
    513 valores que uno de 3 segundos. Para la Ley de Benford se
    necesita un conjunto masivo de valores (cientos de miles) que
    conserve esa variabilidad ventana a ventana, igual que se hace
    con las magnitudes FFT.

    La fórmula utilizada es exactamente la que Welch aplica
    internamente a cada segmento antes de promediar:

        PSD(k) = |X(k)|² / (fs · S)

    donde X(k) = FFT del segmento enventanado, fs = sample rate,
    y S = Σ w(n)² es el factor de normalización de la ventana.
    Esto garantiza que los valores tengan unidades de V²/Hz
    (densidad espectral de potencia), no magnitudes arbitrarias.
"""

import numpy as np

from app.fft import FRAME_SIZE, HOP_LENGTH, dividir_en_ventanas


def extraer_psd(muestras: np.ndarray, sample_rate: int,
                frame_size: int = FRAME_SIZE,
                hop_length: int = HOP_LENGTH) -> dict:
    """
    Calcula la PSD de cada ventana individual del audio.

    A diferencia de scipy.signal.welch (que promedia todas las ventanas
    en un solo vector), esta función conserva el periodograma de cada
    ventana por separado y los concatena en un único arreglo 1D.

    Parámetros
    ----------
    muestras : np.ndarray 1D
        Señal de audio mono, ya estandarizada a 16kHz.
    sample_rate : int
        Frecuencia de muestreo (16000 Hz en este proyecto).
    frame_size : int
        Tamaño de cada ventana en muestras (default: 1024).
    hop_length : int
        Salto entre ventanas en muestras (default: 512).

    Retorna
    -------
    dict con:
        valores_psd: np.ndarray 1D (float32) con la PSD de todas las
                     ventanas concatenadas. Siempre >= 0.
        num_ventanas: cantidad de ventanas procesadas.
        bins_por_ventana: bins de frecuencia por ventana (frame_size // 2 + 1).
        total_valores: longitud total del arreglo de PSD.
    """
    # Reutiliza el mismo ventaneo que fft.py (valida longitud mínima).
    ventanas = dividir_en_ventanas(muestras, frame_size, hop_length)

    ventana_hann = np.hanning(frame_size).astype(np.float32)

    # Factor de normalización para densidad espectral de potencia.
    # S = Σ w(n)² es el estándar para la escala "density" de Welch.
    factor_normalizacion = sample_rate * np.sum(ventana_hann ** 2)

    num_ventanas, _ = ventanas.shape
    bins_por_ventana = frame_size // 2 + 1

    psd = np.empty((num_ventanas, bins_por_ventana), dtype=np.float32)

    for i in range(num_ventanas):
        frame_enventanado = ventanas[i] * ventana_hann
        espectro = np.fft.rfft(frame_enventanado)
        # |X(k)|² / (fs · S) → unidades de V²/Hz
        psd[i] = (np.abs(espectro) ** 2) / factor_normalizacion

    valores_psd = psd.reshape(-1)

    return {
        "valores_psd": valores_psd,
        "num_ventanas": int(num_ventanas),
        "bins_por_ventana": int(bins_por_ventana),
        "total_valores": int(valores_psd.size),
    }
