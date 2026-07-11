"""
stft.py

Responsabilidad única de este módulo:
    Tomar las muestras PCM de un audio ya estandarizado (mono, 16kHz)
    y calcular el espectrograma STFT (Short-Time Fourier Transform),
    devolviendo todas las magnitudes como un vector 1D.

        Audio (muestras PCM)
            ↓
        scipy.signal.stft
            ↓
        Matriz compleja Zxx (513 × N ventanas)
            ↓
        Magnitudes |Zxx|
            ↓
        Vector 1D (todos los valores concatenados)

    ¿En qué se diferencia de fft.py y psd.py?
    Las tres representaciones parten del mismo ventaneo y FFT, pero
    cada una aplica una transformación distinta sobre los coeficientes:

        - FFT  (fft.py):  |X(k)|            → magnitud cruda
        - PSD  (psd.py):  |X(k)|² / (fs·Σw²) → densidad de potencia (V²/Hz)
        - STFT (este):    |X(k)| / Σw        → magnitud normalizada (scipy)

    La normalización de scipy.signal.stft divide por la suma de la
    ventana (Σw), lo que hace que las magnitudes sean invariantes a
    la amplitud de la ventana. Esto reescala los valores respecto a
    la FFT cruda, generando un rango dinámico diferente y, por tanto,
    una distribución de primeros dígitos potencialmente distinta
    para el análisis de Benford.

    Se usa scipy.signal.stft con boundary=None y padded=False para
    que el número de segmentos coincida exactamente con los módulos
    FFT y PSD (sin padding artificial en los bordes).
"""

import numpy as np
from scipy.signal import stft

from app.fft import FRAME_SIZE, HOP_LENGTH


def extraer_stft(muestras: np.ndarray, sample_rate: int,
                 nperseg: int = FRAME_SIZE,
                 noverlap: int = HOP_LENGTH) -> dict:
    """
    Calcula el espectrograma STFT y devuelve todas las magnitudes
    como un arreglo 1D para el análisis de Benford.

    Utiliza scipy.signal.stft con los mismos parámetros de ventaneo
    que FFT y PSD (Hann, 1024, 512) para que las representaciones
    sean directamente comparables.

    Parámetros
    ----------
    muestras : np.ndarray 1D
        Señal de audio mono, ya estandarizada a 16kHz.
    sample_rate : int
        Frecuencia de muestreo (16000 Hz en este proyecto).
    nperseg : int
        Tamaño de cada ventana en muestras (default: 1024).
    noverlap : int
        Solapamiento entre ventanas en muestras (default: 512).

    Retorna
    -------
    dict con:
        valores_stft: np.ndarray 1D (float32) con las magnitudes del
                      espectrograma, de todas las ventanas concatenadas.
        num_ventanas: cantidad de ventanas (segmentos temporales).
        bins_por_ventana: bins de frecuencia por ventana (nperseg // 2 + 1).
        total_valores: longitud total del arreglo.
    """
    if len(muestras) < nperseg:
        raise ValueError(
            f"El audio tiene {len(muestras)} muestras, "
            f"insuficientes para calcular la STFT con nperseg={nperseg}."
        )

    # boundary=None, padded=False: sin padding artificial en los bordes,
    # para que el número de ventanas coincida con FFT y PSD.
    _frecuencias, _tiempos, Zxx = stft(
        muestras,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    # Zxx tiene forma (bins_frecuencia, num_ventanas).
    # np.abs convierte los coeficientes complejos a magnitudes.
    magnitudes = np.abs(Zxx).astype(np.float32)

    bins_por_ventana, num_ventanas = magnitudes.shape

    # Transponer a (num_ventanas, bins_por_ventana) antes de aplanar,
    # para mantener la misma organización que FFT y PSD: primero todos
    # los bins de la ventana 1, luego los de la ventana 2, etc.
    valores_stft = magnitudes.T.ravel()

    return {
        "valores_stft": valores_stft,
        "num_ventanas": int(num_ventanas),
        "bins_por_ventana": int(bins_por_ventana),
        "total_valores": int(valores_stft.size),
    }
