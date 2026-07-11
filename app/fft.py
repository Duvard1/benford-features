"""
fft.py

Responsabilidad única de este módulo:
    Tomar las muestras PCM de un audio ya estandarizado (mono, 16kHz)
    y extraer las magnitudes de la FFT, ventana por ventana, siguiendo
    el flujo metodológico acordado:

        Audio (muestras PCM)
            ↓
        División en ventanas (frames)
            ↓
        Ventana Hann aplicada a cada frame
            ↓
        FFT por ventana
            ↓
        Magnitudes (todas, sin submuestrear)

    Este módulo NO calcula el primer dígito significativo ni la
    distribución de Benford. Solo entrega el arreglo completo de
    magnitudes, tal como se acordó: "utilizar todas las magnitudes
    obtenidas por la FFT", sin recortar a una cantidad fija.
"""

import numpy as np

# Parámetros acordados en el documento de contexto.
# No son una ley: son un estándar por eficiencia (potencias de dos).
# Podrán modificarse más adelante para experimentar.
FRAME_SIZE = 1024      # tamaño de la ventana, en muestras
HOP_LENGTH = 512        # salto entre ventanas, en muestras


def dividir_en_ventanas(muestras: np.ndarray, frame_size: int = FRAME_SIZE,
                         hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Divide el audio completo en pequeños fragmentos (ventanas/frames)
    superpuestos, tal como lo describe el documento de contexto:

        Audio completo -> Ventana 1, Ventana 2, Ventana 3, ...

    No se aplica padding: si las últimas muestras no alcanzan para
    formar una ventana completa, simplemente se descartan (son una
    fracción mínima frente a cientos de miles de magnitudes totales).

    Retorna
    -------
    np.ndarray de forma (num_ventanas, frame_size)
        Cada fila es una ventana de audio sin procesar todavía.
    """
    total_muestras = len(muestras)

    if total_muestras < frame_size:
        raise ValueError(
            f"El audio tiene {total_muestras} muestras, "
            f"insuficientes para formar una sola ventana de {frame_size}."
        )

    num_ventanas = 1 + (total_muestras - frame_size) // hop_length

    ventanas = np.empty((num_ventanas, frame_size), dtype=np.float32)
    for i in range(num_ventanas):
        inicio = i * hop_length
        fin = inicio + frame_size
        ventanas[i] = muestras[inicio:fin]

    return ventanas


def extraer_magnitudes_fft(muestras: np.ndarray, frame_size: int = FRAME_SIZE,
                            hop_length: int = HOP_LENGTH) -> dict:
    """
    Aplica el flujo completo: ventaneo -> ventana Hann -> FFT -> magnitudes.

    Decisión técnica: se usa la FFT real (np.fft.rfft) en lugar de la FFT
    completa. Para una señal real (el audio lo es), el espectro completo
    es simétrico: la segunda mitad es un espejo exacto de la primera, por
    lo que no aporta información nueva, solo magnitudes duplicadas. Usar
    rfft entrega el espectro no redundante (frame_size // 2 + 1 bins por
    ventana) sin alterar en nada la distribución de primeros dígitos,
    y es el estándar para señales de audio reales.

    Las magnitudes se concatenan de TODAS las ventanas en un único
    arreglo 1D, siguiendo la recomendación del documento de contexto:
    no recortar a una cantidad fija, usar todo lo que produce la FFT.

    Retorna
    -------
    dict con:
        magnitudes: np.ndarray 1D (float32) con todas las magnitudes,
                    concatenadas de todas las ventanas. Siempre >= 0.
        num_ventanas: cantidad de ventanas procesadas.
        bins_por_ventana: cantidad de magnitudes por ventana (frame_size // 2 + 1).
        total_magnitudes: longitud total del arreglo de magnitudes.
    """
    ventanas = dividir_en_ventanas(muestras, frame_size, hop_length)

    # Ventana Hann: reduce el efecto de "fuga espectral" (spectral leakage)
    # que ocurre al cortar el audio en fragmentos abruptos.
    ventana_hann = np.hanning(frame_size).astype(np.float32)

    num_ventanas, _ = ventanas.shape
    bins_por_ventana = frame_size // 2 + 1

    magnitudes = np.empty((num_ventanas, bins_por_ventana), dtype=np.float32)

    for i in range(num_ventanas):
        frame_enventanado = ventanas[i] * ventana_hann
        espectro = np.fft.rfft(frame_enventanado)
        # |espectro| == sqrt(Real^2 + Imaginario^2), siempre positivo.
        magnitudes[i] = np.abs(espectro)

    magnitudes_planas = magnitudes.reshape(-1)

    return {
        "magnitudes": magnitudes_planas,
        "num_ventanas": int(num_ventanas),
        "bins_por_ventana": int(bins_por_ventana),
        "total_magnitudes": int(magnitudes_planas.size),
    }
