# Toma muestras y extrae magnitudes FFT y entraga arreglo de magnitudes
import numpy as np
# tamaño de la ventana, en muestras, estandar por eficiencia (potencias de dos)
FRAME_SIZE = 1024
HOP_LENGTH = 512        

def dividir_en_ventanas(muestras: np.ndarray, frame_size: int = FRAME_SIZE,
                         hop_length: int = HOP_LENGTH) -> np.ndarray:
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
