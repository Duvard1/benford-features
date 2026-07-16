# Tomar arreglo de magnitudes y aplicar ley de benford
from pathlib import Path
import numpy as np

DIGITOS = np.arange(1, 10)

def cargar_magnitudes(ruta_npy: Path) -> np.ndarray:
    return np.load(ruta_npy)

def primer_digito_significativo(valores: np.ndarray) -> dict:
    valores = np.asarray(valores)
    total_valores = int(valores.size)
    mascara_validos = valores > 0
    validos = valores[mascara_validos]
    valores_excluidos = total_valores - int(validos.size)

    if validos.size == 0:
        raise ValueError(
            "No hay magnitudes válidas (> 0) para analizar con la Ley de "
            "Benford. Todas las magnitudes recibidas son cero o negativas."
        )

    exponentes = np.floor(np.log10(validos))
    primeros_digitos = (validos / (10.0 ** exponentes)).astype(np.int64)

    primeros_digitos = np.clip(primeros_digitos, 1, 9)

    return {
        "primeros_digitos": primeros_digitos,
        "total_valores": total_valores,
        "valores_excluidos": int(valores_excluidos),
        "total_valores_validos": int(validos.size),
    }


def construir_histograma(primeros_digitos: np.ndarray) -> dict:
    # bincount necesita minlength=10 para cubrir el índice 9;
    conteos = np.bincount(primeros_digitos, minlength=10)[1:10]
    total = int(conteos.sum())
    frecuencias_observadas = conteos / total

    return {
        "digitos": DIGITOS.tolist(),
        "conteo_observado": conteos.tolist(),
        "frecuencia_observada": frecuencias_observadas.tolist(),
    }


def distribucion_esperada_benford() -> dict:
    frecuencias_esperadas = np.log10(1 + 1 / DIGITOS)
    return {
        "digitos": DIGITOS.tolist(),
        "frecuencia_esperada": frecuencias_esperadas.tolist(),
    }

def analizar_benford(magnitudes: np.ndarray) -> dict:
    #orquestador
    resultado_digitos = primer_digito_significativo(magnitudes)
    histograma = construir_histograma(resultado_digitos["primeros_digitos"])
    esperada = distribucion_esperada_benford()

    return {
        "total_valores": resultado_digitos["total_valores"],
        "valores_excluidos": resultado_digitos["valores_excluidos"],
        "total_valores_validos": resultado_digitos["total_valores_validos"],
        "digitos": histograma["digitos"],
        "conteo_observado": histograma["conteo_observado"],
        "frecuencia_observada": histograma["frecuencia_observada"],
        "frecuencia_esperada_benford": esperada["frecuencia_esperada"],
    }
