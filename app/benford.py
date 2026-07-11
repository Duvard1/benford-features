"""
benford.py

Responsabilidad única de este módulo:
    Tomar el arreglo de magnitudes (por ahora, magnitudes FFT) y aplicar
    la Ley de Benford:

        Magnitudes
            ↓
        Primer dígito significativo
            ↓
        Histograma (conteo observado por dígito 1-9)
            ↓
        Distribución esperada de Benford

    Este módulo NO calcula las métricas de desviación (χ², MAD, KL, JS).
    Eso corresponde a metrics.py, en la siguiente etapa.
"""

from pathlib import Path

import numpy as np

DIGITOS = np.arange(1, 10)  # 1, 2, ..., 9


def cargar_magnitudes(ruta_npy: Path) -> np.ndarray:
    """
    Carga un arreglo de magnitudes previamente guardado por fft.py
    (archivo .npy en /temp).
    """
    return np.load(ruta_npy)


def primer_digito_significativo(valores: np.ndarray) -> dict:
    """
    Extrae el primer dígito distinto de cero de cada valor, de forma
    vectorizada (sin loops en Python, importante porque puede haber
    millones de magnitudes).

    Ejemplos (los mismos del documento de contexto):
        0.000381 -> 3
        18.42    -> 1
        245      -> 2

    Método: para x > 0,
        primer_digito = floor( x / 10**floor(log10(x)) )

    Las magnitudes FFT nunca son negativas (se calculan como
    sqrt(Real^2 + Imaginario^2)), pero SÍ pueden ser exactamente 0
    (silencio total en esa frecuencia/ventana). log10(0) no está
    definido y el 0 no tiene "primer dígito significativo", así que
    esos valores se excluyen del análisis y se reportan aparte.

    Retorna
    -------
    dict con:
        primeros_digitos: np.ndarray de enteros (1-9), uno por cada
                           valor válido (> 0).
        total_valores: cantidad total de magnitudes recibidas.
        valores_excluidos: cantidad de magnitudes == 0 (o negativas,
                            como medida de seguridad) que se excluyeron.
        total_valores_validos: cantidad de magnitudes realmente usadas.
    """
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

    # Corrección de errores de punto flotante: en casos límite
    # (ej. x muy cercano a una potencia de 10), la división puede dar
    # 9.999999999 -> int() trunca a 9, o a veces 10.0000000001 -> 10.
    # Se recorta al rango válido [1, 9] para evitar dígitos imposibles.
    primeros_digitos = np.clip(primeros_digitos, 1, 9)

    return {
        "primeros_digitos": primeros_digitos,
        "total_valores": total_valores,
        "valores_excluidos": int(valores_excluidos),
        "total_valores_validos": int(validos.size),
    }


def construir_histograma(primeros_digitos: np.ndarray) -> dict:
    """
    Construye el histograma observado: cuántas veces aparece cada
    dígito del 1 al 9 como primer dígito significativo.
    """
    # bincount necesita minlength=10 para cubrir el índice 9;
    # descartamos el índice 0 porque el dígito 0 nunca es válido aquí.
    conteos = np.bincount(primeros_digitos, minlength=10)[1:10]
    total = int(conteos.sum())
    frecuencias_observadas = conteos / total

    return {
        "digitos": DIGITOS.tolist(),
        "conteo_observado": conteos.tolist(),
        "frecuencia_observada": frecuencias_observadas.tolist(),
    }


def distribucion_esperada_benford() -> dict:
    """
    Distribución teórica de Benford para el primer dígito:

        P(d) = log10(1 + 1/d),  d = 1, 2, ..., 9

    No depende de los datos: es siempre la misma.
    """
    frecuencias_esperadas = np.log10(1 + 1 / DIGITOS)

    return {
        "digitos": DIGITOS.tolist(),
        "frecuencia_esperada": frecuencias_esperadas.tolist(),
    }


def analizar_benford(magnitudes: np.ndarray) -> dict:
    """
    Orquesta el flujo completo de este módulo:

        magnitudes -> primer dígito -> histograma -> distribución esperada

    Retorna un diccionario listo para combinarse con metrics.py
    (χ², MAD, KL, JS) en la siguiente etapa, y para devolverse como
    parte de la respuesta JSON del endpoint.
    """
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
