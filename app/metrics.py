"""
metrics.py

Responsabilidad única de este módulo:
    Cuantificar qué tan lejos está la distribución observada del primer
    dígito significativo (calculada en benford.py) respecto de la
    distribución teórica de Benford.

    Se implementan cuatro métricas, tal como se acordó:
        - Chi-cuadrado (χ²)                  -> ¿la diferencia es
                                                 estadísticamente
                                                 significativa?
        - Mean Absolute Deviation (MAD)       -> ¿qué tan grande es la
                                                 diferencia, en promedio?
        - Kullback-Leibler (KL)               -> ¿cuánta información se
                                                 "pierde" al asumir Benford
                                                 en vez de lo observado?
        - Jensen-Shannon (JS)                 -> versión simétrica y
                                                 acotada de KL, más fácil
                                                 de comparar entre audios.

    Ninguna de estas métricas decide por sí sola si un audio es "real"
    o "IA". Sirven para comparar numéricamente varios audios entre sí
    y, eventualmente, varias representaciones (FFT, PSD, MFCC, etc.)
    entre sí.
"""

import numpy as np
from scipy.stats import chi2 as chi2_dist

# Umbrales de referencia para MAD en el test del primer dígito,
# publicados por Nigrini (2012) para pruebas de Benford con 9 categorías.
# Son una referencia orientativa, no una regla absoluta.
UMBRALES_MAD = (
    (0.006, "Conformidad cercana (close conformity)"),
    (0.012, "Conformidad aceptable (acceptable conformity)"),
    (0.015, "Conformidad marginal (marginally acceptable)"),
    (float("inf"), "No conformidad (nonconformity)"),
)

NIVEL_SIGNIFICANCIA = 0.05  # umbral estándar para el p-valor de χ²


def chi_cuadrado(conteo_observado, frecuencia_esperada, total_valores_validos: int) -> dict:
    """
    Prueba de bondad de ajuste χ² entre el conteo observado por dígito
    y el conteo esperado según Benford (frecuencia_esperada * N).

    Un p-valor bajo (< 0.05) indica que la distribución observada se
    aleja de Benford de forma estadísticamente significativa (poco
    probable que sea por azar). Un p-valor alto indica que no hay
    evidencia suficiente para decir que se aleja de Benford.
    """
    conteo_observado = np.asarray(conteo_observado, dtype=np.float64)
    frecuencia_esperada = np.asarray(frecuencia_esperada, dtype=np.float64)
    conteo_esperado = frecuencia_esperada * total_valores_validos

    estadistico = float(np.sum((conteo_observado - conteo_esperado) ** 2 / conteo_esperado))
    grados_libertad = int(len(conteo_observado) - 1)  # 9 dígitos -> 8 g.l.
    p_valor = float(chi2_dist.sf(estadistico, grados_libertad))

    return {
        "estadistico": estadistico,
        "grados_libertad": grados_libertad,
        "p_valor": p_valor,
        "significativo_al_5pct": bool(p_valor < NIVEL_SIGNIFICANCIA),
    }


def _interpretar_mad(valor: float) -> str:
    for umbral, etiqueta in UMBRALES_MAD:
        if valor < umbral:
            return etiqueta
    return UMBRALES_MAD[-1][1]  # inalcanzable, por seguridad


def mean_absolute_deviation(frecuencia_observada, frecuencia_esperada) -> dict:
    """
    MAD: promedio de las diferencias absolutas entre frecuencia
    observada y esperada, dígito por dígito. Es la métrica más
    intuitiva: está en la misma escala que las frecuencias (0 a 1).
    """
    observada = np.asarray(frecuencia_observada, dtype=np.float64)
    esperada = np.asarray(frecuencia_esperada, dtype=np.float64)

    valor = float(np.mean(np.abs(observada - esperada)))

    return {
        "valor": valor,
        "interpretacion": _interpretar_mad(valor),
    }


def _kl_divergencia(p, q, base: float = 2.0) -> float:
    """
    KL(P || Q) = sum( p_i * log(p_i / q_i) ), solo sobre los i donde
    p_i > 0 (por convención, 0 * log(0/q) = 0).

    Se asume que q_i > 0 en todos los dígitos donde p_i > 0 (cierto
    para la distribución de Benford, que nunca es cero en 1-9).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    mascara = p > 0
    terminos = p[mascara] * np.log(p[mascara] / q[mascara])
    valor = float(np.sum(terminos))

    return valor / np.log(base)


def kullback_leibler(frecuencia_observada, frecuencia_esperada) -> dict:
    """
    Divergencia KL de la distribución observada respecto a la
    esperada (Benford). Se reporta en base 2 (bits): 0 significa
    distribuciones idénticas; valores más altos indican mayor
    divergencia. A diferencia de MAD, KL penaliza más fuerte las
    diferencias en dígitos donde Benford espera poca frecuencia
    (dígitos altos, 7-9).
    """
    valor = _kl_divergencia(frecuencia_observada, frecuencia_esperada, base=2.0)
    return {"valor": valor, "unidad": "bits (log base 2)"}


def jensen_shannon(frecuencia_observada, frecuencia_esperada) -> dict:
    """
    Divergencia de Jensen-Shannon: versión simétrica y acotada de KL.
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), donde M = (P+Q)/2.

    En base 2, está siempre entre 0 (distribuciones idénticas) y 1
    (distribuciones totalmente distintas), lo que la hace más fácil
    de comparar entre audios distintos que KL (que no está acotada).
    """
    p = np.asarray(frecuencia_observada, dtype=np.float64)
    q = np.asarray(frecuencia_esperada, dtype=np.float64)
    m = 0.5 * (p + q)

    kl_p_m = _kl_divergencia(p, m, base=2.0)
    kl_q_m = _kl_divergencia(q, m, base=2.0)
    valor = 0.5 * kl_p_m + 0.5 * kl_q_m

    return {"valor": valor, "unidad": "bits (log base 2), acotado en [0, 1]"}


def comparar_con_benford(resultado_benford: dict) -> dict:
    """
    Orquesta el cálculo de las cuatro métricas a partir del resultado
    de benford.analizar_benford(...).

    Retorna un diccionario con las cuatro métricas y un resumen breve,
    listo para devolverse en la respuesta JSON o para comparar entre
    varios audios / representaciones.
    """
    conteo_observado = resultado_benford["conteo_observado"]
    frecuencia_observada = resultado_benford["frecuencia_observada"]
    frecuencia_esperada = resultado_benford["frecuencia_esperada_benford"]
    total_valores_validos = resultado_benford["total_valores_validos"]

    chi2_resultado = chi_cuadrado(conteo_observado, frecuencia_esperada, total_valores_validos)
    mad_resultado = mean_absolute_deviation(frecuencia_observada, frecuencia_esperada)
    kl_resultado = kullback_leibler(frecuencia_observada, frecuencia_esperada)
    js_resultado = jensen_shannon(frecuencia_observada, frecuencia_esperada)

    return {
        "chi_cuadrado": chi2_resultado,
        "mad": mad_resultado,
        "kl_divergencia": kl_resultado,
        "js_divergencia": js_resultado,
    }