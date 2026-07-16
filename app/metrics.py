# Cuantifica diferencia entre distribuion observada y esperada(Benford)

import numpy as np
from scipy.stats import chi2 as chi2_dist

# Umbrales de referencia para MAD en el test del primer dígito, umbrales de nigrini
UMBRALES_MAD = (
    (0.006, "Conformidad cercana"),
    (0.012, "Conformidad aceptable"),
    (0.015, "Conformidad marginal"),
    (float("inf"), "No conformidad"),
)

# umbral estándar para el p-valor de χ²
NIVEL_SIGNIFICANCIA = 0.05  


def chi_cuadrado(conteo_observado, frecuencia_esperada, total_valores_validos: int) -> dict:
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
    observada = np.asarray(frecuencia_observada, dtype=np.float64)
    esperada = np.asarray(frecuencia_esperada, dtype=np.float64)

    valor = float(np.mean(np.abs(observada - esperada)))

    return {
        "valor": valor,
        "interpretacion": _interpretar_mad(valor),
    }


def _kl_divergencia(p, q, base: float = 2.0) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    mascara = p > 0
    terminos = p[mascara] * np.log(p[mascara] / q[mascara])
    valor = float(np.sum(terminos))

    return valor / np.log(base)


def kullback_leibler(frecuencia_observada, frecuencia_esperada) -> dict:
    valor = _kl_divergencia(frecuencia_observada, frecuencia_esperada, base=2.0)
    return {"valor": valor, "unidad": "bits (log base 2)"}


def jensen_shannon(frecuencia_observada, frecuencia_esperada) -> dict:
    p = np.asarray(frecuencia_observada, dtype=np.float64)
    q = np.asarray(frecuencia_esperada, dtype=np.float64)
    m = 0.5 * (p + q)

    kl_p_m = _kl_divergencia(p, m, base=2.0)
    kl_q_m = _kl_divergencia(q, m, base=2.0)
    valor = 0.5 * kl_p_m + 0.5 * kl_q_m

    return {"valor": valor, "unidad": "bits (log base 2), acotado en [0, 1]"}


def comparar_con_benford(resultado_benford: dict) -> dict:
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