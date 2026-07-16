# Benford features

Análisis de audio con la Ley de Benford para distinguir voces reales de voces clonadas por IA. Flujo completo implementado para múltiples representaciones de audio: conversión/estandarización del audio, extracción de características espectrales/cepstrales, aplicación de la Ley de Benford, y cálculo de métricas de desviación ($\chi^2$, MAD, KL, JS).
La API soporta múltiples representaciones para analizar el comportamiento y la distribución de sus primeros dígitos significativos.

---

## Requisitos previoss

- Python 3.10+
- **FFmpeg instalado en el sistema** (no es una librería de Python, es un binario). La conversión se invoca desde Python vía `subprocess`.

Verificar que esté instalado:

```bash
ffmpeg -version
```

Si no lo tienes:
- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: descargar desde https://ffmpeg.org/download.html y agregarlo al PATH

---

## Instalación

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## Ejecutar el servidor

```bash
uvicorn app.main:app --reload
```

La API quedará disponible en: http://127.0.0.1:8000
Documentación interactiva automática (Swagger): http://127.0.0.1:8000/docs

---

## Probar el endpoint

Para realizar un análisis, envía una petición POST al endpoint `/benford-features` especificando el archivo de audio y la representación espectral deseada (`feature`):

```bash
curl -X POST http://127.0.0.1:8000/benford-features \
  -F "file=@ruta/a/tu/audio.aac" \
  -F "feature=fft"
```

### Parámetros de `feature` soportados:
- **`fft`**: ¿Qué frecuencias tiene el audio?
- **`psd`**: ¿Cuánta energía tiene cada frecuencia?
- **`stft`**: ¿Qué frecuencias aparecen en cada instante?
- **`mel`**: Igual que la STFT, pero imitando el oído humano.
- **`logmel`**: Mel Spectrogram con volumen en escala logarítmica.
- **`mfcc`**: Un resumen matemático del Log-Mel.

### Ejemplo de respuesta esperada (para `feature=fft`):

```json
{
  "status": "ok",
  "archivo_original": "audio.aac",
  "feature": "fft",
  "audio_convertido": {
    "sample_rate": 16000,
    "canales": 1,
    "num_muestras": 48669,
    "duracion_segundos": 3.042,
    "wav_path": "temp/9e1f96867d7a48c2a047f4eac6f7c368.wav"
  },
  "fft_info": {
    "num_ventanas": 94,
    "bins_por_ventana": 513,
    "total_magnitudes": 48222,
    "magnitudes_stats": {
      "min": 3.07e-07,
      "max": 22.63,
      "mean": 0.0929
    },
    "magnitudes_path": "temp/e7d0f0ab04e24888b7614fcfa7f48f18.npy"
  },
  "benford": {
    "total_valores": 48222,
    "valores_excluidos": 0,
    "total_valores_validos": 48222,
    "digitos": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "conteo_observado": [19531, 11947, 4814, 2346, 1809, 1870, 1869, 2071, 1965],
    "frecuencia_observada": [0.405, 0.2477, 0.0998, 0.0487, 0.0375, 0.0388, 0.0388, 0.0429, 0.0407],
    "frecuencia_esperada_benford": [0.301, 0.1761, 0.1249, 0.0969, 0.0792, 0.0669, 0.058, 0.0512, 0.0458]
  },
  "metricas": {
    "chi_cuadrado": {
      "estadistico": 8951.14,
      "grados_libertad": 8,
      "p_valor": 0.0,
      "significativo_al_5pct": true
    },
    "mad": {
      "valor": 0.0396,
      "interpretacion": "No conformidad (nonconformity)"
    },
    "kl_divergencia": {
      "valor": 0.1061,
      "unidad": "bits (log base 2)"
    },
    "js_divergencia": {
      "valor": 0.0275,
      "unidad": "bits (log base 2), acotado en [0, 1]"
    }
  }
}
```

> [!NOTE]
>`valores_excluidos` cuenta cuántos valores eran exactamente 0 (silencio en esa frecuencia/ventana) y por lo tanto no tienen primer dígito significativo definido. Se excluyen del análisis pero se reportan.

---

## Cómo interpretar las métricas

- **MAD (Mean Absolute Deviation)**: Promedio de diferencias absolutas entre las frecuencias. Se interpreta bajo los umbrales de Nigrini (2012):
  * `< 0.006`: Conformidad cercana (close conformity).
  * `0.006 - 0.012`: Conformidad aceptable (acceptable conformity).
  * `0.012 - 0.015`: Conformidad marginal (marginally acceptable).
  * `> 0.015`: No conformidad (nonconformity).
- **$\chi^2$ (chi-cuadrado)**: Evalúa si la diferencia entre la distribución observada y la distribución de Benford es estadísticamente significativa. Un `p_valor < 0.05` indica no conformidad estadística. Debido al gran volumen de datos, suele ser muy sensible.
- **KL (Kullback-Leibler)**: Medida asimétrica que penaliza más fuertemente las diferencias en los dígitos menos frecuentes de Benford (7, 8, 9).
- **JS (Jensen-Shannon)**: Versión simétrica y acotada en $[0, 1]$ de la divergencia KL. Muy adecuada para comparar directamente la similitud entre dos audios (ej. real vs clonado).

---

## Estructura del proyecto

```
Ley_Benford/
│
├── app/
│   ├── __init__.py
│   ├── main.py        # Punto de entrada de FastAPI
│   ├── routes.py      # Definición de endpoints (POST /benford-features)
│   ├── audio.py       # Conversión y estandarización del audio a WAV PCM mono 16kHz
│   ├── fft.py         # Extracción de magnitudes de la FFT real con ventaneo Hann
│   ├── psd.py         # Estimación de Power Spectral Density por ventana (Periodograma)
│   ├── stft.py        # Cálculo del espectrograma STFT (magnitudes normalizadas)
│   ├── mel.py         # Mel Espectrograma en potencia lineal (sin compresión logarítmica)
│   ├── logmel.py      # Log-Mel Espectrograma (bandas Mel convertidas a escala en dB)
│   ├── mfcc.py        # Coeficientes Cepstrales en las Frecuencias de Mel (MFCC, 13 coefs)
│   ├── benford.py     # Análisis de Benford (primer dígito significativo e histograma)
│   ├── metrics.py     # Cálculo de métricas de desviación (χ², MAD, KL y JS)
│   └── utils.py       # Utilidades (validación de extensiones y nombres únicos)
│
├── uploads/           # Audios originales subidos para análisis
├── temp/              # Audios temporales WAV y archivos .npy
└── requirements.txt   # Librerías de Python requeridas
```

---

## Decisiones Metodológicas y de Diseño

### 1. FFT Real (`rfft`) frente a FFT Completa
Para señales reales como el audio, el espectro de la FFT es simétrico (la segunda mitad es un espejo redundante de la primera). Para evitar duplicidad de dígitos, se utiliza `np.fft.rfft` que entrega solo los $N/2 + 1$ bins únicos (513 bins para un tamaño de ventana de 1024).

### 2. Welch vs. Periodograma Individual (PSD)
En vez de utilizar el método Welch convencional (que promedia los periodogramas en una única firma de frecuencias), calculamos el periodograma individual para cada ventana ($|X(k)|^2 / (f_s \cdot \sum w^2)$). Esto conserva la variabilidad temporal y provee el volumen masivo de datos necesario para el correcto análisis de la Ley de Benford.

### 3. STFT Normalizada
A diferencia de la FFT cruda, la STFT de `scipy.signal.stft` normaliza las magnitudes dividiéndolas por la suma de la ventana ($\sum w$), logrando magnitudes invariables ante la escala de la ventana.

### 4. Mel vs. Log-Mel
Permite comparar de forma aislada el impacto de la compresión logarítmica frente a la aplicación del banco de filtros Mel. Dado que `librosa.power_to_db` genera valores típicamente negativos en dB, en `logmel.py` se aplica un desplazamiento constante basado en el valor mínimo para asegurar que todos los valores ingresados a la Ley de Benford sean estrictamente positivos ($> 0$), lo cual no afecta a los dígitos relativos.

### 5. Tratamiento de MFCC (Valor Absoluto)
Los coeficientes MFCC pueden ser negativos. Para evitar sesgar el análisis de Benford con desplazamientos artificiales gigantescos (que dominarían el primer dígito con el dígito del desplazamiento), se utiliza el **valor absoluto** de los coeficientes ($|MFCC| + \epsilon$), conservando intacto su rango dinámico original.

---

## Qué hace esta etapa (y qué NO hace todavía)

**Hace:**
1. Recibe cualquier formato de audio soportado (aac, mp3, wav, m4a, ogg).
2. Estandariza a WAV PCM 16kHz Mono.
3. Extrae la característica espectral solicitada (`fft`, `psd`, `stft`, `mel`, `logmel`, `mfcc`).
4. Extrae primeros dígitos y calcula la distribución de frecuencias observadas.
5. Calcula todas las métricas de desviación estadística respecto a la Ley de Benford ($\chi^2$, MAD, KL, JS).

**Todavía NO hace:**
- Modelado predictivo o clasificación automática con Machine Learning. La API provee las métricas descriptivas y matemáticas necesarias para que un sistema o modelo superior realice la toma de decisión (clasificar como real o IA).