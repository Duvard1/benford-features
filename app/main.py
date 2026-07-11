"""
main.py

Punto de entrada de la aplicación FastAPI.
Su única responsabilidad es crear la app e incluir las rutas
definidas en routes.py.
"""

from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="Benford Audio Analysis API",
    description=(
        "MVP para analizar si la Ley de Benford permite distinguir "
        "voces reales de voces clonadas por IA, a partir de "
        "representaciones numéricas extraídas del audio (FFT, PSD, "
        "MFCC, etc.). Etapa actual: preprocesamiento del audio."
    ),
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    """Endpoint de salud, útil para verificar que la API está viva."""
    return {"status": "ok", "mensaje": "Benford Audio Analysis API funcionando"}
