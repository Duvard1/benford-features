from fastapi import FastAPI
from app.routes import router

# Crear Proyecto  FastAPI
app = FastAPI(
    title="Benford Features",
    description=(
        "Analizar si la Ley de Benford permite distinguir "
        "voces reales de voces clonadas por IA, a partir de "
        "representaciones numéricas extraídas del audio (FFT, PSD, "
        "MFCC, etc.)."
    ),
    version="0.1.0",
)

# Incluir rutas
app.include_router(router)

#Health Check
@app.get("/")
def root():
    return {"status": "ok", "mensaje": "Benford Features funcionando"}
