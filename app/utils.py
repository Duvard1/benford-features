#Funciones auxiliares de propósito general:

import uuid
from pathlib import Path

# Extensiones de audio que el sistema acepta como entrada.
# Todo lo que no esté en esta lista será rechazado en el endpoint.
ALLOWED_EXTENSIONS = {".aac", ".mp3", ".wav", ".m4a", ".ogg", ".opus"}


def validar_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        permitidas = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Formato de archivo no soportado: '{extension}'. "
            f"Formatos permitidos: {permitidas}"
        )
    return extension


def generar_nombre_unico(extension: str) -> str:
    if not extension.startswith("."):
        extension = f".{extension}"
    return f"{uuid.uuid4().hex}{extension}"
