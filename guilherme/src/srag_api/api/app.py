from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from srag_api.api.models import HealthResponse
from srag_api.api.routes.epidemiology import router


app = FastAPI(
    title="SRAG Epidemiological API",
    description="API publica e somente-leitura para consultas epidemiologicas SRAG/SIVEP-Gripe.",
    version="0.3.0",
)
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return {"status": "ok", "service": "srag-api"}


@app.exception_handler(FileNotFoundError)
async def parquet_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def domain_value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
