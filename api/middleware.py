from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

async def logging_middleware(request: Request, call_next):
    logger.info(f"{request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"{request.method} {request.url} -> {response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "error": "Internal Server Error"})