from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn
import os

from app.core.config import settings
from app.api.router import api_router
from app.utils.logger import logger, api_logger

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加请求处理中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录API请求
        api_logger.info(
            f"{request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Process Time: {process_time:.4f}s"
        )
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        api_logger.error(
            f"{request.method} {request.url.path} "
            f"- Error: {str(e)} "
            f"- Process Time: {process_time:.4f}s"
        )
        
        return JSONResponse(
            status_code=500,
            content={"detail": f"内部服务器错误: {str(e)}"}
        )

# 注册API路由
app.include_router(api_router, prefix="/api")

# 根路由
@app.get("/")
async def root():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs_url": "/docs",
        "api_prefix": "/api"
    }

# 直接运行时的入口
if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    os.makedirs(settings.BACKUPS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
    
    # 启动服务
    logger.info(f"启动 {settings.APP_NAME} 服务")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    ) 