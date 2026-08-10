"""学习资料助手的最小 FastAPI 服务。"""

import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, AsyncIterator, Literal
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.langchain_rag import LangChainRAGError, source_label
from app.material_ingestion import (
    MaterialConflictError,
    MaterialDeleteResult,
    MaterialFile,
    MaterialIndexError,
    MaterialManager,
    MaterialNotFoundError,
    MaterialRollbackError,
    MaterialSyncResult,
    MaterialTooLargeError,
    MaterialValidationError,
    StagedMaterial,
)
from app.operation_guard import (
    OperationGuard,
    OperationProtectionError,
)
from app.rag_service import (
    RAGService,
    RAGServiceInitializationError,
    create_rag_service,
)
from app.request_history import RequestHistoryError, RequestHistoryWriter
from app.url_safety import UnsafeURLError
from app.web_materials import (
    WebMaterialConversionError,
    WebMaterialFetchError,
    WebMaterialPreview,
    WebMaterialService,
    WebMaterialTooLargeError,
    WebMaterialValidationError,
)


MAX_QUESTION_LENGTH = 2000
REQUEST_ID_HEADER = "X-Request-ID"
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
STATIC_DIR = WEB_DIR / "static"
ERROR_PAGE = WEB_DIR / "error.html"
SECURITY_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
_service_lock = Lock()
request_logger = logging.getLogger("uvicorn.error")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def strip_and_validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("问题不能为空。")
        return question


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class StagedMaterialResponse(BaseModel):
    upload_id: str
    filename: str
    operation: Literal["add", "replace"]
    status: Literal["staged"] = "staged"
    size_bytes: int
    document_units: int
    chunk_count: int
    embedding_batch_count: int


class MaterialIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_api_cost: Literal[True]


class MaterialIndexResponse(BaseModel):
    filename: str
    operation: Literal["add", "replace"]
    status: Literal["indexed"] = "indexed"
    added: int
    deleted: int
    unchanged: int
    cleanup_pending: bool


class MaterialResponse(BaseModel):
    filename: str
    size_bytes: int
    status: Literal["stored"] = "stored"


class MaterialListResponse(BaseModel):
    materials: list[MaterialResponse]


class MaterialDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_delete: Literal[True]


class MaterialDeleteResponse(BaseModel):
    filename: str
    status: Literal["deleted"] = "deleted"
    deleted_records: int
    cleanup_pending: bool


class WebMaterialPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    operation: Literal["add", "replace"] = "add"


class WebMaterialPreviewResponse(BaseModel):
    upload_id: str
    filename: str
    operation: Literal["add", "replace"]
    status: Literal["staged"] = "staged"
    requested_url: str
    canonical_url: str
    title: str
    crawled_at: str
    content_sha256: str
    markdown: str
    redirect_count: int
    size_bytes: int
    document_units: int
    chunk_count: int
    embedding_batch_count: int


@asynccontextmanager
async def lifespan(api: FastAPI) -> AsyncIterator[None]:
    """在进程退出时释放按需创建的 Chroma 客户端。"""
    try:
        manager = getattr(api.state, "material_manager", None)
        if manager is not None:
            try:
                cleaned = manager.cleanup_stale_pending_uploads()
            except Exception:
                cleaned = 0
                request_logger.warning(
                    json.dumps(
                        {"event": "stale_pending_upload_cleanup_failed"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            if cleaned:
                request_logger.info(
                    json.dumps(
                        {"event": "stale_pending_uploads_cleaned", "count": cleaned},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        yield
    finally:
        service = getattr(api.state, "rag_service", None)
        try:
            if service is not None:
                service.close()
        finally:
            api.state.rag_service = None


app = FastAPI(
    title="智能学习资料助手 API",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.rag_service = None
app.state.material_manager = MaterialManager()
app.state.request_history_writer = RequestHistoryWriter()
app.state.operation_guard = OperationGuard()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def error_category_for_status(status_code: int) -> str | None:
    """把 HTTP 状态码映射为稳定、无敏感详情的错误类别。"""
    if status_code < 400:
        return None
    if status_code == status.HTTP_422_UNPROCESSABLE_CONTENT:
        return "request_validation"
    if status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        return "rate_limited"
    if status_code == status.HTTP_502_BAD_GATEWAY:
        return "rag_processing"
    if status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return "rag_unavailable"
    if status_code < 500:
        return "client_error"
    return "server_error"


def request_route_path(request: Request) -> str:
    """只记录已匹配的路由模板，避免记录查询参数或任意未知路径。"""
    route = request.scope.get("route")
    if getattr(route, "name", None) == "not_found_page":
        return "unmatched"
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    request_path = request.scope.get("path")
    if isinstance(request_path, str) and (
        request_path == "/static" or request_path.startswith("/static/")
    ):
        return "/static"
    return "unmatched"


def write_request_log(
    request: Request,
    *,
    request_id: str,
    status_code: int,
    elapsed_ms: int,
    error_category: str | None = None,
) -> None:
    """持久化并向 Uvicorn 控制台写入隐私安全的单行 JSON 请求日志。"""
    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "event": "http_request_completed",
        "request_id": request_id,
        "method": request.method,
        "path": request_route_path(request),
        "status_code": status_code,
        "elapsed_ms": elapsed_ms,
        "error_category": error_category or error_category_for_status(status_code),
    }
    history_writer = getattr(request.app.state, "request_history_writer", None)
    if history_writer is not None:
        try:
            history_writer.write(record)
        except RequestHistoryError:
            request_logger.warning(
                json.dumps(
                    {"event": "request_history_write_failed"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    request_logger.info(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    )


def prefers_html_error(request: Request) -> bool:
    """只为普通浏览器页面请求返回 HTML，API 和静态资源仍使用 JSON。"""
    accept = request.headers.get("accept", "")
    path = request.scope.get("path", "")
    return (
        request.method == "GET"
        and "text/html" in accept
        and not str(path).startswith(("/api/", "/static/", "/docs", "/redoc"))
        and path != "/health"
    )


@app.middleware("http")
async def log_http_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """为每次 HTTP 请求生成 request_id、耗时和最小结构化日志。"""
    request_id = uuid4().hex
    request.state.request_id = request_id
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        if prefers_html_error(request):
            response = FileResponse(
                ERROR_PAGE,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                headers=SECURITY_RESPONSE_HEADERS,
            )
        else:
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "服务器内部错误。"},
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        write_request_log(
            request,
            request_id=request_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            elapsed_ms=round((perf_counter() - started) * 1000),
            error_category="unhandled_exception",
        )
        return response

    response.headers[REQUEST_ID_HEADER] = request_id
    write_request_log(
        request,
        request_id=request_id,
        status_code=response.status_code,
        elapsed_ms=round((perf_counter() - started) * 1000),
        error_category=getattr(request.state, "error_category", None),
    )
    return response


def get_rag_service(request: Request) -> RAGService:
    """首次问答时初始化服务，之后在当前进程内复用。"""
    service = getattr(request.app.state, "rag_service", None)
    if service is not None:
        return service

    with _service_lock:
        service = getattr(request.app.state, "rag_service", None)
        if service is not None:
            return service
        try:
            service = create_rag_service()
        except RAGServiceInitializationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG 服务暂不可用。",
            ) from error
        request.app.state.rag_service = service
        return service


def get_rag_service_provider(request: Request) -> Callable[[], RAGService]:
    """注入惰性获取器，避免非法请求提前初始化 RAG。"""
    return lambda: get_rag_service(request)


def get_material_manager(request: Request) -> MaterialManager:
    """返回当前进程复用的资料管理服务。"""
    return request.app.state.material_manager


def get_operation_guard(request: Request) -> OperationGuard:
    """返回当前进程共享的并发、频率和费用保护器。"""
    return request.app.state.operation_guard


def get_web_material_service(request: Request) -> WebMaterialService:
    """基于当前资料管理器构造轻量网页预览服务。"""
    return WebMaterialService(request.app.state.material_manager)


def invalidate_rag_service(api: FastAPI) -> None:
    """资料索引变化后关闭旧 Chroma 客户端，让下一次问答重新初始化。"""
    with _service_lock:
        service = getattr(api.state, "rag_service", None)
        api.state.rag_service = None
    if service is not None:
        try:
            service.close()
        except Exception:
            request_logger.error(
                json.dumps(
                    {"event": "rag_service_refresh_cleanup_failed"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )


def raise_material_http_error(error: Exception) -> None:
    """将资料管理错误映射为不泄露内部路径和配置的 HTTP 错误。"""
    if isinstance(error, MaterialTooLargeError):
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    if isinstance(error, MaterialValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    if isinstance(error, MaterialConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    if isinstance(error, MaterialNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    if isinstance(error, MaterialRollbackError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="资料与索引未能恢复一致，请暂停操作并检查服务。",
        ) from error
    if isinstance(error, MaterialIndexError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="资料索引处理失败，原有资料状态已恢复。",
        ) from error
    raise error


def raise_operation_http_error(error: OperationProtectionError) -> None:
    """把保护拒绝映射为可重试且不泄露内部状态的 429。"""
    headers = None
    if error.retry_after is not None:
        headers = {"Retry-After": str(error.retry_after)}
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=str(error),
        headers=headers,
    ) from error


def raise_web_material_http_error(error: Exception) -> None:
    """将网页预览错误映射为不泄露目标响应和内部异常的 HTTP 错误。"""
    if isinstance(error, WebMaterialTooLargeError):
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error
    if isinstance(error, (UnsafeURLError, WebMaterialValidationError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    if isinstance(error, WebMaterialConversionError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="网页 Markdown 预览服务暂不可用。",
        ) from error
    if isinstance(error, WebMaterialFetchError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="公开网页抓取失败。",
        ) from error
    raise error


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """只检查 API 进程可响应，不调用 Embedding、Chat 或 Chroma。"""
    return HealthResponse()


@app.get("/", include_in_schema=False)
def web_page() -> FileResponse:
    """返回同源的最小问答页面，不初始化 RAG 或调用模型。"""
    return FileResponse(
        WEB_DIR / "index.html",
        headers=SECURITY_RESPONSE_HEADERS,
    )


@app.get("/api/materials", response_model=MaterialListResponse)
def list_materials(
    manager: Annotated[MaterialManager, Depends(get_material_manager)],
) -> MaterialListResponse:
    """零费用列出正式资料文件，不解析内容或打开 Chroma。"""
    materials: tuple[MaterialFile, ...] = manager.list_materials()
    return MaterialListResponse(
        materials=[
            MaterialResponse(
                filename=material.filename,
                size_bytes=material.size_bytes,
            )
            for material in materials
        ]
    )


@app.post(
    "/api/materials/stage",
    response_model=StagedMaterialResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def stage_material_upload(
    file: Annotated[UploadFile, File(description="TXT、Markdown 或 PDF 学习资料")],
    manager: Annotated[MaterialManager, Depends(get_material_manager)],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    operation: Annotated[Literal["add", "replace"], Form()] = "add",
) -> StagedMaterialResponse:
    """零费用暂存并解析上传文件；不会读取模型配置或创建 Embedding。"""
    try:
        with guard.acquire("stage"):
            staged: StagedMaterial = manager.stage_upload(
                filename=file.filename,
                content_type=file.content_type,
                stream=file.file,
                operation=operation,
            )
    except OperationProtectionError as error:
        raise_operation_http_error(error)
        raise
    except Exception as error:
        raise_material_http_error(error)
        raise
    finally:
        file.file.close()
    return StagedMaterialResponse(
        upload_id=staged.upload_id,
        filename=staged.filename,
        operation=staged.operation,
        size_bytes=staged.size_bytes,
        document_units=staged.document_units,
        chunk_count=staged.chunk_count,
        embedding_batch_count=staged.embedding_batch_count,
    )


@app.post(
    "/api/web-materials/preview",
    response_model=WebMaterialPreviewResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def preview_web_material(
    payload: WebMaterialPreviewRequest,
    request: Request,
    service: Annotated[WebMaterialService, Depends(get_web_material_service)],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
) -> WebMaterialPreviewResponse:
    """抓取公开单页并暂存 Markdown；不会打开 Chroma 或调用 Embedding。"""
    try:
        with guard.acquire("web_preview"):
            preview: WebMaterialPreview = await service.preview(
                payload.url,
                operation=payload.operation,
            )
    except OperationProtectionError as error:
        request.state.error_category = "web_preview_protected"
        raise_operation_http_error(error)
        raise
    except Exception as error:
        request.state.error_category = "web_preview"
        if isinstance(
            error,
            (
                MaterialTooLargeError,
                MaterialValidationError,
                MaterialConflictError,
                MaterialNotFoundError,
                MaterialRollbackError,
                MaterialIndexError,
            ),
        ):
            raise_material_http_error(error)
            raise
        raise_web_material_http_error(error)
        raise
    return WebMaterialPreviewResponse(
        upload_id=preview.upload_id,
        filename=preview.filename,
        operation=preview.operation,
        requested_url=preview.requested_url,
        canonical_url=preview.canonical_url,
        title=preview.title,
        crawled_at=preview.crawled_at,
        content_sha256=preview.content_sha256,
        markdown=preview.markdown,
        redirect_count=preview.redirect_count,
        size_bytes=preview.size_bytes,
        document_units=preview.document_units,
        chunk_count=preview.chunk_count,
        embedding_batch_count=preview.embedding_batch_count,
    )


@app.post(
    "/api/materials/{upload_id}/index",
    response_model=MaterialIndexResponse,
)
def index_staged_material(
    upload_id: str,
    payload: MaterialIndexRequest,
    request: Request,
    manager: Annotated[MaterialManager, Depends(get_material_manager)],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
) -> MaterialIndexResponse:
    """取得明确费用确认后提交文件并复用现有 Chroma 增量同步。"""
    del payload
    try:
        with guard.acquire("index") as lease:
            lease.reserve_units(manager.estimate_index_batches(upload_id))
            result: MaterialSyncResult = manager.commit_staged(upload_id)
            invalidate_rag_service(request.app)
    except OperationProtectionError as error:
        raise_operation_http_error(error)
        raise
    except Exception as error:
        if isinstance(error, MaterialRollbackError):
            invalidate_rag_service(request.app)
        raise_material_http_error(error)
        raise
    return MaterialIndexResponse(
        filename=result.filename,
        operation=result.operation,
        added=result.added,
        deleted=result.deleted,
        unchanged=result.unchanged,
        cleanup_pending=result.cleanup_pending,
    )


@app.delete(
    "/api/materials/{filename}",
    response_model=MaterialDeleteResponse,
)
def delete_material(
    filename: str,
    payload: MaterialDeleteRequest,
    request: Request,
    manager: Annotated[MaterialManager, Depends(get_material_manager)],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
) -> MaterialDeleteResponse:
    """取得明确删除确认后移除资料及对应 Chroma 记录。"""
    del payload
    try:
        with guard.acquire("delete"):
            result: MaterialDeleteResult = manager.delete_material(filename)
            invalidate_rag_service(request.app)
    except OperationProtectionError as error:
        raise_operation_http_error(error)
        raise
    except Exception as error:
        if isinstance(error, MaterialRollbackError):
            invalidate_rag_service(request.app)
        raise_material_http_error(error)
        raise
    return MaterialDeleteResponse(
        filename=result.filename,
        deleted_records=result.deleted_records,
        cleanup_pending=result.cleanup_pending,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask_documents(
    payload: AskRequest,
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    service_provider: Annotated[
        Callable[[], RAGService],
        Depends(get_rag_service_provider),
    ],
) -> AskResponse:
    """调用现有 RAG 服务并返回答案及可核对来源。"""
    try:
        with guard.acquire("ask", units=1):
            service = service_provider()
            result = service.ask(payload.question)
    except OperationProtectionError as error:
        raise_operation_http_error(error)
        raise
    except LangChainRAGError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="RAG 问答处理失败。",
        ) from error

    return AskResponse(
        answer=result.answer,
        sources=[source_label(document) for document in result.sources],
    )


@app.get(
    "/{unmatched_path:path}",
    include_in_schema=False,
    name="not_found_page",
)
def not_found_page(unmatched_path: str, request: Request) -> Response:
    """返回不包含用户路径或内部信息的统一安全错误页。"""
    del unmatched_path
    if not prefers_html_error(request):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "页面或接口不存在。"},
        )
    return FileResponse(
        ERROR_PAGE,
        status_code=status.HTTP_404_NOT_FOUND,
        headers=SECURITY_RESPONSE_HEADERS,
    )
