"""学习资料助手的最小 FastAPI 服务。"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Any, AsyncIterator, Literal
from uuid import UUID, uuid4

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
from pydantic import BaseModel, ConfigDict, Field, field_validator
from fastapi.staticfiles import StaticFiles
from app.auth import (
    AUTH_COOKIE_NAME,
    AUTH_SESSION_TTL,
    AuthenticationCredentialsError,
    AuthenticationConflictError,
    AuthenticationService,
    AuthenticationValidationError,
)

from app.agent_service import (
    AGENT_MAX_MESSAGE_LENGTH,
    AgentExecutionError,
    AgentResult,
    AgentService,
    AgentTimeoutError,
    create_agent_service,
)
from app.document_jobs import (
    DOCUMENT_JOB_DB_PATH,
    DocumentJobConflictError,
    DocumentJobError,
    DocumentJobNotFoundError,
    DocumentJobRecord,
    DocumentJobService,
)
from app.langchain_rag import LangChainRAGError, source_label
from app.learning_data import (
    MAX_HISTORY_ITEMS,
    ConversationMessageRecord,
    LearningActivityRecord,
    LearningDataConflictError,
    LearningDataNotFoundError,
    LearningDataStore,
    LearningSessionRecord,
    UserRecord,
)
from app.material_ingestion import (
    BatchStageResult,
    MaterialConflictError,
    MaterialDeleteResult,
    MaterialFile,
    MaterialIndexError,
    MaterialIndexReadOnlyError,
    MaterialManager,
    MaterialNotFoundError,
    MaterialRollbackError,
    MaterialTooLargeError,
    MaterialUpload,
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
from app.security_limits import MAX_UPLOAD_FILES_PER_BATCH
from app.study_workflow import (
    WORKFLOW_MAX_GOAL_LENGTH,
    WORKFLOW_MAX_PROGRESS_NOTE_LENGTH,
    ApprovalDecision,
    StudyWorkflowConflictError,
    StudyWorkflowNotFoundError,
    StudyWorkflowResult,
    StudyWorkflowService,
    WorkflowStatus,
    open_sqlite_study_workflow_service,
)
from app.tutor_workflow import (
    TUTOR_MAX_MESSAGE_LENGTH,
    LearningAction,
    LearningSummaryDraft,
    QuizDraft,
    TutorExecutionError,
    TutorIntent,
    TutorResult,
    TutorTimeoutError,
    TutorWorkflowService,
    create_tutor_workflow_service,
)
from app.url_safety import UnsafeURLError, proxy_fake_ip_compatibility_enabled
from app.web_materials import (
    WebMaterialConversionError,
    WebMaterialFetchError,
    WebMaterialPreview,
    WebMaterialService,
    WebMaterialTooLargeError,
    WebMaterialValidationError,
)
from app.user_workspace import (
    USER_WORKSPACES_DIR,
    create_user_material_manager,
    user_workspace_paths,
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
_material_manager_lock = Lock()
_workflow_service_lock = asyncio.Lock()
_learning_data_lock = asyncio.Lock()
_tutor_service_lock = asyncio.Lock()
_document_job_service_lock = asyncio.Lock()
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


class CitationResponse(BaseModel):
    citation_id: str
    evidence_id: str
    material_id: str
    chunk_id: str
    source: str
    filename: str
    page: int | None
    chunk_index: int
    excerpt: str
    locator: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    citations: list[CitationResponse]


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=AGENT_MAX_MESSAGE_LENGTH)
    confirm_api_cost: Literal[True]
    allow_web_preview: bool = False

    @field_validator("message")
    @classmethod
    def strip_and_validate_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Agent 消息不能为空。")
        return message


class AgentResponse(BaseModel):
    answer: str
    sources: list[str]
    tools_used: list[str]


class TutorChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=TUTOR_MAX_MESSAGE_LENGTH)
    session_id: UUID
    confirm_api_cost: Literal[True]

    @field_validator("message")
    @classmethod
    def strip_and_validate_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Tutor 消息不能为空。")
        return message


class TutorEvidenceResponse(BaseModel):
    context_id: str
    evidence_id: str
    material_id: str
    chunk_id: str
    source: str
    filename: str
    source_type: str
    page: int | None
    section: str | None
    chunk_index: int
    excerpt: str
    locator: str
    content_hash: str
    canonical_url: str | None


class TutorChatResponse(BaseModel):
    session_id: str
    intent: TutorIntent
    topic: str
    answer: str
    sources: list[str]
    citations: list[CitationResponse]
    evidence: list[TutorEvidenceResponse]
    learning_action: LearningAction
    quiz: QuizDraft | None
    summary: LearningSummaryDraft | None
    tools_used: list[str]
    route_trace: list[str]


class AuthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(AuthRequest):
    display_name: str = Field(min_length=1, max_length=80)


class UserResponse(BaseModel):
    user_id: UUID
    email: str | None
    display_name: str | None
    created_at: datetime
    updated_at: datetime | None


class LearningSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1, max_length=200)

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, value: str) -> str:
        topic = value.strip()
        if not topic:
            raise ValueError("学习主题不能为空。")
        return topic


class LearningSessionResponse(BaseModel):
    session_id: UUID
    user_id: UUID
    topic: str
    created_at: datetime
    updated_at: datetime


class LearningSessionListResponse(BaseModel):
    sessions: list[LearningSessionResponse]


class ConversationMessageResponse(BaseModel):
    message_id: UUID
    session_id: UUID
    role: Literal["user", "tutor"]
    content: str
    intent: TutorIntent | None
    created_at: datetime


class LearningActivityResponse(BaseModel):
    record_id: UUID
    user_id: UUID
    session_id: UUID
    topic: str
    activity_type: str
    created_at: datetime
    metadata: dict[str, Any]


class LearningHistoryResponse(BaseModel):
    messages: list[ConversationMessageResponse]
    learning_records: list[LearningActivityResponse]


class StudyWorkflowStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=WORKFLOW_MAX_GOAL_LENGTH)
    confirm_api_cost: Literal[True]

    @field_validator("goal")
    @classmethod
    def strip_and_validate_goal(cls, value: str) -> str:
        goal = value.strip()
        if not goal:
            raise ValueError("学习目标不能为空。")
        return goal


class StudyWorkflowConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ApprovalDecision


class StudyWorkflowProgressRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str = Field(min_length=1, max_length=WORKFLOW_MAX_PROGRESS_NOTE_LENGTH)
    complete_current_task: bool

    @field_validator("note")
    @classmethod
    def strip_and_validate_note(cls, value: str) -> str:
        note = value.strip()
        if not note:
            raise ValueError("进度记录不能为空。")
        return note


class StudyWorkflowRetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_api_cost: Literal[True]


class StudyWorkflowDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_delete: Literal[True]


class StudyWorkflowTaskResponse(BaseModel):
    id: str
    title: str
    status: Literal["pending", "in_progress", "completed"]


class StudyWorkflowResponse(BaseModel):
    workflow_id: str
    goal: str
    status: WorkflowStatus
    evidence_summary: str
    sources: list[str]
    tasks: list[StudyWorkflowTaskResponse]
    current_task_index: int
    progress_history: list[str]
    retry_count: int
    review: str
    route_trace: list[str]
    error: str
    approval_required: bool


class StudyWorkflowDeleteResponse(BaseModel):
    workflow_id: str
    status: Literal["deleted"] = "deleted"


class StagedMaterialResponse(BaseModel):
    upload_id: str
    filename: str
    operation: Literal["add", "replace"]
    status: Literal["staged"] = "staged"
    size_bytes: int
    document_units: int
    chunk_count: int
    embedding_batch_count: int


class StagedMaterialFailureResponse(BaseModel):
    filename: str
    reason: str


class BatchStagedMaterialResponse(BaseModel):
    status: Literal["staged", "partial", "failed"]
    staged: list[StagedMaterialResponse]
    failures: list[StagedMaterialFailureResponse]
    total_files: int
    staged_count: int
    failed_count: int
    total_chunks: int
    embedding_batch_count: int


class MaterialIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_api_cost: Literal[True]


class MaterialBatchIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_UPLOAD_FILES_PER_BATCH,
    )
    confirm_api_cost: Literal[True]

    @field_validator("upload_ids")
    @classmethod
    def reject_duplicate_upload_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("upload_ids 不能重复。")
        return value


class DocumentJobResultResponse(BaseModel):
    filenames: list[str]
    added: int
    deleted: int
    unchanged: int
    cleanup_pending: bool


class DocumentJobResponse(BaseModel):
    job_id: str
    filenames: list[str]
    status: Literal["pending", "processing", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    error_message: str | None
    result: DocumentJobResultResponse | None
    created_at: str
    started_at: str | None
    finished_at: str | None


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
    """在进程退出时释放按需创建的数据库与 Chroma 客户端。"""
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
        if (
            getattr(api.state, "document_job_service", None) is None
            and DOCUMENT_JOB_DB_PATH.is_file()
        ):
            try:
                api.state.document_job_service = await DocumentJobService.open(
                    material_manager_factory=lambda user_id: material_manager_for_user(
                        api, user_id
                    ),
                    operation_guard=api.state.operation_guard,
                    invalidate_rag=lambda user_id: invalidate_rag_service(
                        api, user_id
                    ),
                )
            except Exception:
                request_logger.error(
                    json.dumps(
                        {"event": "document_job_service_recovery_failed"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        yield
    finally:
        document_job_service = getattr(api.state, "document_job_service", None)
        if document_job_service is not None:
            try:
                await document_job_service.close()
            except Exception:
                request_logger.error(
                    json.dumps(
                        {"event": "document_job_service_cleanup_failed"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        api.state.document_job_service = None
        workflow_service = getattr(api.state, "study_workflow_service", None)
        if workflow_service is not None:
            try:
                await workflow_service.close()
            except Exception:
                request_logger.error(
                    json.dumps(
                        {"event": "study_workflow_service_cleanup_failed"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        api.state.study_workflow_service = None
        learning_store = getattr(api.state, "learning_data_store", None)
        if learning_store is not None:
            try:
                await learning_store.close()
            except Exception:
                request_logger.error(
                    json.dumps(
                        {"event": "learning_data_store_cleanup_failed"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        api.state.learning_data_store = None
        services = tuple(getattr(api.state, "rag_services", {}).values())
        for service in services:
            try:
                service.close()
            except Exception:
                request_logger.error(
                    json.dumps(
                        {"event": "rag_service_cleanup_failed"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        api.state.agent_services.clear()
        api.state.tutor_services.clear()
        api.state.rag_services.clear()


app = FastAPI(
    title="智能学习资料助手 API",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.rag_services = {}
app.state.agent_services = {}
app.state.tutor_services = {}
app.state.study_workflow_service = None
app.state.learning_data_store = None
app.state.document_job_service = None
app.state.material_manager = MaterialManager()
app.state.material_managers = {}
app.state.user_workspaces_dir = USER_WORKSPACES_DIR
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


def _auth_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    header_token: str | None = None
    if authorization:
        scheme, separator, credentials = authorization.partition(" ")
        if separator != " " or scheme.casefold() != "bearer" or not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登录状态无效或已过期。",
            )
        header_token = credentials.strip()
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    if header_token and cookie_token and header_token != cookie_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录状态无效或已过期。",
        )
    return header_token or cookie_token


async def get_current_user(request: Request) -> UserRecord:
    """只信任后端验证过的 Cookie 或 Bearer 凭证。"""
    try:
        store = await get_learning_data_store(request)
        user = await AuthenticationService(store).authenticate(_auth_token(request))
    except AuthenticationCredentialsError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录状态无效或已过期。",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error
    request.state.current_user_id = user.user_id
    return user


def get_rag_service(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> RAGService:
    """按认证用户的独立 Chroma 目录惰性创建 RAG 服务。"""
    services = request.app.state.rag_services
    service = services.get(current_user.user_id)
    if service is not None:
        return service

    with _service_lock:
        service = services.get(current_user.user_id)
        if service is not None:
            return service
        workspace = user_workspace_paths(
            current_user.user_id,
            workspaces_dir=request.app.state.user_workspaces_dir,
        )
        try:
            service = create_rag_service(workspace.vector_store)
        except RAGServiceInitializationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG 服务暂不可用。",
            ) from error
        services[current_user.user_id] = service
        return service


def get_rag_service_provider(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> Callable[[], RAGService]:
    """注入惰性获取器，避免非法请求提前初始化 RAG。"""
    return lambda: get_rag_service(request, current_user)


def get_agent_service(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> AgentService:
    """按用户创建受限 Agent，并复用该用户自己的资料与索引。"""
    services = request.app.state.agent_services
    service = services.get(current_user.user_id)
    if service is not None:
        return service

    while True:
        rag_service = get_rag_service(request, current_user)
        with _service_lock:
            service = services.get(current_user.user_id)
            if service is not None:
                return service
            if request.app.state.rag_services.get(current_user.user_id) is not rag_service:
                continue
            manager = get_user_material_manager(request, current_user)
            service = create_agent_service(
                rag_service,
                manager,
                WebMaterialService(
                    manager,
                    allow_proxy_fake_ip=proxy_fake_ip_compatibility_enabled(),
                ),
            )
            services[current_user.user_id] = service
            return service


def get_agent_service_provider(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> Callable[[], AgentService]:
    """返回惰性 Agent 获取器，确保无效或未确认费用的请求不会初始化模型。"""
    return lambda: get_agent_service(request, current_user)


async def get_learning_data_store(request: Request) -> LearningDataStore:
    """按需打开用户学习数据库，不初始化 RAG 或调用模型。"""
    store = getattr(request.app.state, "learning_data_store", None)
    if store is not None:
        return store
    async with _learning_data_lock:
        store = getattr(request.app.state, "learning_data_store", None)
        if store is None:
            try:
                store = await LearningDataStore.open()
            except Exception as error:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="学习数据服务暂不可用。",
                ) from error
            request.app.state.learning_data_store = store
        return store


async def get_tutor_service(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> TutorWorkflowService:
    """按用户创建 Tutor；学习记录与 RAG 都使用同一 current_user。"""
    services = request.app.state.tutor_services
    service = services.get(current_user.user_id)
    if service is not None:
        return service

    learning_store = await get_learning_data_store(request)
    while True:
        rag_service = await asyncio.to_thread(
            get_rag_service,
            request,
            current_user,
        )
        async with _tutor_service_lock:
            with _service_lock:
                service = services.get(current_user.user_id)
                if service is not None:
                    return service
                if request.app.state.rag_services.get(current_user.user_id) is not rag_service:
                    continue
                service = create_tutor_workflow_service(
                    rag_service,
                    learning_store,
                )
                services[current_user.user_id] = service
                return service


def get_tutor_service_provider(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> Callable[[], Awaitable[TutorWorkflowService]]:
    """返回惰性 Tutor 获取器，非法或未确认费用的请求不会初始化模型。"""
    return lambda: get_tutor_service(request, current_user)


async def get_study_workflow_service(request: Request) -> StudyWorkflowService:
    """按需打开本地 SQLite checkpointer，不初始化 RAG 或模型。"""
    service = getattr(request.app.state, "study_workflow_service", None)
    if service is not None:
        return service
    async with _workflow_service_lock:
        service = getattr(request.app.state, "study_workflow_service", None)
        if service is None:
            service = await open_sqlite_study_workflow_service()
            request.app.state.study_workflow_service = service
        return service


def material_manager_for_user(api: FastAPI, user_id: str) -> MaterialManager:
    managers = api.state.material_managers
    manager = managers.get(user_id)
    if manager is not None:
        return manager
    with _material_manager_lock:
        manager = managers.get(user_id)
        if manager is None:
            manager = create_user_material_manager(
                user_id,
                workspaces_dir=api.state.user_workspaces_dir,
            )
            manager.cleanup_stale_pending_uploads()
            managers[user_id] = manager
        return manager


def get_user_material_manager(
    request: Request,
    current_user: UserRecord,
) -> MaterialManager:
    return material_manager_for_user(request.app, current_user.user_id)


def get_material_manager(
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> MaterialManager:
    return get_user_material_manager(request, current_user)


def get_operation_guard(request: Request) -> OperationGuard:
    """返回当前进程共享的并发、频率和费用保护器。"""
    return request.app.state.operation_guard


async def get_document_job_service(
    request: Request,
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> DocumentJobService:
    """按需打开任务数据库并启动单进程后台 Worker。"""
    service = getattr(request.app.state, "document_job_service", None)
    if service is not None:
        return service
    async with _document_job_service_lock:
        service = getattr(request.app.state, "document_job_service", None)
        if service is None:
            try:
                service = await DocumentJobService.open(
                    material_manager_factory=lambda user_id: material_manager_for_user(
                        request.app, user_id
                    ),
                    operation_guard=guard,
                    invalidate_rag=lambda user_id: invalidate_rag_service(
                        request.app, user_id
                    ),
                )
            except Exception as error:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="文档任务服务暂不可用。",
                ) from error
            request.app.state.document_job_service = service
        return service


def get_web_material_service(
    request: Request,
    manager: Annotated[MaterialManager, Depends(get_material_manager)],
) -> WebMaterialService:
    """基于当前资料管理器构造轻量网页预览服务。"""
    return WebMaterialService(
        manager,
        allow_proxy_fake_ip=proxy_fake_ip_compatibility_enabled(),
    )


def invalidate_rag_service(api: FastAPI, user_id: str) -> None:
    """只刷新发生索引变化的用户服务，不影响其他学习空间。"""
    with _service_lock:
        service = api.state.rag_services.pop(user_id, None)
        api.state.agent_services.pop(user_id, None)
        api.state.tutor_services.pop(user_id, None)
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
    if isinstance(error, MaterialIndexReadOnlyError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    if isinstance(error, MaterialIndexError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="资料索引处理失败，原有资料状态已恢复。",
        ) from error
    raise error


def raise_learning_data_http_error(error: Exception) -> None:
    """将学习数据异常映射为不暴露归属关系和数据库细节的响应。"""
    if isinstance(error, LearningDataNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户或学习会话不存在。",
        ) from error
    if isinstance(error, LearningDataConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="学习数据状态冲突。",
        ) from error
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="学习数据服务暂不可用。",
    ) from error


def raise_document_job_http_error(error: Exception) -> None:
    if isinstance(error, DocumentJobNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档处理任务不存在。",
        ) from error
    if isinstance(error, DocumentJobConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    if isinstance(error, ValueError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="文档任务服务暂不可用。",
    ) from error


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


def raise_study_workflow_http_error(error: Exception) -> None:
    """映射工作流状态错误，不泄露检查点路径、正文或内部异常。"""
    if isinstance(error, StudyWorkflowNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="学习工作流不存在。",
        ) from error
    if isinstance(error, StudyWorkflowConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="当前学习工作流状态不允许此操作。",
        ) from error
    raise error


def user_response(record: UserRecord) -> UserResponse:
    return UserResponse(
        user_id=record.user_id,
        email=record.email,
        display_name=record.display_name,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def learning_session_response(
    record: LearningSessionRecord,
) -> LearningSessionResponse:
    return LearningSessionResponse(
        session_id=record.session_id,
        user_id=record.user_id,
        topic=record.topic,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def conversation_message_response(
    record: ConversationMessageRecord,
) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        message_id=record.message_id,
        session_id=record.session_id,
        role=record.role,
        content=record.content,
        intent=record.intent,
        created_at=record.created_at,
    )


def learning_activity_response(
    record: LearningActivityRecord,
) -> LearningActivityResponse:
    return LearningActivityResponse(
        record_id=record.record_id,
        user_id=record.user_id,
        session_id=record.session_id,
        topic=record.topic,
        activity_type=record.activity_type,
        created_at=record.created_at,
        metadata=record.metadata,
    )


def document_job_response(record: DocumentJobRecord) -> DocumentJobResponse:
    result = (
        DocumentJobResultResponse.model_validate(record.result)
        if record.result is not None
        else None
    )
    return DocumentJobResponse(
        job_id=record.job_id,
        filenames=list(record.filenames),
        status=record.status,
        progress=record.progress,
        error_message=record.error_message,
        result=result,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
    )


def study_workflow_response(result: StudyWorkflowResult) -> StudyWorkflowResponse:
    return StudyWorkflowResponse(
        workflow_id=result.workflow_id,
        goal=result.goal,
        status=result.status,
        evidence_summary=result.evidence_summary,
        sources=list(result.sources),
        tasks=[StudyWorkflowTaskResponse(**task) for task in result.tasks],
        current_task_index=result.current_task_index,
        progress_history=list(result.progress_history),
        retry_count=result.retry_count,
        review=result.review,
        route_trace=list(result.route_trace),
        error=result.error,
        approval_required=result.approval_required,
    )


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


@app.post(
    "/api/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    store: Annotated[LearningDataStore, Depends(get_learning_data_store)],
) -> UserResponse:
    try:
        session = await AuthenticationService(store).register(
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
        )
    except AuthenticationConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except AuthenticationValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=session.token,
        max_age=int(AUTH_SESSION_TTL.total_seconds()),
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )
    return user_response(session.user)


@app.post("/api/auth/login", response_model=UserResponse)
async def login_user(
    payload: AuthRequest,
    request: Request,
    response: Response,
    store: Annotated[LearningDataStore, Depends(get_learning_data_store)],
) -> UserResponse:
    try:
        session = await AuthenticationService(store).login(
            email=payload.email,
            password=payload.password,
        )
    except (AuthenticationCredentialsError, AuthenticationValidationError) as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误。",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=session.token,
        max_age=int(AUTH_SESSION_TTL.total_seconds()),
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )
    return user_response(session.user)


@app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout_user(
    request: Request,
    response: Response,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[LearningDataStore, Depends(get_learning_data_store)],
) -> None:
    del current_user
    await AuthenticationService(store).logout(_auth_token(request))
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
    )


@app.get("/api/auth/me", response_model=UserResponse)
async def get_authenticated_user(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> UserResponse:
    return user_response(current_user)


@app.post(
    "/api/sessions",
    response_model=LearningSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_learning_session(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    payload: LearningSessionCreateRequest,
    store: Annotated[LearningDataStore, Depends(get_learning_data_store)],
) -> LearningSessionResponse:
    try:
        record = await store.create_session(current_user.user_id, payload.topic)
        return learning_session_response(record)
    except Exception as error:
        raise_learning_data_http_error(error)
        raise


@app.get(
    "/api/sessions",
    response_model=LearningSessionListResponse,
)
async def list_learning_sessions(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[LearningDataStore, Depends(get_learning_data_store)],
) -> LearningSessionListResponse:
    try:
        records = await store.list_sessions(
            current_user.user_id,
            limit=MAX_HISTORY_ITEMS,
        )
    except Exception as error:
        raise_learning_data_http_error(error)
        raise
    return LearningSessionListResponse(
        sessions=[learning_session_response(record) for record in records]
    )


@app.get(
    "/api/history",
    response_model=LearningHistoryResponse,
)
async def get_learning_history(
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[LearningDataStore, Depends(get_learning_data_store)],
    session_id: UUID | None = None,
) -> LearningHistoryResponse:
    """按用户边界读取有限对话和学习行为；可进一步限定 Session。"""
    normalized_user_id = current_user.user_id
    try:
        if session_id is None:
            messages = await store.list_user_messages(
                normalized_user_id,
                limit=MAX_HISTORY_ITEMS,
            )
            records = await store.list_learning_records(
                normalized_user_id,
                limit=MAX_HISTORY_ITEMS,
            )
        else:
            normalized_session_id = str(session_id)
            messages = await store.list_messages(
                normalized_user_id,
                normalized_session_id,
                limit=MAX_HISTORY_ITEMS,
            )
            records = await store.list_learning_records(
                normalized_user_id,
                session_id=normalized_session_id,
                limit=MAX_HISTORY_ITEMS,
            )
    except Exception as error:
        raise_learning_data_http_error(error)
        raise
    return LearningHistoryResponse(
        messages=[conversation_message_response(record) for record in messages],
        learning_records=[learning_activity_response(record) for record in records],
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
    file: Annotated[
        UploadFile,
        File(description="TXT、Markdown、DOCX 或 PDF 学习资料"),
    ],
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
    "/api/materials/stage-batch",
    response_model=BatchStagedMaterialResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def stage_material_batch(
    files: Annotated[
        list[UploadFile],
        File(description="一次上传多个 TXT、Markdown、DOCX 或 PDF 学习资料"),
    ],
    response: Response,
    manager: Annotated[MaterialManager, Depends(get_material_manager)],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    operation: Annotated[Literal["add", "replace"], Form()] = "add",
) -> BatchStagedMaterialResponse:
    """逐文件零费用暂存与解析；返回每个成功和失败结果。"""
    try:
        with guard.acquire("stage"):
            result: BatchStageResult = manager.stage_upload_batch(
                tuple(
                    MaterialUpload(
                        filename=file.filename,
                        content_type=file.content_type,
                        stream=file.file,
                    )
                    for file in files
                ),
                operation=operation,
            )
    except OperationProtectionError as error:
        raise_operation_http_error(error)
        raise
    except Exception as error:
        raise_material_http_error(error)
        raise
    finally:
        for file in files:
            file.file.close()

    if not result.staged:
        batch_status = "failed"
        response.status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif result.failures:
        batch_status = "partial"
        response.status_code = status.HTTP_207_MULTI_STATUS
    else:
        batch_status = "staged"
    staged_responses = [
        StagedMaterialResponse(
            upload_id=item.upload_id,
            filename=item.filename,
            operation=item.operation,
            size_bytes=item.size_bytes,
            document_units=item.document_units,
            chunk_count=item.chunk_count,
            embedding_batch_count=item.embedding_batch_count,
        )
        for item in result.staged
    ]
    failure_responses = [
        StagedMaterialFailureResponse(
            filename=item.filename,
            reason=item.reason,
        )
        for item in result.failures
    ]
    return BatchStagedMaterialResponse(
        status=batch_status,
        staged=staged_responses,
        failures=failure_responses,
        total_files=len(files),
        staged_count=len(staged_responses),
        failed_count=len(failure_responses),
        total_chunks=sum(item.chunk_count for item in result.staged),
        embedding_batch_count=sum(
            item.embedding_batch_count for item in result.staged
        ),
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
    "/api/materials/batch/index",
    response_model=DocumentJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_staged_material_batch(
    payload: MaterialBatchIndexRequest,
    service: Annotated[DocumentJobService, Depends(get_document_job_service)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> DocumentJobResponse:
    """一次确认后创建批量文档任务，不在请求内执行解析或 Embedding。"""
    try:
        job = await service.enqueue(current_user.user_id, payload.upload_ids)
    except Exception as error:
        if isinstance(error, (DocumentJobError, ValueError)):
            raise_document_job_http_error(error)
            raise
        raise_material_http_error(error)
        raise
    return document_job_response(job)


@app.post(
    "/api/materials/{upload_id}/index",
    response_model=DocumentJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_staged_material(
    upload_id: str,
    payload: MaterialIndexRequest,
    service: Annotated[DocumentJobService, Depends(get_document_job_service)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> DocumentJobResponse:
    """取得明确费用确认后创建文档任务，并立即返回任务 ID。"""
    del payload
    try:
        job = await service.enqueue(current_user.user_id, (upload_id,))
    except Exception as error:
        if isinstance(error, (DocumentJobError, ValueError)):
            raise_document_job_http_error(error)
            raise
        raise_material_http_error(error)
        raise
    return document_job_response(job)


@app.get(
    "/api/jobs/{job_id}",
    response_model=DocumentJobResponse,
)
async def get_document_job(
    job_id: str,
    service: Annotated[DocumentJobService, Depends(get_document_job_service)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
) -> DocumentJobResponse:
    """查询文档后台任务的持久化状态和最终索引摘要。"""
    try:
        job = await service.get_job(current_user.user_id, job_id)
    except Exception as error:
        raise_document_job_http_error(error)
        raise
    return document_job_response(job)


@app.delete(
    "/api/materials/{filename}",
    response_model=MaterialDeleteResponse,
)
def delete_material(
    filename: str,
    payload: MaterialDeleteRequest,
    request: Request,
    manager: Annotated[MaterialManager, Depends(get_material_manager)],
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
) -> MaterialDeleteResponse:
    """取得明确删除确认后移除资料及对应 Chroma 记录。"""
    del payload
    try:
        with guard.acquire("delete"):
            result: MaterialDeleteResult = manager.delete_material(filename)
            invalidate_rag_service(request.app, current_user.user_id)
    except OperationProtectionError as error:
        raise_operation_http_error(error)
        raise
    except Exception as error:
        if isinstance(error, MaterialRollbackError):
            invalidate_rag_service(request.app, current_user.user_id)
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
        citations=[
            CitationResponse(
                citation_id=citation.citation_id,
                evidence_id=citation.evidence_id,
                material_id=citation.material_id,
                chunk_id=citation.chunk_id,
                source=citation.source,
                filename=citation.filename,
                page=citation.page,
                chunk_index=citation.chunk_index,
                excerpt=citation.excerpt,
                locator=citation.locator,
            )
            for citation in result.citations
        ],
    )


@app.post("/api/agent", response_model=AgentResponse)
async def run_agent(
    payload: AgentRequest,
    request: Request,
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    service_provider: Annotated[
        Callable[[], AgentService],
        Depends(get_agent_service_provider),
    ],
) -> AgentResponse:
    """在显式费用确认和单次授权边界内运行受限 LangChain Agent。"""
    try:
        with guard.acquire("agent", units=1):
            service = await asyncio.to_thread(service_provider)
            result: AgentResult = await service.ask(
                payload.message,
                allow_web_preview=payload.allow_web_preview,
            )
    except OperationProtectionError as error:
        request.state.error_category = "agent_protected"
        raise_operation_http_error(error)
        raise
    except AgentTimeoutError as error:
        request.state.error_category = "agent_timeout"
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Agent 处理超时。",
        ) from error
    except AgentExecutionError as error:
        request.state.error_category = "agent_processing"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Agent 处理失败。",
        ) from error

    return AgentResponse(
        answer=result.answer,
        sources=list(result.sources),
        tools_used=list(result.tools_used),
    )


@app.post("/api/tutor/chat", response_model=TutorChatResponse)
async def tutor_chat(
    payload: TutorChatRequest,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    store: Annotated[LearningDataStore, Depends(get_learning_data_store)],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    service_provider: Annotated[
        Callable[[], Awaitable[TutorWorkflowService]],
        Depends(get_tutor_service_provider),
    ],
) -> TutorChatResponse:
    """在显式费用确认下执行单 Tutor 的有状态学习工作流。"""
    try:
        await store.get_session(
            current_user.user_id,
            str(payload.session_id),
        )
        with guard.acquire("agent", units=1):
            service = await service_provider()
            result: TutorResult = await service.chat(
                current_user.user_id,
                str(payload.session_id),
                payload.message,
            )
    except OperationProtectionError as error:
        request.state.error_category = "tutor_protected"
        raise_operation_http_error(error)
        raise
    except TutorTimeoutError as error:
        request.state.error_category = "tutor_timeout"
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Tutor 处理超时。",
        ) from error
    except LearningDataNotFoundError as error:
        request.state.error_category = "tutor_not_found"
        raise_learning_data_http_error(error)
        raise
    except TutorExecutionError as error:
        request.state.error_category = "tutor_processing"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Tutor 处理失败。",
        ) from error

    return TutorChatResponse(
        session_id=result.session_id,
        intent=result.intent,
        topic=result.topic,
        answer=result.answer,
        sources=list(result.sources),
        citations=[CitationResponse(**citation) for citation in result.citations],
        evidence=[TutorEvidenceResponse(**item) for item in result.evidence],
        learning_action=result.learning_action,
        quiz=QuizDraft.model_validate(result.quiz) if result.quiz else None,
        summary=(
            LearningSummaryDraft.model_validate(result.summary)
            if result.summary
            else None
        ),
        tools_used=list(result.tools_used),
        route_trace=list(result.route_trace),
    )


@app.post(
    "/api/study-workflows",
    response_model=StudyWorkflowResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_study_workflow(
    payload: StudyWorkflowStartRequest,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    workflow_service: Annotated[
        StudyWorkflowService,
        Depends(get_study_workflow_service),
    ],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    agent_service_provider: Annotated[
        Callable[[], AgentService],
        Depends(get_agent_service_provider),
    ],
) -> StudyWorkflowResponse:
    """确认费用后创建学习计划，并在写入进度前暂停等待人工确认。"""
    try:
        with guard.acquire("agent", units=1):
            agent_service = await asyncio.to_thread(agent_service_provider)
            result = await workflow_service.start(
                current_user.user_id,
                str(uuid4()),
                payload.goal,
                agent_service=agent_service,
            )
    except OperationProtectionError as error:
        request.state.error_category = "workflow_protected"
        raise_operation_http_error(error)
        raise
    except Exception as error:
        request.state.error_category = "workflow"
        raise_study_workflow_http_error(error)
        raise
    return study_workflow_response(result)


@app.post(
    "/api/study-workflows/{workflow_id}/confirm",
    response_model=StudyWorkflowResponse,
)
async def confirm_study_workflow(
    workflow_id: UUID,
    payload: StudyWorkflowConfirmRequest,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    workflow_service: Annotated[
        StudyWorkflowService,
        Depends(get_study_workflow_service),
    ],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
) -> StudyWorkflowResponse:
    """用同一 thread_id 恢复 interrupt，并按批准或拒绝分支继续。"""
    try:
        with guard.acquire("workflow"):
            result = await workflow_service.confirm(
                current_user.user_id,
                str(workflow_id),
                payload.decision,
            )
    except OperationProtectionError as error:
        request.state.error_category = "workflow_protected"
        raise_operation_http_error(error)
        raise
    except Exception as error:
        request.state.error_category = "workflow"
        raise_study_workflow_http_error(error)
        raise
    return study_workflow_response(result)


@app.post(
    "/api/study-workflows/{workflow_id}/progress",
    response_model=StudyWorkflowResponse,
)
async def update_study_workflow_progress(
    workflow_id: UUID,
    payload: StudyWorkflowProgressRequest,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    workflow_service: Annotated[
        StudyWorkflowService,
        Depends(get_study_workflow_service),
    ],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
) -> StudyWorkflowResponse:
    """零模型调用记录当前任务进度，并在全部完成后生成确定性复盘。"""
    try:
        with guard.acquire("workflow"):
            result = await workflow_service.record_progress(
                current_user.user_id,
                str(workflow_id),
                payload.note,
                complete_current_task=payload.complete_current_task,
            )
    except OperationProtectionError as error:
        request.state.error_category = "workflow_protected"
        raise_operation_http_error(error)
        raise
    except Exception as error:
        request.state.error_category = "workflow"
        raise_study_workflow_http_error(error)
        raise
    return study_workflow_response(result)


@app.post(
    "/api/study-workflows/{workflow_id}/retry",
    response_model=StudyWorkflowResponse,
)
async def retry_study_workflow(
    workflow_id: UUID,
    payload: StudyWorkflowRetryRequest,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    workflow_service: Annotated[
        StudyWorkflowService,
        Depends(get_study_workflow_service),
    ],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
    agent_service_provider: Annotated[
        Callable[[], AgentService],
        Depends(get_agent_service_provider),
    ],
) -> StudyWorkflowResponse:
    """再次确认费用后执行唯一一次手动 Agent 重试。"""
    try:
        with guard.acquire("agent", units=1):
            await workflow_service.assert_retryable(
                current_user.user_id, str(workflow_id)
            )
            agent_service = await asyncio.to_thread(agent_service_provider)
            result = await workflow_service.retry(
                current_user.user_id,
                str(workflow_id),
                agent_service=agent_service,
            )
    except OperationProtectionError as error:
        request.state.error_category = "workflow_protected"
        raise_operation_http_error(error)
        raise
    except Exception as error:
        request.state.error_category = "workflow"
        raise_study_workflow_http_error(error)
        raise
    return study_workflow_response(result)


@app.get(
    "/api/study-workflows/{workflow_id}",
    response_model=StudyWorkflowResponse,
)
async def get_study_workflow(
    workflow_id: UUID,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    workflow_service: Annotated[
        StudyWorkflowService,
        Depends(get_study_workflow_service),
    ],
) -> StudyWorkflowResponse:
    """零模型调用读取指定 thread_id 的最新持久化状态。"""
    try:
        result = await workflow_service.get(current_user.user_id, str(workflow_id))
    except Exception as error:
        request.state.error_category = "workflow"
        raise_study_workflow_http_error(error)
        raise
    return study_workflow_response(result)


@app.delete(
    "/api/study-workflows/{workflow_id}",
    response_model=StudyWorkflowDeleteResponse,
)
async def delete_study_workflow(
    workflow_id: UUID,
    payload: StudyWorkflowDeleteRequest,
    request: Request,
    current_user: Annotated[UserRecord, Depends(get_current_user)],
    workflow_service: Annotated[
        StudyWorkflowService,
        Depends(get_study_workflow_service),
    ],
    guard: Annotated[OperationGuard, Depends(get_operation_guard)],
) -> StudyWorkflowDeleteResponse:
    """经显式确认删除一个工作流的全部本地检查点。"""
    del payload
    try:
        with guard.acquire("workflow"):
            await workflow_service.delete(current_user.user_id, str(workflow_id))
    except OperationProtectionError as error:
        request.state.error_category = "workflow_protected"
        raise_operation_http_error(error)
        raise
    except Exception as error:
        request.state.error_category = "workflow"
        raise_study_workflow_http_error(error)
        raise
    return StudyWorkflowDeleteResponse(workflow_id=str(workflow_id))


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
