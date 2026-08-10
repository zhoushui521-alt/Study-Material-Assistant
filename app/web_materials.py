"""安全获取单个公开网页，并用 Crawl4AI 生成可确认入库的 Markdown。"""

import asyncio
import hashlib
import http.client
import os
import re
import socket
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from time import monotonic
from urllib.parse import urljoin, urlsplit

from app.chunk_documents import (
    PROJECT_ROOT,
    WebSourceMetadata,
    encode_web_source_marker,
)
from app.material_ingestion import MaterialManager, StagedMaterial, UploadOperation
from app.url_safety import (
    Resolver,
    ValidatedPublicURL,
    resolve_host_addresses,
    validate_public_http_url,
)


MAX_WEB_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_WEB_MARKDOWN_CHARACTERS = 30_000
MAX_WEB_REDIRECTS = 5
WEB_FETCH_TIMEOUT_SECONDS = 20.0
WEB_USER_AGENT = "StudyMaterialAssistant/1.0"
CRAWL4AI_RUNTIME_DIR = PROJECT_ROOT / "data" / "crawl4ai_runtime"
ALLOWED_HTML_MEDIA_TYPES = {"text/html", "application/xhtml+xml"}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
TITLE_MAX_CHARACTERS = 200


class WebMaterialError(RuntimeError):
    """网页资料预览无法安全完成。"""


class WebMaterialValidationError(WebMaterialError):
    """网页响应或生成内容不符合第一版导入范围。"""


class WebMaterialTooLargeError(WebMaterialValidationError):
    """网页响应或 Markdown 超过安全上限。"""


class WebMaterialFetchError(WebMaterialError):
    """受控公网 HTTP 请求失败。"""


class WebMaterialConversionError(WebMaterialError):
    """Crawl4AI 未能生成可用 Markdown。"""


@dataclass(frozen=True)
class HTTPFetchResult:
    status_code: int
    headers: Mapping[str, str]
    html: str | None = None


@dataclass(frozen=True)
class FetchedHTML:
    requested_url: str
    canonical_url: str
    html: str
    redirect_count: int


@dataclass(frozen=True)
class WebMaterialPreview:
    upload_id: str
    filename: str
    operation: UploadOperation
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


FetchOnce = Callable[[ValidatedPublicURL, float, int], HTTPFetchResult]
MarkdownConverter = Callable[[str, str], str]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """保留原始 Host，但直接连接已经校验的公网 IP。"""

    def __init__(
        self,
        hostname: str,
        port: int,
        resolved_address: str,
        timeout: float,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout)
        self._resolved_address = resolved_address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._resolved_address, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """直连已校验 IP，同时使用原始域名执行 TLS SNI 与证书校验。"""

    def __init__(
        self,
        hostname: str,
        port: int,
        resolved_address: str,
        timeout: float,
    ) -> None:
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._resolved_address = resolved_address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._resolved_address, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except Exception:
            raw_socket.close()
            raise


def _response_charset(content_type: str) -> str:
    match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type, re.I)
    return match.group(1) if match else "utf-8"


def _request_one_address(
    target: ValidatedPublicURL,
    address: str,
    timeout: float,
    max_response_bytes: int,
) -> HTTPFetchResult:
    parsed = urlsplit(target.canonical_url)
    connection_class = (
        _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    )
    connection = connection_class(
        target.hostname,
        target.port,
        address,
        timeout,
    )
    request_target = parsed.path or "/"
    if parsed.query:
        request_target = f"{request_target}?{parsed.query}"
    try:
        connection.request(
            "GET",
            request_target,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "identity",
                "Connection": "close",
                "Host": parsed.netloc,
                "User-Agent": WEB_USER_AGENT,
            },
        )
        response = connection.getresponse()
        headers = {key.lower(): value.strip() for key, value in response.getheaders()}
        status_code = response.status
        if status_code in REDIRECT_STATUS_CODES:
            return HTTPFetchResult(status_code=status_code, headers=headers)
        if not 200 <= status_code < 300:
            raise WebMaterialFetchError("公开网页返回了不支持的 HTTP 状态。")

        content_type = headers.get("content-type", "")
        media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
        if media_type not in ALLOWED_HTML_MEDIA_TYPES:
            raise WebMaterialValidationError("网页地址必须返回 HTML 内容。")
        content_encoding = headers.get("content-encoding", "identity").lower()
        if content_encoding not in {"", "identity"}:
            raise WebMaterialValidationError("网页响应使用了不支持的内容编码。")
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise WebMaterialValidationError("网页响应长度无效。") from error
            if declared_length < 0:
                raise WebMaterialValidationError("网页响应长度无效。")
            if declared_length > max_response_bytes:
                raise WebMaterialTooLargeError("网页响应超过 2 MiB 安全限制。")

        body = response.read(max_response_bytes + 1)
        if len(body) > max_response_bytes:
            raise WebMaterialTooLargeError("网页响应超过 2 MiB 安全限制。")
        charset = _response_charset(content_type)
        try:
            html = body.decode(charset, errors="replace")
        except LookupError as error:
            raise WebMaterialValidationError("网页响应字符编码无效。") from error
        if not html.strip():
            raise WebMaterialValidationError("网页没有可预览的 HTML 内容。")
        return HTTPFetchResult(
            status_code=status_code,
            headers=headers,
            html=html,
        )
    finally:
        connection.close()


def fetch_once_from_validated_target(
    target: ValidatedPublicURL,
    timeout: float,
    max_response_bytes: int,
    *,
    clock: Callable[[], float] = monotonic,
) -> HTTPFetchResult:
    """只连接本轮 DNS 校验得到的公网地址，避免浏览器再次独立解析。"""
    last_error: Exception | None = None
    deadline = clock() + timeout
    for address in target.resolved_addresses:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        try:
            return _request_one_address(
                target,
                address,
                remaining,
                max_response_bytes,
            )
        except (OSError, http.client.HTTPException) as error:
            last_error = error
    raise WebMaterialFetchError("无法连接公开网页。") from last_error


def safe_fetch_public_html(
    raw_url: str,
    *,
    resolver: Resolver = resolve_host_addresses,
    fetch_once: FetchOnce = fetch_once_from_validated_target,
    allow_proxy_fake_ip: bool = False,
    max_redirects: int = MAX_WEB_REDIRECTS,
    max_response_bytes: int = MAX_WEB_RESPONSE_BYTES,
    timeout_seconds: float = WEB_FETCH_TIMEOUT_SECONDS,
    clock: Callable[[], float] = monotonic,
) -> FetchedHTML:
    """受控获取单页 HTML；每个重定向目标都重新解析并校验。"""
    if max_redirects < 0 or max_response_bytes <= 0 or timeout_seconds <= 0:
        raise ValueError("网页抓取限制必须有效。")
    current = validate_public_http_url(
        raw_url,
        resolver=resolver,
        allow_proxy_fake_ip=allow_proxy_fake_ip,
    )
    requested_url = current.canonical_url
    seen_urls = {current.canonical_url}
    deadline = clock() + timeout_seconds

    for redirect_count in range(max_redirects + 1):
        remaining = deadline - clock()
        if remaining <= 0:
            raise WebMaterialFetchError("公开网页抓取超时。")
        result = fetch_once(current, remaining, max_response_bytes)
        if result.status_code in REDIRECT_STATUS_CODES:
            if redirect_count >= max_redirects:
                raise WebMaterialValidationError("网页重定向次数超过安全限制。")
            location = result.headers.get("location")
            if not location:
                raise WebMaterialValidationError("网页重定向缺少目标地址。")
            redirected_url = urljoin(current.canonical_url, location)
            current = validate_public_http_url(
                redirected_url,
                resolver=resolver,
                allow_proxy_fake_ip=allow_proxy_fake_ip,
            )
            if current.canonical_url in seen_urls:
                raise WebMaterialValidationError("网页重定向形成循环。")
            seen_urls.add(current.canonical_url)
            continue
        if not 200 <= result.status_code < 300:
            raise WebMaterialFetchError("公开网页返回了不支持的 HTTP 状态。")
        if result.html is None:
            raise WebMaterialFetchError("公开网页没有返回可用内容。")
        return FetchedHTML(
            requested_url=requested_url,
            canonical_url=current.canonical_url,
            html=result.html,
            redirect_count=redirect_count,
        )
    raise WebMaterialValidationError("网页重定向次数超过安全限制。")


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._inside_title = False
        self._parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag.casefold() == "title":
            self._inside_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self._inside_title = False

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self._parts.append(data)

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._parts).split())[:TITLE_MAX_CHARACTERS]


def extract_page_title(html: str, canonical_url: str) -> str:
    parser = _TitleParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.title or urlsplit(canonical_url).hostname or "网页资料"


def crawl4ai_markdown(html: str, base_url: str) -> str:
    """使用 Crawl4AI 的清理与 Markdown 生成组件处理已安全获取的 HTML。"""
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(CRAWL4AI_RUNTIME_DIR)
    try:
        from crawl4ai import DefaultMarkdownGenerator, LXMLWebScrapingStrategy
    except Exception as error:
        raise WebMaterialConversionError("Crawl4AI 网页预览组件不可用。") from error

    try:
        scraped = LXMLWebScrapingStrategy().scrap(
            base_url,
            html,
            excluded_tags=[
                "script",
                "style",
                "noscript",
                "form",
                "iframe",
                "object",
                "embed",
            ],
            remove_forms=True,
            exclude_all_images=True,
            process_iframes=False,
        )
        generated = DefaultMarkdownGenerator(
            options={"ignore_images": True},
        ).generate_markdown(
            scraped.cleaned_html,
            base_url=base_url,
            citations=False,
        )
    except Exception as error:
        raise WebMaterialConversionError("Crawl4AI 无法生成网页预览。") from error
    markdown = generated.raw_markdown.replace("\x00", "").strip()
    if not markdown:
        raise WebMaterialConversionError("网页没有可生成 Markdown 的正文。")
    return markdown


def web_material_filename(canonical_url: str) -> str:
    hostname = urlsplit(canonical_url).hostname or "web"
    hostname_slug = re.sub(r"[^a-z0-9]+", "-", hostname.casefold()).strip("-")
    hostname_slug = hostname_slug[:50] or "web"
    url_digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:12]
    return f"web-{hostname_slug}-{url_digest}.md"


class WebMaterialService:
    """把公开网页转换为现有 MaterialManager 可提交的暂存 Markdown。"""

    def __init__(
        self,
        material_manager: MaterialManager,
        *,
        fetch_html: Callable[[str], FetchedHTML] | None = None,
        allow_proxy_fake_ip: bool = False,
        convert_html: MarkdownConverter = crawl4ai_markdown,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._material_manager = material_manager
        self._fetch_html = (
            fetch_html
            if fetch_html is not None
            else lambda raw_url: safe_fetch_public_html(
                raw_url,
                allow_proxy_fake_ip=allow_proxy_fake_ip,
            )
        )
        self._convert_html = convert_html
        self._now = now

    async def preview(
        self,
        raw_url: str,
        *,
        operation: str = "add",
    ) -> WebMaterialPreview:
        """抓取并暂存 Markdown；此步骤不打开 Chroma 或调用 Embedding。"""
        if operation not in {"add", "replace"}:
            raise WebMaterialValidationError("网页资料操作只能是 add 或 replace。")
        fetched = await asyncio.to_thread(self._fetch_html, raw_url)
        markdown = await asyncio.to_thread(
            self._convert_html,
            fetched.html,
            fetched.canonical_url,
        )
        if len(markdown) > MAX_WEB_MARKDOWN_CHARACTERS:
            raise WebMaterialTooLargeError(
                "网页 Markdown 超过 30000 个字符的预览与费用保护上限。"
            )

        title = extract_page_title(fetched.html, fetched.canonical_url)
        crawled_at = self._now().astimezone(UTC).isoformat().replace("+00:00", "Z")
        document_body = (
            f"# {title}\n\n"
            f"原始网页：{fetched.canonical_url}\n\n"
            f"{markdown}\n"
        )
        content_sha256 = hashlib.sha256(document_body.encode("utf-8")).hexdigest()
        metadata = WebSourceMetadata(
            canonical_url=fetched.canonical_url,
            title=title,
            crawled_at=crawled_at,
            content_sha256=content_sha256,
        )
        document_text = (
            f"{encode_web_source_marker(metadata)}\n"
            f"{document_body}"
        )
        staged: StagedMaterial = self._material_manager.stage_upload(
            filename=web_material_filename(fetched.canonical_url),
            content_type="text/markdown",
            stream=BytesIO(document_text.encode("utf-8")),
            operation=operation,
        )
        return WebMaterialPreview(
            upload_id=staged.upload_id,
            filename=staged.filename,
            operation=staged.operation,
            requested_url=fetched.requested_url,
            canonical_url=fetched.canonical_url,
            title=title,
            crawled_at=crawled_at,
            content_sha256=content_sha256,
            markdown=markdown,
            redirect_count=fetched.redirect_count,
            size_bytes=staged.size_bytes,
            document_units=staged.document_units,
            chunk_count=staged.chunk_count,
            embedding_batch_count=staged.embedding_batch_count,
        )
