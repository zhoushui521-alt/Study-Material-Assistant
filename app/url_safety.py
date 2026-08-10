"""网页资料导入使用的公网 HTTP(S) URL 安全校验。"""

import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from urllib.parse import SplitResult, urlsplit, urlunsplit


MAX_URL_LENGTH = 2048
ALLOWED_PORTS = {"http": 80, "https": 443}
Resolver = Callable[[str, int], Iterable[str]]


class UnsafeURLError(ValueError):
    """URL 可能访问本机、内网、保留网络或不受支持的协议。"""


@dataclass(frozen=True)
class ValidatedPublicURL:
    canonical_url: str
    hostname: str
    port: int
    resolved_addresses: tuple[str, ...]


def resolve_host_addresses(hostname: str, port: int) -> tuple[str, ...]:
    """解析目标的全部 A/AAAA 地址；调用方仍须在实际连接时防止 DNS 重绑定。"""
    try:
        records = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise UnsafeURLError("URL 域名无法解析。") from error
    return tuple(sorted({str(record[4][0]) for record in records}))


def _parse_address(value: str) -> IPv4Address | IPv6Address:
    try:
        return ip_address(value)
    except ValueError as error:
        raise UnsafeURLError("URL 域名解析结果包含无效 IP 地址。") from error


def _require_global_address(value: str) -> str:
    address = _parse_address(value)
    if not address.is_global:
        raise UnsafeURLError("URL 只能解析到公网 IP 地址。")
    return address.compressed


def _canonical_netloc(hostname: str, port: int, scheme: str) -> str:
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    return rendered_host if port == ALLOWED_PORTS[scheme] else f"{rendered_host}:{port}"


def validate_public_http_url(
    raw_url: str,
    *,
    resolver: Resolver = resolve_host_addresses,
) -> ValidatedPublicURL:
    """校验一个 URL；抓取器会对初始地址和每次重定向目标调用本函数。"""
    if not isinstance(raw_url, str) or not raw_url or raw_url != raw_url.strip():
        raise UnsafeURLError("URL 不能为空或包含首尾空白。")
    if len(raw_url) > MAX_URL_LENGTH:
        raise UnsafeURLError(f"URL 不能超过 {MAX_URL_LENGTH} 个字符。")
    if "\\" in raw_url or any(
        ord(character) < 32 or ord(character) == 127 for character in raw_url
    ):
        raise UnsafeURLError("URL 包含不允许的控制字符或反斜杠。")

    try:
        parsed: SplitResult = urlsplit(raw_url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port or ALLOWED_PORTS.get(scheme)
    except ValueError as error:
        raise UnsafeURLError("URL 格式无效。") from error

    if scheme not in ALLOWED_PORTS:
        raise UnsafeURLError("URL 只允许 http 或 https 协议。")
    if not hostname or port is None:
        raise UnsafeURLError("URL 必须包含有效域名。")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("URL 不能包含用户名或密码。")
    if parsed.fragment:
        raise UnsafeURLError("URL 不能包含片段标识。")
    if port != ALLOWED_PORTS[scheme]:
        raise UnsafeURLError("URL 只允许协议对应的默认端口。")
    if "%" in hostname:
        raise UnsafeURLError("URL 不支持带作用域标识的主机地址。")

    try:
        literal_address = ip_address(hostname)
    except ValueError:
        try:
            canonical_hostname = (
                hostname.rstrip(".").encode("idna").decode("ascii").lower()
            )
        except UnicodeError as error:
            raise UnsafeURLError("URL 域名格式无效。") from error
        if not canonical_hostname or len(canonical_hostname) > 253:
            raise UnsafeURLError("URL 域名格式无效。")
        resolved = tuple(resolver(canonical_hostname, port))
        if not resolved:
            raise UnsafeURLError("URL 域名没有可用的 A 或 AAAA 地址。")
        safe_addresses = tuple(sorted({_require_global_address(value) for value in resolved}))
    else:
        canonical_hostname = literal_address.compressed
        safe_addresses = (_require_global_address(canonical_hostname),)

    canonical_url = urlunsplit(
        (
            scheme,
            _canonical_netloc(canonical_hostname, port, scheme),
            parsed.path or "/",
            parsed.query,
            "",
        )
    )
    return ValidatedPublicURL(
        canonical_url=canonical_url,
        hostname=canonical_hostname,
        port=port,
        resolved_addresses=safe_addresses,
    )
