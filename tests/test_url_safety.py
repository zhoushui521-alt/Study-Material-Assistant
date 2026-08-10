import unittest

from app.url_safety import UnsafeURLError, validate_public_http_url


class URLSafetyTests(unittest.TestCase):
    def test_accepts_http_url_only_when_all_resolved_addresses_are_global(self) -> None:
        calls = []

        def resolver(hostname: str, port: int) -> tuple[str, ...]:
            calls.append((hostname, port))
            return ("8.8.8.8", "2001:4860:4860::8888")

        result = validate_public_http_url(
            "https://Example.COM/course?q=rag",
            resolver=resolver,
        )

        self.assertEqual(result.canonical_url, "https://example.com/course?q=rag")
        self.assertEqual(result.hostname, "example.com")
        self.assertEqual(result.port, 443)
        self.assertEqual(calls, [("example.com", 443)])

    def test_rejects_non_http_schemes(self) -> None:
        for url in ("file:///etc/passwd", "data:text/plain,hello", "ftp://8.8.8.8/"):
            with self.subTest(url=url):
                with self.assertRaisesRegex(UnsafeURLError, "http 或 https"):
                    validate_public_http_url(url, resolver=lambda host, port: ())

    def test_rejects_credentials_fragment_non_default_port_and_backslash(self) -> None:
        urls = (
            "https://user:password@example.com/",
            "https://example.com/#section",
            "https://example.com:8443/",
            "https://example.com\\@127.0.0.1/",
        )
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(UnsafeURLError):
                    validate_public_http_url(
                        url,
                        resolver=lambda host, port: ("8.8.8.8",),
                    )

    def test_rejects_literal_non_global_addresses(self) -> None:
        urls = (
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[fc00::1]/",
            "http://[fe80::1]/",
        )
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaisesRegex(UnsafeURLError, "公网 IP"):
                    validate_public_http_url(url)

    def test_rejects_dns_result_when_any_address_is_not_global(self) -> None:
        with self.assertRaisesRegex(UnsafeURLError, "公网 IP"):
            validate_public_http_url(
                "https://example.com/",
                resolver=lambda host, port: ("8.8.8.8", "127.0.0.1"),
            )

    def test_rejects_domain_without_addresses(self) -> None:
        with self.assertRaisesRegex(UnsafeURLError, "没有可用"):
            validate_public_http_url(
                "https://example.com/",
                resolver=lambda host, port: (),
            )


if __name__ == "__main__":
    unittest.main()
