"""
Tests for media download retry logic added in PR #2982.

Covers:
- gateway/platforms/base.py:       cache_image_from_url
- gateway/platforms/slack.py:      SlackAdapter._download_slack_file
                                    SlackAdapter._download_slack_file_bytes
- gateway/platforms/mattermost.py: MattermostAdapter._send_url_as_file

All async tests use asyncio.run() directly — pytest-asyncio is not installed
in this environment.
"""

import asyncio
import socket
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

# ---------------------------------------------------------------------------
# Helpers for building httpx exceptions
# ---------------------------------------------------------------------------

def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://example.com/img.jpg")
    response = httpx.Response(status_code=status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


def _make_timeout_error() -> httpx.TimeoutException:
    return httpx.TimeoutException("timed out")


def _make_stream_response(content: bytes = b"\xff\xd8\xff fake media"):
    """Build a mock httpx response suitable for ``client.stream()`` usage.

    Exposes ``raise_for_status``, an empty ``headers`` mapping (no
    Content-Length), and an ``aiter_bytes`` async iterator yielding the body
    in one chunk — matching how ``_read_httpx_body_with_limit`` consumes it.
    """
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {}

    async def _aiter():
        yield content

    resp.aiter_bytes = lambda: _aiter()
    return resp


def _make_stream_client(*, responses=None, side_effect=None):
    """Build a mock httpx client whose ``.stream()`` is an async CM.

    ``responses`` is a list of response objects (or exceptions) returned on
    successive ``.stream()`` calls; ``side_effect`` is a single exception
    raised on every call. The returned client also supports being used as an
    ``async with`` context manager (``httpx.AsyncClient(...)``).
    """
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    call_state = {"i": 0}

    def _stream(method, url, **kwargs):
        idx = call_state["i"]
        call_state["i"] += 1
        if side_effect is not None:
            raise side_effect
        item = responses[idx]
        if isinstance(item, Exception):
            raise item
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=item)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    mock_client.stream = MagicMock(side_effect=_stream)
    mock_client._call_state = call_state
    return mock_client


# ---------------------------------------------------------------------------
# cache_image_from_bytes (base.py)
# ---------------------------------------------------------------------------


class TestCacheImageFromBytes:
    """Tests for gateway.platforms.base.cache_image_from_bytes"""

    def test_caches_valid_jpeg(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        from gateway.platforms.base import cache_image_from_bytes
        path = cache_image_from_bytes(b"\xff\xd8\xff fake jpeg data", ".jpg")
        assert path.endswith(".jpg")


    def test_rejects_html_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        from gateway.platforms.base import cache_image_from_bytes
        with pytest.raises(ValueError, match="non-image data"):
            cache_image_from_bytes(b"<!DOCTYPE html><html><title>Slack</title></html>", ".png")


# ---------------------------------------------------------------------------
# cache_image_from_url (base.py)
# ---------------------------------------------------------------------------

@patch("tools.url_safety.is_safe_url", return_value=True)
class TestCacheImageFromUrl:
    """Tests for gateway.platforms.base.cache_image_from_url"""

    def test_success_on_first_attempt(self, _mock_safe, tmp_path, monkeypatch):
        """A clean 200 response caches the image and returns a path."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        mock_client = _make_stream_client(
            responses=[_make_stream_response(b"\xff\xd8\xff fake jpeg")]
        )

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                from gateway.platforms.base import cache_image_from_url
                return await cache_image_from_url(
                    "http://example.com/img.jpg", ext=".jpg"
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        mock_client.stream.assert_called_once()

    def test_retries_on_timeout_then_succeeds(self, _mock_safe, tmp_path, monkeypatch):
        """A timeout on the first attempt is retried; second attempt succeeds."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        mock_client = _make_stream_client(
            responses=[_make_timeout_error(), _make_stream_response(b"\xff\xd8\xff image data")]
        )
        mock_sleep = AsyncMock()

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                from gateway.platforms.base import cache_image_from_url
                return await cache_image_from_url(
                    "http://example.com/img.jpg", ext=".jpg", retries=2
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        assert mock_client.stream.call_count == 2
        mock_sleep.assert_called_once()


class TestCacheImageFromUrlConnectGuard:
    def test_blocks_private_dns_answer_at_connect_time(self, tmp_path, monkeypatch):
        """A hostname that rebinds after preflight must not reach TCP connect."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        for proxy_var in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            monkeypatch.delenv(proxy_var, raising=False)

        answers = [
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))],
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))],
        ]

        def fake_getaddrinfo(host, port, *args, **kwargs):
            assert host == "rebind.test"
            return answers.pop(0)

        from httpcore._backends.auto import AutoBackend

        async def fail_connect_tcp(
            self,
            host,
            port,
            timeout=None,
            local_address=None,
            socket_options=None,
        ):
            raise AssertionError(f"TCP connect attempted for {host}:{port}")

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        monkeypatch.setattr(AutoBackend, "connect_tcp", fail_connect_tcp)

        async def run():
            from gateway.platforms.base import cache_image_from_url
            await cache_image_from_url("http://rebind.test/image.jpg", ext=".jpg", retries=0)

        with pytest.raises(ValueError, match="during connect"):
            asyncio.run(run())

        assert answers == []


# ---------------------------------------------------------------------------
# cache_audio_from_url (base.py)
# ---------------------------------------------------------------------------

@patch("tools.url_safety.is_safe_url", return_value=True)
class TestCacheAudioFromUrl:
    """Tests for gateway.platforms.base.cache_audio_from_url"""

    def test_success_on_first_attempt(self, _mock_safe, tmp_path, monkeypatch):
        """A clean 200 response caches the audio and returns a path."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        mock_client = _make_stream_client(
            responses=[_make_stream_response(b"\x00\x01 fake audio")]
        )

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                from gateway.platforms.base import cache_audio_from_url
                return await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg"
                )

        path = asyncio.run(run())
        assert path.endswith(".ogg")
        mock_client.stream.assert_called_once()

    def test_retries_on_timeout_then_succeeds(self, _mock_safe, tmp_path, monkeypatch):
        """A timeout on the first attempt is retried; second attempt succeeds."""
        monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio")

        mock_client = _make_stream_client(
            responses=[_make_timeout_error(), _make_stream_response(b"audio data")]
        )
        mock_sleep = AsyncMock()

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client), \
                 patch("asyncio.sleep", mock_sleep):
                from gateway.platforms.base import cache_audio_from_url
                return await cache_audio_from_url(
                    "http://example.com/voice.ogg", ext=".ogg", retries=2
                )

        path = asyncio.run(run())
        assert path.endswith(".ogg")
        assert mock_client.stream.call_count == 2
        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# SSRF redirect guard tests (base.py)
# ---------------------------------------------------------------------------


class TestSSRFRedirectGuard:
    """cache_image_from_url / cache_audio_from_url must reject redirects
    that land on private/internal hosts (e.g. cloud metadata endpoint)."""

    def _make_redirect_response(self, target_url: str):
        """Build a mock httpx response that looks like a redirect."""
        resp = MagicMock()
        resp.is_redirect = True
        resp.next_request = MagicMock(url=target_url)
        return resp

    def _make_client_capturing_hooks(self):
        """Return (mock_client, captured_kwargs dict) where captured_kwargs
        will contain the kwargs passed to httpx.AsyncClient()."""
        captured = {}
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        def factory(*args, **kwargs):
            captured.update(kwargs)
            return mock_client

        return mock_client, captured, factory

    def test_image_blocks_private_redirect(self, tmp_path, monkeypatch):
        """cache_image_from_url rejects a redirect to a private IP."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")

        redirect_resp = self._make_redirect_response(
            "http://169.254.169.254/latest/meta-data"
        )
        mock_client, captured, factory = self._make_client_capturing_hooks()

        def fake_stream(method, _url, **kwargs):
            async def _aenter(*a):
                # Simulate httpx invoking the response event hooks on the stream.
                for hook in captured["event_hooks"]["response"]:
                    await hook(redirect_resp)
                return redirect_resp
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(side_effect=_aenter)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        mock_client.stream = MagicMock(side_effect=fake_stream)

        def fake_safe(url):
            return url == "https://public.example.com/image.png"

        async def run():
            with patch("tools.url_safety.is_safe_url", side_effect=fake_safe), \
                 patch("httpx.AsyncClient", side_effect=factory):
                from gateway.platforms.base import cache_image_from_url
                await cache_image_from_url(
                    "https://public.example.com/image.png", ext=".png"
                )

        with pytest.raises(ValueError, match="Blocked redirect"):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Slack mock setup (mirrors existing test_slack.py approach)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


def _make_slack_adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._running = True
    return adapter


# ---------------------------------------------------------------------------
# SlackAdapter diagnostics helpers
# ---------------------------------------------------------------------------

class TestSlackAttachmentDiagnostics:

    def test_download_failure_403_returns_permission_notice(self):
        adapter = _make_slack_adapter()
        exc = _make_http_status_error(403)
        detail = adapter._describe_slack_download_failure(exc, file_obj={"name": "report.pdf"})
        assert "403" in detail
        assert "permission or scope" in detail


# ---------------------------------------------------------------------------
# SlackAdapter._download_slack_file
# ---------------------------------------------------------------------------

@pytest.fixture
def slack_public_dns(monkeypatch):
    import socket
    import tools.url_safety as url_safety

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "false")
    monkeypatch.setattr(url_safety, "_allow_private_resolved", False)
    addresses = {
        "files.slack.com": "93.184.216.34",
        "public.example.test": "93.184.216.34",
        "second.example.test": "93.184.216.35",
        "files.slack.com.evil.test": "93.184.216.36",
        "internal.example.test": "10.0.0.1",
        "10.0.0.1": "10.0.0.1",
    }
    answers = []

    def resolve(host, port=None):
        if host not in addresses:
            raise socket.gaierror(host)
        answer = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                   (addresses[host], port or 443))]
        answers.append((host, answer))
        return answer

    monkeypatch.setattr(url_safety, "_getaddrinfo", resolve)
    return addresses, answers


@pytest.fixture
def slack_http(monkeypatch, slack_public_dns):
    import httpx

    requests = []
    client_class = httpx.AsyncClient

    def install(handler):
        class Transport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request):
                requests.append(request)
                return handler(request)

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_class(
            transport=Transport(), **kw
        ))
        return requests

    return install


@pytest.mark.usefixtures("slack_public_dns")
class TestSlackDownloadSlackFile:
    """Tests for SlackAdapter._download_slack_file"""

    def test_success_on_first_attempt(self, tmp_path, monkeypatch):
        """Successful download on first try returns a cached file path."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        adapter = _make_slack_adapter()

        fake_response = MagicMock()
        fake_response.content = b"\x89PNG\r\n\x1a\n fake png"
        fake_response.raise_for_status = MagicMock()
        fake_response.headers = {"content-type": "image/png"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                return await adapter._download_slack_file(
                    "https://files.slack.com/img.jpg", ext=".jpg"
                )

        path = asyncio.run(run())
        assert path.endswith(".jpg")
        from pathlib import Path
        assert Path(path).read_bytes() == fake_response.content
        mock_client.get.assert_called_once()

    def test_rejects_html_response(self, tmp_path, monkeypatch):
        """An HTML sign-in page from Slack is rejected, not cached as image."""
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "img")
        adapter = _make_slack_adapter()

        fake_response = MagicMock()
        fake_response.content = b"<!DOCTYPE html><html><title>Slack</title></html>"
        fake_response.raise_for_status = MagicMock()
        fake_response.headers = {"content-type": "text/html; charset=utf-8"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                await adapter._download_slack_file(
                    "https://files.slack.com/img.jpg", ext=".jpg"
                )

        with pytest.raises(ValueError, match="HTML instead of media"):
            asyncio.run(run())

        # Verify nothing was cached
        img_dir = tmp_path / "img"
        if img_dir.exists():
            assert list(img_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# SlackAdapter._download_slack_file_bytes
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("slack_public_dns")
class TestSlackDownloadSlackFileBytes:
    """Tests for SlackAdapter._download_slack_file_bytes"""

    def test_success_returns_bytes(self):
        """Successful download returns raw bytes."""
        adapter = _make_slack_adapter()

        fake_response = MagicMock()
        fake_response.content = b"raw bytes here"
        fake_response.raise_for_status = MagicMock()
        fake_response.headers = {"content-type": "application/pdf"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def run():
            with patch("httpx.AsyncClient", return_value=mock_client):
                return await adapter._download_slack_file_bytes(
                    "https://files.slack.com/file.bin"
                )

        result = asyncio.run(run())
        assert result == b"raw bytes here"

    def test_public_cross_origin_redirects_omit_bearer(self, slack_http):
        import httpx

        targets = {
            "files.slack.com": "https://public.example.test/first",
            "public.example.test": "https://second.example.test/last",
        }

        def respond(request):
            target = targets.get(request.url.host)
            if target:
                return httpx.Response(302, headers={"location": target})
            return httpx.Response(200, content=b"redirected file")

        requests = slack_http(respond)
        adapter = _make_slack_adapter()
        adapter.config.token = "fixture"
        assert asyncio.run(adapter._download_slack_file_bytes(
            "https://files.slack.com/file.bin"
        )) == b"redirected file"
        assert [r.url.host for r in requests] == [
            "files.slack.com", "public.example.test", "second.example.test",
        ]
        assert requests[0].headers["authorization"] == "Bearer fixture"
        assert all("authorization" not in r.headers for r in requests[1:])

    def test_retry_preserves_timeout_and_content(self, monkeypatch, slack_http):
        import httpx

        attempts = []

        def respond(request):
            attempts.append(request)
            if len(attempts) == 1:
                return httpx.Response(429)
            return httpx.Response(200, content=b"retried file")

        requests = slack_http(respond)
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        assert asyncio.run(_make_slack_adapter()._download_slack_file_bytes(
            "https://files.slack.com/retry"
        )) == b"retried file"
        assert len(requests) == 2
        assert all(r.extensions["timeout"]["read"] == 30.0 for r in requests)
        sleep.assert_awaited_once_with(1.5)

    @pytest.mark.parametrize("target", [
        "https://10.0.0.1/private", "https://internal.example.test/private",
    ])
    def test_redirect_rejects_private_resolved_target(self, slack_http, slack_public_dns, target):
        import httpx
        from urllib.parse import urlparse

        requests = slack_http(lambda request: httpx.Response(
            302, headers={"location": target}
        ) if request.url.host == "files.slack.com" else httpx.Response(200, content=b"unsafe"))
        adapter = _make_slack_adapter()
        adapter.config.token = "fixture"
        with pytest.raises(ValueError, match="Blocked"):
            asyncio.run(adapter._download_slack_file_bytes("https://files.slack.com/file.bin"))
        assert len(requests) == 1
        assert requests[0].headers["authorization"] == "Bearer fixture"
        assert any(host == urlparse(target).hostname and answer[0][4][0] == "10.0.0.1"
                   for host, answer in slack_public_dns[1])

    @pytest.mark.parametrize("url", [
        "https://public.example.test/non-slack",
        "https://files.slack.com.evil.test/lookalike",
        "http://files.slack.com/plain",
    ])
    def test_rejected_initial_url_never_sends_token(self, slack_http, url):
        import httpx

        requests = slack_http(lambda request: httpx.Response(200, content=b"unsafe"))
        adapter = _make_slack_adapter()
        with pytest.raises(ValueError, match="Blocked"):
            asyncio.run(adapter._download_slack_file_bytes(url))
        assert requests == []

    def test_private_slack_dns_answer_is_rejected(self, slack_http, slack_public_dns):
        import httpx

        slack_public_dns[0]["files.slack.com"] = "10.0.0.1"
        requests = slack_http(lambda request: httpx.Response(200, content=b"unsafe"))
        with pytest.raises(ValueError, match="Blocked"):
            asyncio.run(_make_slack_adapter()._download_slack_file_bytes(
                "https://files.slack.com/private"
            ))
        assert requests == []
        assert slack_public_dns[1][0][1][0][4][0] == "10.0.0.1"

    def test_unresolved_slack_dns_is_rejected(self, slack_http, slack_public_dns):
        import httpx

        del slack_public_dns[0]["files.slack.com"]
        requests = slack_http(lambda request: httpx.Response(200, content=b"unsafe"))
        with pytest.raises(ValueError, match="Blocked"):
            asyncio.run(_make_slack_adapter()._download_slack_file_bytes(
                "https://files.slack.com/unresolved"
            ))
        assert requests == []
        assert slack_public_dns[1] == []

    def test_dns_rebinding_is_rejected_before_tcp_connect(self, monkeypatch, slack_public_dns):
        import socket
        import tools.url_safety as url_safety
        from httpcore._backends.auto import AutoBackend

        answers = []
        connects = AsyncMock(side_effect=AssertionError("unsafe TCP connect attempted"))

        def resolve(host, port=None):
            ip = "93.184.216.34" if port is None else "10.0.0.1"
            answers.append((host, port, ip))
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 443))]

        monkeypatch.setattr(url_safety, "_getaddrinfo", resolve)
        monkeypatch.setattr(AutoBackend, "connect_tcp", connects)
        with pytest.raises(url_safety.SSRFConnectionBlocked, match="during connect"):
            asyncio.run(_make_slack_adapter()._download_slack_file_bytes(
                "https://files.slack.com/rebound"
            ))
        assert answers == [
            ("files.slack.com", None, "93.184.216.34"),
            ("files.slack.com", 443, "10.0.0.1"),
        ]
        connects.assert_not_awaited()


# ---------------------------------------------------------------------------
# MattermostAdapter._send_url_as_file
# ---------------------------------------------------------------------------

def _make_mm_adapter():
    """Build a minimal MattermostAdapter with mocked internals."""
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    config = PlatformConfig(
        enabled=True, token="mm-token-fake",
        extra={"url": "https://mm.example.com"},
    )
    adapter = MattermostAdapter(config)
    adapter._session = MagicMock()
    adapter._upload_file = AsyncMock(return_value="file-id-123")
    adapter._api_post = AsyncMock(return_value={"id": "post-id-abc"})
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    return adapter


def _make_aiohttp_resp(status: int, content: bytes = b"file bytes",
                       content_type: str = "image/jpeg"):
    """Build a context-manager mock for an aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.content_type = content_type
    resp.read = AsyncMock(return_value=content)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


@patch("tools.url_safety.is_safe_url", return_value=True)
class TestMattermostSendUrlAsFile:
    """Tests for MattermostAdapter._send_url_as_file"""


    def test_retries_on_429_then_succeeds(self, _mock_safe):
        """429 on first attempt is retried; 200 on second attempt succeeds."""
        adapter = _make_mm_adapter()

        resp_429 = _make_aiohttp_resp(429)
        resp_200 = _make_aiohttp_resp(200)
        adapter._session.get = MagicMock(side_effect=[resp_429, resp_200])

        mock_sleep = AsyncMock()

        async def run():
            with patch("asyncio.sleep", mock_sleep):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", None, None
                )

        result = asyncio.run(run())
        assert result.success
        assert adapter._session.get.call_count == 2
        mock_sleep.assert_called_once()


    def test_falls_back_on_client_error(self, _mock_safe):
        """aiohttp.ClientError on every attempt falls back to send() with URL."""
        import aiohttp

        adapter = _make_mm_adapter()

        error_resp = MagicMock()
        error_resp.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientConnectionError("connection refused")
        )
        error_resp.__aexit__ = AsyncMock(return_value=False)
        adapter._session.get = MagicMock(return_value=error_resp)

        async def run():
            with patch("asyncio.sleep", new_callable=AsyncMock):
                return await adapter._send_url_as_file(
                    "C123", "http://cdn.example.com/img.png", None, None
                )

        asyncio.run(run())

        adapter.send.assert_called_once()
        text_arg = adapter.send.call_args[0][1]
        assert "http://cdn.example.com/img.png" in text_arg

