"""Matrix recovery contracts with synthetic SDK, store and monotonic clock."""

import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix import adapter as matrix


@pytest.fixture
def adapter():
    instance = matrix.MatrixAdapter(PlatformConfig(extra={}))
    instance._client = SimpleNamespace(handle_sync=lambda data: [], crypto=None)
    return instance


def encrypted_sync(room="!room:example.org", session="session", **content):
    return {"rooms": {"join": {room: {"timeline": {"events": [{
        "type": "m.room.encrypted", "sender": "@peer:example.org",
        "content": {"algorithm": "m.megolm.v1.aes-sha2", "session_id": session,
                    "sender_key": "sender-key", **content},
    }]}}}}}


@pytest.fixture
def crypto(adapter):
    value = SimpleNamespace(
        crypto_store=SimpleNamespace(has_group_session=AsyncMock(return_value=False),
                                     get_devices=AsyncMock(return_value={})),
        request_room_key=AsyncMock(),
    )
    adapter._client.crypto = value
    return value


@pytest.mark.asyncio
async def test_room_session_requests_deduplicate_for_five_minutes(adapter, crypto, monkeypatch):
    import time
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    data = encrypted_sync()
    events = data["rooms"]["join"]["!room:example.org"]["timeline"]["events"]
    events.append(events[0].copy())
    await adapter._dispatch_sync(data)
    await adapter._dispatch_sync(data)
    assert crypto.request_room_key.await_count == 1
    await adapter._dispatch_sync(encrypted_sync("!other:example.org"))
    await adapter._dispatch_sync(encrypted_sync(session="other"))
    assert crypto.request_room_key.await_count == 3
    now[0] = 299.99
    await adapter._dispatch_sync(data)
    assert crypto.request_room_key.await_count == 3
    now[0] = 300.0
    await adapter._dispatch_sync(data)
    assert crypto.request_room_key.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["request", "devices", "store"])
async def test_key_request_failure_does_not_drop_sync(adapter, crypto, failure):
    target = {"request": crypto.request_room_key, "devices": crypto.crypto_store.get_devices,
              "store": crypto.crypto_store.has_group_session}[failure]
    target.side_effect = OSError("synthetic failure")
    dispatched = []
    adapter._client.handle_sync = lambda data: dispatched.append(data) or []
    await adapter._dispatch_sync(encrypted_sync())
    assert len(dispatched) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [{"session_id": []}, {"session_id": ""},
                                    {"sender_key": None}, {"session_id": 7}])
async def test_malformed_session_skipped_without_dropping_other_events(adapter, crypto, content):
    await adapter._dispatch_sync(encrypted_sync(**content))
    crypto.request_room_key.assert_not_awaited()
    await adapter._dispatch_sync(encrypted_sync())
    crypto.request_room_key.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_store_failure_enters_bounded_sync_retry(adapter, monkeypatch, caplog):
    adapter._client.sync_store = SimpleNamespace(get_next_batch=AsyncMock(side_effect=OSError("store offline")))
    since = []
    sleeps = []

    async def sync(**kwargs):
        since.append(kwargs)
        if len(since) == 1:
            raise OSError("network offline")
        adapter._closing = True
        return {}

    async def sleep(delay):
        sleeps.append(delay)

    adapter._client.sync = sync
    monkeypatch.setattr(matrix.asyncio, "sleep", sleep)
    await adapter._sync_loop()
    assert since == [{"since": None, "timeout": 30000}] * 2
    assert sleeps == [5, 0]
    assert "sync store read failed" in caplog.text
    assert adapter._last_sync_ts > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [None, RuntimeError("synthetic crash")])
async def test_sync_task_reports_unexpected_exit(adapter, caplog, exception):
    import asyncio
    from plugins.platforms.matrix import recovery

    async def finish():
        if exception:
            raise exception

    adapter._sync_loop = finish
    task = recovery.start_sync_loop(adapter)
    try:
        await task
    except RuntimeError:
        pass
    await asyncio.sleep(0)
    assert "sync loop" in caplog.text
    assert ("synthetic crash" if exception else "unexpected") in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", ["closing", "cancelled", "closing_exception"])
async def test_sync_task_shutdown_is_quiet(adapter, caplog, shutdown):
    import asyncio
    from plugins.platforms.matrix import recovery
    errors = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: errors.append(context))

    async def finish():
        if shutdown == "closing_exception":
            raise OSError("shutdown race")

    try:
        adapter._closing = shutdown != "cancelled"
        adapter._sync_loop = finish
        task = recovery.start_sync_loop(adapter)
        if shutdown == "cancelled":
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        assert not errors
        assert "sync loop" not in caplog.text
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("log_level, expected", [(10, 2), (20, 0)])
async def test_sync_heartbeat_is_rate_limited_debug_with_health_timestamp(adapter, monkeypatch, caplog, log_level, expected):
    import logging
    from plugins.platforms.matrix import recovery
    clock = [0.0]
    monkeypatch.setattr(recovery, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    adapter._client.sync_store = SimpleNamespace(get_next_batch=AsyncMock(return_value="resume"))
    ticks = iter([1.0, 59.0, 60.0, 61.0, 120.0])

    async def sync(**kwargs):
        clock[0] = next(ticks)
        if clock[0] == 120.0:
            adapter._closing = True
        return {}

    adapter._client.sync = sync
    with caplog.at_level(log_level):
        await adapter._sync_loop()
    heartbeats = [record for record in caplog.records if "heartbeat" in record.message]
    assert len(heartbeats) == expected
    assert all(record.levelno == logging.DEBUG for record in heartbeats)
    assert adapter._last_sync_ts > 0


@pytest.mark.asyncio
async def test_missing_room_key_requested_before_sdk_dispatch(adapter):
    calls = []

    async def request(*args, **kwargs):
        calls.append((args, kwargs))

    adapter._client.crypto = SimpleNamespace(
        crypto_store=SimpleNamespace(has_group_session=AsyncMock(return_value=False),
                                     get_devices=AsyncMock(return_value={"DEVICE": object()})),
        request_room_key=request,
    )
    adapter._client.handle_sync = lambda data: calls.append("dispatch") or []
    await adapter._dispatch_sync(encrypted_sync())
    assert calls == [(("!room:example.org", "sender-key", "session"),
                      {"from_devices": {"@peer:example.org": ["DEVICE"]}, "timeout": 0}),
                     "dispatch"]


@pytest.mark.asyncio
async def test_no_crypto_still_dispatches_encrypted_sync(adapter):
    dispatched = []
    adapter._client.handle_sync = lambda data: dispatched.append(data) or []
    data = encrypted_sync()
    await adapter._dispatch_sync(data)
    assert dispatched == [data]
    assert not hasattr(adapter._client, "_hermes_room_key_requested")


@pytest.mark.asyncio
async def test_present_room_key_skips_request_and_still_dispatches(adapter, crypto):
    crypto.crypto_store.has_group_session.return_value = True
    dispatched = []
    adapter._client.handle_sync = lambda data: dispatched.append(data) or []
    data = encrypted_sync()
    await adapter._dispatch_sync(data)
    crypto.crypto_store.has_group_session.assert_awaited_once_with("!room:example.org", "session")
    crypto.crypto_store.get_devices.assert_not_awaited()
    crypto.request_room_key.assert_not_awaited()
    assert dispatched == [data]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["store", "devices", "request"])
async def test_blocked_recovery_batch_times_out_before_sdk_dispatch(
    adapter, crypto, monkeypatch, caplog, blocked_stage,
):
    from plugins.platforms.matrix import recovery

    monkeypatch.setattr(recovery, "ROOM_KEY_RECOVERY_TIMEOUT", 0.01, raising=False)
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    never = asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        try:
            await never.wait()
        finally:
            cancelled.set()

    target = {"store": crypto.crypto_store.has_group_session,
              "devices": crypto.crypto_store.get_devices,
              "request": crypto.request_room_key}[blocked_stage]
    target.side_effect = block
    data = encrypted_sync()
    events = data["rooms"]["join"]["!room:example.org"]["timeline"]["events"]
    events.append(encrypted_sync(session="second")["rooms"]["join"]["!room:example.org"]["timeline"]["events"][0])
    dispatched = []
    adapter._client.handle_sync = lambda value: dispatched.append(value) or []

    async with asyncio.timeout(2):
        await adapter._dispatch_sync(data)
    assert entered.is_set()
    assert cancelled.is_set()
    assert dispatched == [data]
    # Exhausting one batch budget must not grant the second event another budget.
    assert target.await_count == 1
    assert len([record for record in caplog.records
                if "room key recovery timed out" in record.message]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["store", "request"])
async def test_outer_cancellation_during_recovery_propagates_without_timeout(
    adapter, crypto, monkeypatch, caplog, blocked_stage,
):
    from plugins.platforms.matrix import recovery

    monkeypatch.setattr(recovery, "ROOM_KEY_RECOVERY_TIMEOUT", 10.0, raising=False)
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    never = asyncio.Event()

    async def block(*args, **kwargs):
        entered.set()
        try:
            await never.wait()
        finally:
            cancelled.set()

    target = (crypto.crypto_store.has_group_session if blocked_stage == "store"
              else crypto.request_room_key)
    target.side_effect = block
    dispatched = []
    adapter._client.handle_sync = lambda data: dispatched.append(data) or []
    task = asyncio.create_task(adapter._dispatch_sync(encrypted_sync()))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert task.cancelled()
        assert cancelled.is_set()
        assert not dispatched
        assert "room key recovery timed out" not in caplog.text
        assert "room key request failed" not in caplog.text
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
