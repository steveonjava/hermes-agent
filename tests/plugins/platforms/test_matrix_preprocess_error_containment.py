"""Offline preprocessing failures must not skip normal SDK sibling dispatch."""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import json
import pytest
from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter
from mautrix.types import EventType, ToDeviceEvent


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "decrypt", "room_key", "forwarded_key", "dispatch"])
async def test_key_preprocessing_failure_does_not_skip_sdk_after_cursor_save(failure, monkeypatch, caplog):
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("network prohibited"))
    adapter = MatrixAdapter(PlatformConfig(extra={}))
    decoded_type = (EventType.FORWARDED_ROOM_KEY if failure == "forwarded_key" else
                    ToDeviceEvent.deserialize({"type": "m.key.verification.ready",
                        "sender": "@peer:example.org", "content": {}}).type
                    if failure == "dispatch" else EventType.ROOM_KEY)
    decoded = SimpleNamespace(type=decoded_type)
    crypto = SimpleNamespace(
        _decrypt_olm_event=AsyncMock(return_value=decoded),
        _receive_room_key=AsyncMock(), _receive_forwarded_room_key=AsyncMock(),
    )
    fault = OSError("synthetic preprocess store failure")
    if failure == "decrypt":
        crypto._decrypt_olm_event.side_effect = fault
    elif failure == "room_key":
        crypto._receive_room_key.side_effect = fault
    elif failure == "forwarded_key":
        crypto._receive_forwarded_room_key.side_effect = fault
    sdk = Mock(return_value=[])
    dispatch = Mock(return_value=[])
    if failure == "dispatch":
        dispatch.side_effect = fault
    store = SimpleNamespace(put_next_batch=AsyncMock())
    client = SimpleNamespace(crypto=crypto, handle_sync=sdk, sync_store=store, dispatch_event=dispatch)
    adapter._client = client
    data = {"next_batch": "diagnostic-next", "rooms": {"join": {"!room:example.org": {
        "timeline": {"events": [{"type": "m.room.message", "sender": "@peer:example.org",
                                  "content": {"msgtype": "m.text", "body": "sibling"}}]}}}},
        "to_device": {"events": [{"type": "m.room.encrypted", "sender": "@peer:example.org",
            "content": {"algorithm": "m.olm.v1.curve25519-aes-sha2", "sender_key": "synthetic-curve",
                        "ciphertext": {}}}]}}
    with caplog.at_level(logging.DEBUG):
        assert await adapter._absorb_sync(client, data) == "diagnostic-next"
    assert "synthetic preprocess store failure" not in caplog.text
    store.put_next_batch.assert_awaited_once_with("diagnostic-next")
    crypto._decrypt_olm_event.assert_awaited_once()
    if failure == "room_key":
        crypto._receive_room_key.assert_awaited_once_with(decoded)
    if failure == "forwarded_key":
        crypto._receive_forwarded_room_key.assert_awaited_once_with(decoded)
    if failure == "dispatch":
        dispatch.assert_called_once()
    print(json.dumps({"failure_seam": failure, "resume_token_persisted": True,
                      "sdk_dispatch_calls": sdk.call_count}))
    sdk.assert_called_once_with(data)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("network prohibited"))


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["room", "timeline", "event", "cache"])
async def test_missing_key_scan_error_continues_sdk_dispatch(shape, caplog):
    adapter = MatrixAdapter(PlatformConfig(extra={}))
    sibling = AsyncMock()
    sdk = Mock(side_effect=lambda data: [asyncio.create_task(sibling())])
    client = SimpleNamespace(crypto=SimpleNamespace(), handle_sync=sdk)
    adapter._client = client
    room = {"timeline": {"events": []}}
    data = {"rooms": {"join": {"!room:example.org": room}}}
    if shape == "room":
        data["rooms"]["join"]["!room:example.org"] = None
    elif shape == "timeline":
        room["timeline"] = None
    elif shape == "event":
        room["timeline"]["events"] = [None]
    else:
        client._hermes_room_key_requested = {("room", "session"): "private-cache-value"}
    with caplog.at_level(logging.DEBUG):
        await adapter._dispatch_sync(data)
    sdk.assert_called_once_with(data)
    sibling.assert_awaited_once()
    assert "room key recovery failed" in caplog.text
    assert "private-cache-value" not in caplog.text


def olm_event():
    return {"type": "m.room.encrypted", "sender": "@peer:example.org",
            "content": {"algorithm": "m.olm.v1.curve25519-aes-sha2",
                        "sender_key": "synthetic-curve", "ciphertext": {}}}


@pytest.mark.asyncio
async def test_later_bad_raw_event_preserves_tasks_and_unprocessed_events(caplog):
    adapter = MatrixAdapter(PlatformConfig(extra={}))
    decoded = ToDeviceEvent.deserialize({"type": "m.key.verification.ready",
                                        "sender": "@peer:example.org", "content": {}})
    preprocessed = AsyncMock()
    sibling = AsyncMock()
    scheduled = []

    def dispatch(*args, **kwargs):
        task = asyncio.create_task(preprocessed())
        scheduled.append(task)
        return [task]

    first, failed, last = olm_event(), olm_event(), olm_event()
    plain = {"type": "m.dummy", "content": {}}
    data = {"to_device": {"events": [first, None, failed, plain, last]}}
    crypto = SimpleNamespace(_decrypt_olm_event=AsyncMock(side_effect=[
        decoded, OSError("PRIVATE-CRYPTO-CONTENT"), decoded]))
    seen = []

    def handle_sync(value):
        seen.extend(value["to_device"]["events"])
        return [asyncio.create_task(sibling())]

    adapter._client = SimpleNamespace(crypto=crypto, dispatch_event=Mock(side_effect=dispatch),
                                     handle_sync=Mock(side_effect=handle_sync))
    with caplog.at_level(logging.DEBUG):
        try:
            await adapter._dispatch_sync(data)
            assert all(task.done() for task in scheduled)
        finally:
            # Drain real scheduled tasks even on RED, without concealing the failure.
            await asyncio.gather(*scheduled)
    assert seen == [None, failed, plain]
    assert seen[1] is failed
    assert data["to_device"]["events"] == seen
    assert len(scheduled) == 2
    assert preprocessed.await_count == 2
    sibling.assert_awaited_once()
    assert "PRIVATE-CRYPTO-CONTENT" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["store", "devices", "request"])
async def test_room_key_request_error_is_redacted(seam, caplog):
    adapter = MatrixAdapter(PlatformConfig(extra={}))
    store = SimpleNamespace(has_group_session=AsyncMock(return_value=False),
                            get_devices=AsyncMock(return_value={}))
    crypto = SimpleNamespace(crypto_store=store, request_room_key=AsyncMock())
    target = {"store": store.has_group_session, "devices": store.get_devices,
              "request": crypto.request_room_key}[seam]
    target.side_effect = OSError("PRIVATE-REQUEST-CONTENT")
    sdk = Mock(return_value=[])
    adapter._client = SimpleNamespace(crypto=crypto, handle_sync=sdk)
    data = {"rooms": {"join": {"!room:example.org": {"timeline": {"events": [{
        "type": "m.room.encrypted", "sender": "@peer:example.org",
        "content": {"algorithm": "m.megolm.v1.aes-sha2", "session_id": "session",
                    "sender_key": "sender-key"}}]}}}}}
    with caplog.at_level(logging.DEBUG):
        await adapter._dispatch_sync(data)
    sdk.assert_called_once_with(data)
    assert "room key request failed" in caplog.text
    assert "PRIVATE-REQUEST-CONTENT" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["decrypt", "room_key", "forwarded_key"])
async def test_outer_cancellation_during_olm_preprocessing_propagates(seam, caplog):
    adapter = MatrixAdapter(PlatformConfig(extra={}))
    entered = asyncio.Event()

    async def block(*args):
        entered.set()
        await asyncio.Event().wait()

    decoded = SimpleNamespace(type=(EventType.FORWARDED_ROOM_KEY
                                   if seam == "forwarded_key" else EventType.ROOM_KEY))
    crypto = SimpleNamespace(_decrypt_olm_event=AsyncMock(return_value=decoded),
                             _receive_room_key=AsyncMock(), _receive_forwarded_room_key=AsyncMock())
    target = {"decrypt": crypto._decrypt_olm_event, "room_key": crypto._receive_room_key,
              "forwarded_key": crypto._receive_forwarded_room_key}[seam]
    target.side_effect = block
    sdk = Mock(return_value=[])
    adapter._client = SimpleNamespace(crypto=crypto, handle_sync=sdk)
    task = asyncio.create_task(adapter._dispatch_sync({"to_device": {"events": [olm_event()]}}))
    try:
        async with asyncio.timeout(2):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        sdk.assert_not_called()
        assert not caplog.records
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_sdk_failure_is_not_reported_as_successful_dispatch():
    adapter = MatrixAdapter(PlatformConfig(extra={}))
    sdk = Mock(side_effect=OSError("SDK genuinely failed"))
    adapter._client = SimpleNamespace(crypto=None, handle_sync=sdk)
    with pytest.raises(OSError, match="SDK genuinely failed"):
        await adapter._dispatch_sync({})
    sdk.assert_called_once_with({})


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["PRIVATE-MALFORMED-CONTAINER", {"events": 7}])
async def test_malformed_to_device_container_does_not_skip_sdk(caplog, payload):
    adapter = MatrixAdapter(PlatformConfig(extra={}))
    sdk = Mock(return_value=[])
    adapter._client = SimpleNamespace(crypto=SimpleNamespace(), handle_sync=sdk)
    data = {"to_device": payload}
    await adapter._dispatch_sync(data)
    sdk.assert_called_once_with(data)
    assert data["to_device"] is payload
    assert "PRIVATE-MALFORMED-CONTAINER" not in caplog.text
