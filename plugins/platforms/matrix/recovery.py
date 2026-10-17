"""Bounded recovery for missing encrypted room sessions."""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


def sync_heartbeat(last_heartbeat, room_count):
    """Keep liveness available to debug logs without normal-level poll noise."""
    now = time.monotonic()
    if now - last_heartbeat >= 60.0:
        logger.debug("Matrix: sync loop heartbeat (rooms=%d)", room_count)
        return now
    return last_heartbeat


def start_sync_loop(adapter):
    """Attach exit reporting at task creation so unexpected deaths stay visible."""
    def report_exit(task):
        if task.cancelled():
            return
        exc = task.exception()
        if adapter._closing:
            return
        if exc is not None:
            logger.error("Matrix: sync loop failed: %r", exc)
        else:
            logger.error("Matrix: sync loop ended unexpectedly")

    task = asyncio.create_task(adapter._sync_loop())
    task.add_done_callback(report_exit)
    return task


ROOM_KEY_RECOVERY_TIMEOUT = 10.0


async def request_missing_room_keys(client, sync_data):
    """Limit the whole recovery batch so stalled crypto I/O cannot stop sync."""
    try:
        async with asyncio.timeout(ROOM_KEY_RECOVERY_TIMEOUT):
            await _request_missing_room_keys(client, sync_data)
    except TimeoutError:
        logger.warning("Matrix: room key recovery timed out; continuing sync dispatch")
    except Exception:
        logger.warning("Matrix: room key recovery failed; continuing sync dispatch")


async def _request_missing_room_keys(client, sync_data):
    """Request sessions before SDK decryption, which only logs missing keys."""
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        return
    now = time.monotonic()
    requested = getattr(client, "_hermes_room_key_requested", {})
    requested = {key: stamp for key, stamp in requested.items() if now - stamp < 300.0}
    client._hermes_room_key_requested = requested
    for room_id, room in sync_data.get("rooms", {}).get("join", {}).items():
        for event in room.get("timeline", {}).get("events", []):
            content = event.get("content") or {}
            if event.get("type") != "m.room.encrypted" or content.get("algorithm") != "m.megolm.v1.aes-sha2":
                continue
            session = content.get("session_id")
            sender_key = content.get("sender_key")
            sender = event.get("sender")
            if not all(isinstance(value, str) and value.strip()
                       for value in (room_id, session, sender_key, sender)):
                continue
            key = (room_id, session)
            if key in requested:
                continue
            requested[key] = now
            try:
                if await crypto.crypto_store.has_group_session(room_id, session):
                    continue
                devices = await crypto.crypto_store.get_devices(sender) or {}
                await crypto.request_room_key(room_id, sender_key, session,
                                              from_devices={sender: list(devices)}, timeout=0)
            except Exception:
                logger.warning("Matrix: room key request failed; continuing sync dispatch")
