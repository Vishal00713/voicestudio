"""Unit tests verifying exception logging and path redaction in telephony session websocket close."""
import asyncio
import logging
from unittest.mock import AsyncMock
import pytest

from services.telephony.session import _close


@pytest.mark.parametrize(
    "code, raw_path, forbidden_prefix",
    [
        (1000, "/Users/telephony/ws/client.sock", "/Users/telephony"),
        (1001, r"C:\Users\telephony\ws\client.sock", r"C:\Users\telephony"),
        (1008, "C:/Users/telephony/ws/client.sock", "C:/Users/telephony"),
    ],
)
def test_websocket_close_exception_logs_debug_and_scrubs_path(
    caplog, code, raw_path, forbidden_prefix
):
    ws = AsyncMock()
    ws.close.side_effect = RuntimeError(f"peer disconnected unexpectedly at {raw_path}")

    with caplog.at_level(logging.DEBUG, logger="omnivoice.telephony"):
        asyncio.run(_close(ws, code=code))

    assert any(
        f"WebSocket close exception for code={code}" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text
