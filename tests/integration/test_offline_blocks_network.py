import socket

import pytest

from douglas.net.offline_guard import (
    DouglasOfflineError,
    activate_offline_guard,
    deactivate_offline_guard,
)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_offline_guard_blocks_sockets(monkeypatch):
    monkeypatch.setenv("DOUGLAS_OFFLINE", "1")
    activate_offline_guard()
    try:
        with pytest.raises(DouglasOfflineError):
            sock = socket.socket()
            try:
                sock.connect(("example.com", 80))
            finally:
                sock.close()
    finally:
        deactivate_offline_guard()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_offline_guard_blocks_requests(monkeypatch):
    pytest.importorskip("requests")
    import requests

    monkeypatch.setenv("DOUGLAS_OFFLINE", "1")
    activate_offline_guard()
    try:
        with pytest.raises(DouglasOfflineError):
            requests.get("https://example.com", timeout=1)
    finally:
        deactivate_offline_guard()
