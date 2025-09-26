import threading
import time

import pytest

from douglas.agents.locks import FileLockAcquisitionError, FileLockManager


def test_lock_manager_blocks_until_release(tmp_path):
    manager = FileLockManager(
        tmp_path / "locks", default_timeout=2.0, poll_interval=0.05
    )
    target = tmp_path / "shared.txt"
    target.write_text("data", encoding="utf-8")
    order: list[str] = []

    def _holder():
        with manager.acquire([target]):
            order.append("holder")
            time.sleep(0.3)

    thread = threading.Thread(target=_holder)
    thread.start()
    time.sleep(0.1)

    with manager.acquire([target]):
        order.append("second")

    thread.join()
    assert order == ["holder", "second"]


def test_lock_manager_times_out(tmp_path):
    manager = FileLockManager(
        tmp_path / "locks", default_timeout=0.2, poll_interval=0.05
    )
    target = tmp_path / "shared.txt"
    target.write_text("data", encoding="utf-8")

    event = threading.Event()

    def _holder():
        with manager.acquire([target]):
            event.set()
            time.sleep(0.5)

    thread = threading.Thread(target=_holder)
    thread.start()
    event.wait(timeout=1)

    with pytest.raises(FileLockAcquisitionError):
        with manager.acquire([target], timeout=0.1):
            pass

    thread.join()
