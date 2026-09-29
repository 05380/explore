"""Lifecycle helpers for standalone Isaac Sim processes.

The module intentionally has no eager Omniverse imports. Callers must create
``SimulationApp`` before invoking :func:`close_simulation_app_safely`.
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Any


def active_exception_exit_code() -> int:
    """Map the exception currently propagating through a ``finally`` block."""
    error = sys.exc_info()[1]
    if error is None:
        return 0
    if isinstance(error, SystemExit):
        code = error.code
        if code is None:
            return 0
        if isinstance(code, int):
            return int(code)
        return 1
    return 1


def close_simulation_app_safely(
    simulation_app: Any,
    *,
    hard_exit_after_shutdown: bool = False,
    process_exit_code: int = 0,
) -> None:
    """Drain Replicator and close the USD stage before Kit plugin shutdown.

    Isaac Sim 2023.1 can segfault in ``SimulationApp.close`` when RTX camera
    render products outlive their USD stage or Replicator work. Cleanup errors
    are reported, but the final app close is still attempted.
    """
    try:
        import omni.replicator.core as rep

        status = rep.orchestrator.get_status()
        stopped = rep.orchestrator.Status.STOPPED
        stopping = rep.orchestrator.Status.STOPPING
        if status not in (stopped, stopping):
            rep.orchestrator.stop()
        if status != stopped:
            rep.orchestrator.wait_until_complete()
        print("ISAAC_REPLICATOR_DRAINED", flush=True)
    except BaseException as error:
        print(
            f"ISAAC_REPLICATOR_DRAIN_EXCEPTION={type(error).__name__}: {error}",
            flush=True,
        )
    finally:
        status = None
        rep = None

    try:
        import omni.usd

        usd_context = omni.usd.get_context()
        if usd_context.can_close_stage():
            usd_context.close_stage()
            print("ISAAC_USD_STAGE_CLOSED", flush=True)
        else:
            print("ISAAC_USD_STAGE_CLOSE_SKIPPED", flush=True)
    except BaseException as error:
        print(
            f"ISAAC_USD_STAGE_CLOSE_EXCEPTION={type(error).__name__}: {error}",
            flush=True,
        )
    finally:
        usd_context = None

    # Destroy Python wrappers around native Replicator/USD interfaces while
    # their plugins are still loaded. Isaac Sim 2023.1 may otherwise finish
    # native shutdown successfully and then crash during interpreter GC.
    gc.collect()

    simulation_app.close()
    print("ISAAC_NATIVE_SHUTDOWN_RETURNED", flush=True)
    if hard_exit_after_shutdown:
        # All reports are written and Kit has completed native shutdown. Skip
        # only Python interpreter finalization, whose late C++ wrapper
        # destructors are unsafe with this locked Isaac 2023.1 build.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(process_exit_code))
