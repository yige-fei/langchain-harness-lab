"""Stop model experiments if disk space or memory pressure become unsafe.

Called before every model request. It only reads system state; it never deletes
or frees anything.
"""
import ctypes
import os
import shutil

MIN_FREE_GIB = 8.0
CRITICAL_PRESSURE = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical


class ResourceLimitReached(RuntimeError):
    """Raised before a model request when a local resource limit is reached.

    `audit_detail` and `user_hint` are fixed-format, path-free strings that callers may
    record per request (they never contain file paths or environment data).
    """

    def __init__(self, reason: str, value: float, limit: float):
        if reason not in ("low_disk", "memory_pressure"):
            raise ValueError("unknown resource-limit reason")
        self.reason, self.value, self.limit = reason, float(value), float(limit)
        super().__init__(f"{self.audit_detail}; stopping")

    @property
    def audit_detail(self) -> str:
        if self.reason == "low_disk":
            return f"low_disk:free_gib={self.value:.2f}<min_gib={self.limit:.2f}"
        return f"memory_pressure:level={int(self.value)}>=critical={int(self.limit)}"

    @property
    def user_hint(self) -> str:
        return "low free disk space" if self.reason == "low_disk" else "critical memory pressure"


def memory_pressure_level() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value), ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), "sysctlbyname failed")
    return value.value


def free_gib() -> float:
    return shutil.disk_usage(os.path.expanduser("~")).free / 2**30


def check() -> None:
    free, level = free_gib(), memory_pressure_level()
    if free < MIN_FREE_GIB:
        raise ResourceLimitReached("low_disk", free, MIN_FREE_GIB)
    if level >= CRITICAL_PRESSURE:
        raise ResourceLimitReached("memory_pressure", level, CRITICAL_PRESSURE)
