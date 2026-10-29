"""Python code execution for the discovery forum: a LOCAL subprocess, not a security boundary.

What it does. The code an agent writes runs in a fresh Python process (`python -I`), in a throw-away directory that holds copies of the data files
the orchestrator built for that agent (`data/`) and an empty `out/` for what the code produces. The process gets a scrubbed environment (no DATABASE_URL, no
API keys), a wall-clock limit, a memory limit (a Windows job object, or RLIMIT_AS elsewhere), a cap on what it can print, and no child processes where the
platform can forbid them. A prelude then makes the obvious mistakes fail loudly: no sockets, no subprocesses, and `open()` only inside the working directory
and the Python installation.

What it does NOT do. It is not a sandbox against hostile code. The same user account runs it, so code that goes around the prelude (ctypes, os.open, a C
extension) can read what that account can read. The realistic risk is a prompt injected through a passage of a paper (the agents are told to treat
passages as data), and the only channel out is what the code prints, which is stored and shown back to the model. That is why it is OFF unless the server
sets FORUM_ALLOW_LOCAL_CODE=1, why the production compose file never sets it, and why an episode must ask for it explicitly. For a hostile setting use a
container with no network (see the internal project notes).
"""
import csv
import hashlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ENV_FLAG = "FORUM_ALLOW_LOCAL_CODE"
MAX_CODE_CHARS = 20_000
MAX_STDOUT_CHARS = 8_000
MAX_STREAM_BYTES = 400_000          # what is read from a stream before the process is killed for printing too much
MAX_FILES = 6
MAX_FILE_BYTES = 1_000_000
OUTPUT_TYPES = {".png": "image/png", ".csv": "text/csv", ".txt": "text/plain", ".json": "application/json"}
LIBRARIES = ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "networkx")


def enabled():
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes")


def available_libraries():
    return [name for name in LIBRARIES if importlib.util.find_spec(name) is not None]


# --- The data an agent gets ----------------------------------------------------------------------------------------------------------------

def csv_file(columns, rows):
    """(bytes, description) of a CSV with a header. The description is what the agent is told about the file."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    count = 0
    for row in rows:
        writer.writerow(["" if v is None else v for v in row])
        count += 1
    data = buffer.getvalue().encode("utf-8")
    return data, {"columns": list(columns), "rows": count, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


# --- The prelude ---------------------------------------------------------------------------------------------------------------------------

RUNNER = r'''
import builtins, io, os, sys

_root = os.path.realpath(os.getcwd())
_reads = [_root] + [os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix} if p]
if os.name == "nt":
    _reads.append(os.path.realpath(os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "Fonts")))
_real_open = io.open


def _inside(path, roots):
    try:
        real = os.path.realpath(path)
    except (TypeError, ValueError):
        return False
    return any(os.path.commonpath([real, r]) == r for r in roots if os.path.splitdrive(real)[0].lower() == os.path.splitdrive(r)[0].lower())


def _guarded_open(file, mode="r", *args, **kwargs):
    if isinstance(file, int):
        return _real_open(file, mode, *args, **kwargs)
    writing = any(c in mode for c in "wax+")
    if not _inside(file, [_root] if writing else _reads):
        raise PermissionError("the sandbox only lets code read data/ and the Python libraries, and write inside its own folder: " + str(file))
    return _real_open(file, mode, *args, **kwargs)


builtins.open = io.open = _guarded_open


def _no(*args, **kwargs):
    raise PermissionError("the sandbox has no network and starts no processes")


import socket
socket.socket.connect = socket.socket.connect_ex = socket.socket.sendto = _no
socket.create_connection = socket.getaddrinfo = socket.gethostbyname = _no
import subprocess


class _NoPopen(subprocess.Popen):          # a class, not a function: some libraries subclass Popen when they are imported
    def __init__(self, *args, **kwargs):
        _no()


subprocess.Popen = _NoPopen                # run, call and check_* all go through it
for _name in ("system", "popen", "execv", "execve", "execl", "execle", "execlp", "execvp", "spawnl", "spawnv", "startfile", "fork", "forkpty"):
    if hasattr(os, _name):
        setattr(os, _name, _no)

_source = _real_open(os.path.join(_root, "analysis.py"), encoding="utf-8").read()
_globals = {"__name__": "__main__", "__file__": "analysis.py"}
exec(compile(_source, "analysis.py", "exec"), _globals)
'''


# --- Limits --------------------------------------------------------------------------------------------------------------------------------

def _windows_job(memory_bytes):
    """A job object that caps the memory of the process and allows no second process. Returns the handle or None if it cannot be made."""
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    # 0x8 active process limit, 0x100 process memory limit, 0x2000 kill the processes when the job handle closes
    info.BasicLimitInformation.LimitFlags = 0x8 | 0x100 | 0x2000
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = memory_bytes
    if not kernel.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
        kernel.CloseHandle(wintypes.HANDLE(job))
        return None
    return job


def _assign_job(job, process):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    return bool(kernel.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(process._handle))))


def _close_job(job):
    import ctypes
    from ctypes import wintypes
    ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(job))


def _posix_limits(memory_bytes, seconds):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds + 5, seconds + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES * 4, MAX_FILE_BYTES * 4))
    return apply


# --- Running -------------------------------------------------------------------------------------------------------------------------------

@dataclass
class RunResult:
    exit_code: int = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cancelled: bool = False
    killed_for: str = ""
    duration_ms: int = 0
    files: list = field(default_factory=list)          # [{"name", "media_type", "size", "sha256", "content"}]
    dropped: list = field(default_factory=list)        # names of files that were not kept, with the reason
    limits_applied: list = field(default_factory=list)

    @property
    def ok(self):
        return self.exit_code == 0 and not self.timed_out and not self.cancelled


def _drain(stream, sink, state):
    while True:
        chunk = stream.read(4096)
        if not chunk:
            return
        sink.append(chunk)
        state["bytes"] += len(chunk)


def run_python(code, data_files, timeout=60, memory_mb=512, cancel=None):
    """Run `code` once. `data_files` is {name: bytes}, written to data/. Never raises for a failure of the code: the outcome is in the result."""
    code = code if isinstance(code, str) else str(code)
    if len(code) > MAX_CODE_CHARS:
        return RunResult(exit_code=2, stderr=f"The code is longer than {MAX_CODE_CHARS} characters: split the analysis.")
    work = Path(tempfile.mkdtemp(prefix="forum-code-"))
    result = RunResult()
    job = None
    started = time.monotonic()
    try:
        (work / "data").mkdir()
        (work / "out").mkdir()
        (work / ".mpl").mkdir()
        for name, content in data_files.items():
            (work / "data" / name).write_bytes(content)
        (work / "analysis.py").write_text(code, encoding="utf-8")
        (work / "runner.py").write_text(RUNNER, encoding="utf-8")
        env = {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(work / ".mpl"), "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "TEMP": str(work), "TMP": str(work), "TMPDIR": str(work)}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        kwargs = {}
        memory_bytes = int(memory_mb) * 1024 * 1024
        if os.name != "nt":
            kwargs["preexec_fn"] = _posix_limits(memory_bytes, int(timeout))
            result.limits_applied += ["rlimit_as", "rlimit_cpu", "rlimit_fsize"]
        else:
            kwargs["creationflags"] = 0x08000000      # CREATE_NO_WINDOW
        process = subprocess.Popen([sys.executable, "-I", "-B", "-X", "utf8", "runner.py"], cwd=work, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        if os.name == "nt":
            job = _windows_job(memory_bytes)
            if job and _assign_job(job, process):
                result.limits_applied += ["job_memory", "job_single_process"]
        out_bytes, err_bytes, state = [], [], {"bytes": 0}
        threads = [threading.Thread(target=_drain, args=(process.stdout, out_bytes, state), daemon=True),
                   threading.Thread(target=_drain, args=(process.stderr, err_bytes, state), daemon=True)]
        for t in threads:
            t.start()
        while process.poll() is None:
            if cancel is not None and cancel.is_set():
                result.cancelled, result.killed_for = True, "a person pressed Stop"
            elif time.monotonic() - started > timeout:
                result.timed_out, result.killed_for = True, f"the time limit of {timeout} s"
            elif state["bytes"] > MAX_STREAM_BYTES:
                result.killed_for = "printing more than the output limit"
            if result.killed_for:
                process.kill()          # only the process started here: nothing else on the machine is touched
                break
            time.sleep(0.05)
        process.wait()
        for t in threads:
            t.join(timeout=2)
        result.exit_code = process.returncode
        result.stdout = b"".join(out_bytes).decode("utf-8", "replace")[:MAX_STDOUT_CHARS]
        result.stderr = b"".join(err_bytes).decode("utf-8", "replace")[-MAX_STDOUT_CHARS:]
        for path in sorted((work / "out").rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(work / "out").as_posix()
            kind = OUTPUT_TYPES.get(path.suffix.lower())
            if kind is None:
                result.dropped.append(f"{name}: only png, csv, txt and json are kept")
            elif len(result.files) >= MAX_FILES:
                result.dropped.append(f"{name}: at most {MAX_FILES} files are kept")
            elif path.stat().st_size > MAX_FILE_BYTES:
                result.dropped.append(f"{name}: larger than {MAX_FILE_BYTES // 1000} kB")
            else:
                content = path.read_bytes()
                result.files.append({"name": name, "media_type": kind, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "content": content})
    finally:
        if job:
            _close_job(job)
        shutil.rmtree(work, ignore_errors=True)
        result.duration_ms = int((time.monotonic() - started) * 1000)
    return result


def environment():
    """What the code ran on, kept with every run so that a result can be reproduced or explained later."""
    from importlib import metadata
    names = {"sklearn": "scikit-learn"}
    versions = {}
    for lib in available_libraries():
        try:
            versions[lib] = metadata.version(names.get(lib, lib))
        except metadata.PackageNotFoundError:
            versions[lib] = "unknown"
    return {"python": sys.version.split()[0], "platform": sys.platform, "libraries": versions}
