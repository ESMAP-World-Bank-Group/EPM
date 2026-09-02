"""
Send to the data store what changed, and nothing else.

Both uploaders used to re-send their whole folder on every publish. On the results side
that meant about 10 GB every time, including 4 GB of pDispatchComplete re-read and
re-split locally before anything was even sent, whether the run had moved or not.

The rule applied here is the rsync rule: a file is sent again only when the remote does
not have it, does not have it at the same size, or has it older than the local file.

Two decisions explain the rest of this file.

No hashing. Comparing md5 would be exact, but hashing 10 GB costs several minutes on
every publish, which moves the slowness instead of removing it. Size plus date is enough
for files that a run rewrites in full.

No local state file. The bucket is the reference, read in a single listing call. Nothing
to corrupt, nothing to gitignore, nothing that lies when publishing from another machine
or when someone has touched the store. And the first publish after this change is already
fast, because there is no 10 GB priming pass.

When in doubt, send. A skip requires positive proof that the remote copy is good and
newer. A clock skew, a different size, a missing key, a failed listing, all of those lead
to an upload. The worst case is a needless upload, never a missing file.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone

# Enough threads to cover the round trip latency behind the proxy, not enough for R2 to
# start answering 503.
DEFAULT_JOBS = int(os.environ.get("EPM_UPLOAD_JOBS", "8"))

# The clocks of this machine and of the store are not synchronised to the second.
MTIME_TOLERANCE_S = 2.0

RETRIES = 3


def add_sync_args(ap):
    """The three options shared by both uploaders."""
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"parallel uploads (default {DEFAULT_JOBS})")
    ap.add_argument("--force", action="store_true",
                    help="send everything again, even what is already up to date")
    ap.add_argument("--check", action="store_true",
                    help="compare local and remote without sending anything")
    return ap


def human(nbytes):
    """Byte count as a short readable string."""
    n = float(nbytes)
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _size(path):
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _epoch(info):
    ts = info.get("LastModified") or info.get("last_modified")
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if ts.tzinfo is None:            # R2 answers in UTC, not always saying so
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def remote_index(fs, bucket, prefix):
    """Path relative to the prefix -> (size, epoch seconds). Empty when the listing fails."""
    root = f"{bucket}/{prefix}"
    try:
        found = fs.find(root, detail=True)
    except Exception as e:                                    # pragma: no cover
        print(f"  (remote listing unavailable: {e!r} -> everything will be sent again)")
        return {}
    idx = {}
    head = root + "/"
    for key, info in found.items():
        if not key.startswith(head):
            continue
        idx[key[len(head):]] = (int(info.get("size") or 0), _epoch(info))
    return idx


def is_current(local, entry):
    """Does the remote already hold this file, in this version?"""
    if not entry:
        return False
    size, mtime = entry
    try:
        st = local.stat()
    except OSError:
        return False
    return st.st_size == size and mtime + MTIME_TOLERANCE_S >= st.st_mtime


def plan(tasks, idx, force=False):
    """Split the (local path, relative key) pairs between what is sent and what is kept."""
    todo, skipped = [], []
    for local, rel in tasks:
        (todo if force or not is_current(local, idx.get(rel)) else skipped).append((local, rel))
    return todo, skipped


def upload_many(fs, bucket, prefix, tasks, idx, jobs=DEFAULT_JOBS, force=False,
                check=False, label="files"):
    """Send what has to be sent, one progress line per file. Returns (sent, skipped, failed).

    The progress lines matter: uploads run at a few Mbit/s through the proxy, so a large
    publish takes minutes and a silent console reads as a hang.
    """
    todo, skipped = plan(tasks, idx, force)
    total_bytes = sum(_size(p) for p, _ in todo)
    print(f"  {len(tasks)} {label}: {len(todo)} to send ({human(total_bytes)}), "
          f"{len(skipped)} already up to date")
    if check:
        for _, rel in todo[:20]:
            print(f"    [check] would send: {rel}")
        if len(todo) > 20:
            print(f"    [check] ... and {len(todo) - 20} more")
        return 0, len(skipped), []
    if not todo:
        return 0, len(skipped), []

    n = len(todo)
    width = len(str(n))
    started = time.time()
    lock = threading.Lock()
    state = {"done": 0, "bytes": 0}
    failed = []

    def send(item):
        local, rel = item
        for attempt in range(1, RETRIES + 1):
            try:
                fs.put_file(str(local), f"{bucket}/{prefix}/{rel}")
                return None
            except Exception as e:
                if attempt == RETRIES:
                    return (rel, repr(e))
                time.sleep(2 ** attempt)   # the WB proxy drops, the next try goes through

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(send, item): item for item in todo}
        for fut in as_completed(futures):
            local, rel = futures[fut]
            err = fut.result()
            with lock:
                state["done"] += 1
                state["bytes"] += _size(local)
                i = state["done"]
                sent_bytes = state["bytes"]
                if err:
                    failed.append(err)
                    print(f"    [{i:{width}}/{n}] FAILED  {rel}")
                else:
                    pct = 100 * sent_bytes / total_bytes if total_bytes else 100
                    print(f"    [{i:{width}}/{n}] {pct:3.0f}%  {human(_size(local)):>9}  {rel}")

    elapsed = max(time.time() - started, 0.001)
    print(f"    {human(state['bytes'])} in {elapsed:.0f} s ({human(state['bytes'] / elapsed)}/s)")
    return n - len(failed), len(skipped), failed


def report(failed):
    """Summarise the failures and give the exit code. 0 when everything went through."""
    if not failed:
        return 0
    print(f"  FAILURES: {len(failed)} file(s) not sent")
    for rel, err in failed[:10]:
        print(f"    {rel}: {err}")
    if len(failed) > 10:
        print(f"    ... and {len(failed) - 10} more")
    return 1
