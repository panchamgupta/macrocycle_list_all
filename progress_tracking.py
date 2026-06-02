import sys
import time


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def format_elapsed(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


_STAGE_WEIGHTS: dict = {
    "Exclusion rules": 1,
    "SDF parse": 20,
    "Interaction CSV": 2,
    "Molecule ranking": 3,
    "Scaffold summary": 18,
    "Central/unique": 3,
    "Rare motif": 6,
    "Scaffold panel": 8,
    "Per-scaffold CSV": 15,
    "Figure generation": 10,
    "Core CSV writing": 5,
    "HTML report": 8,
    "Run manifest": 1,
}

_PBAR = None


class ProgressBar:
    """Single-line ANSI progress timeline written to stderr."""

    _BAR_W = 34

    def __init__(self, t0: float, stage_weights: dict) -> None:
        self._t0 = t0
        self._tty = sys.stderr.isatty()
        total_w = float(sum(stage_weights.values())) or 1.0
        cum = 0.0
        self._st: dict = {}
        for name, weight in stage_weights.items():
            self._st[name] = [cum / total_w, weight / total_w]
            cum += weight
        self._overall = 0.0
        self._last_len = 0
        self._last_emit: float = 0.0

    def _key(self, phase: str):
        best = None
        best_n = -1
        for key in self._st:
            if phase.startswith(key) and len(key) > best_n:
                best = key
                best_n = len(key)
        return best

    def _advance(self, phase: str, done, total) -> None:
        key = self._key(phase)
        if key is None:
            return
        base, weight = self._st[key]
        within = (done / total) if (done is not None and total and total > 0) else 0.3
        self._overall = min(1.0, base + weight * within)

    def _bar_line(self, phase: str, done, total, extra) -> str:
        elapsed = time.time() - self._t0
        pct = self._overall * 100.0
        filled = int(self._BAR_W * min(1.0, self._overall))
        bar = "\u2588" * filled + "\u2591" * (self._BAR_W - filled)
        if self._overall > 0.02:
            eta_s = max(0.0, elapsed / self._overall * (1.0 - self._overall))
            eta_str = f"ETA {format_elapsed(eta_s)}"
        else:
            eta_str = "ETA --:--"
        label = phase
        if done is not None and total:
            label += f" {done}/{total}"
        elif done is not None:
            label += f" {done}"
        if extra:
            label += f"  {extra}"
        return (
            f"\r\033[K[{format_elapsed(elapsed)}] {pct:5.1f}%"
            f"  [{bar}]  {label[:60]:<60}  {eta_str}"
        )

    def update(self, phase: str, done=None, total=None, extra=None) -> None:
        self._advance(phase, done, total)
        now = time.time()
        if self._tty:
            line = self._bar_line(phase, done, total, extra)
            sys.stderr.write(line)
            self._last_len = len(line)
            sys.stderr.flush()
        else:
            if now - self._last_emit >= 2.0 or (done is not None and done == total):
                elapsed = time.time() - self._t0
                pct = self._overall * 100.0
                core = f"[{format_elapsed(elapsed)}] {pct:5.1f}%  {phase}"
                if done is not None and total:
                    core += f" {done}/{total}"
                elif done is not None:
                    core += f" {done}"
                if extra:
                    core += f" | {extra}"
                sys.stderr.write(core + "\n")
                sys.stderr.flush()
                self._last_emit = now

    def complete(self, phase: str, extra: str = "") -> None:
        key = self._key(phase)
        if key is not None:
            base, weight = self._st[key]
            self._overall = min(1.0, base + weight)
        elapsed = time.time() - self._t0
        if self._tty and self._last_len:
            sys.stderr.write("\r\033[K")
        msg = f"[{format_elapsed(elapsed)}] \u2713  {phase}"
        if extra:
            msg += f"  |  {extra}"
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
        self._last_len = 0
        self._last_emit = time.time()

    def finish(self) -> None:
        if self._tty and self._last_len:
            sys.stderr.write("\r\033[K")
        elapsed = time.time() - self._t0
        sys.stderr.write(
            f"[{format_elapsed(elapsed)}] \u2714  All stages complete"
            f"  |  total {format_elapsed(elapsed)}\n"
        )
        sys.stderr.flush()


def start_progress_bar(start_ts, stage_weights=None):
    global _PBAR
    _PBAR = ProgressBar(start_ts, stage_weights or _STAGE_WEIGHTS)
    return _PBAR


def finish_progress_bar():
    global _PBAR
    if _PBAR is not None:
        _PBAR.finish()
        _PBAR = None


def progress_log(start_ts, phase, done=None, total=None, extra=None):
    global _PBAR
    if _PBAR is not None:
        phase_lc = phase.lower()
        is_complete = (
            phase_lc.endswith(" complete")
            or phase_lc.endswith(" loaded")
            or phase_lc.endswith(" done")
            or phase_lc == "run started"
            or phase_lc.startswith("prefix cleanup")
        )
        if is_complete:
            _PBAR.complete(phase, extra=extra or "")
        else:
            _PBAR.update(phase, done, total, extra)
        return
    elapsed = format_elapsed(time.time() - start_ts)
    if done is None:
        core = f"[{elapsed}] {phase}"
    elif total is None:
        core = f"[{elapsed}] {phase}: {done}"
    else:
        pct = (100.0 * done / total) if total else 0.0
        core = f"[{elapsed}] {phase}: {done}/{total} ({pct:.1f}%)"
    if extra:
        core = f"{core} | {extra}"
    eprint(core)