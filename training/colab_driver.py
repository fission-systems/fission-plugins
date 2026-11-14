"""Runs on the Colab VM (uploaded by colab_run.py): install, run each fine-tune job in order, pack results.

Reads /content/ft/jobs.json = {"pip": [...], "runs": [{"name": str, "args": [str, ...]}]}. Each run executes
`python /content/ft/finetune.py <args> --data /content/ft/data --out /content/ft/out/<name>` and its stdout goes to
out/<name>.log. Status is written to /content/ft/status.json after every phase; results are packed into
/content/ft/results.tar.gz (logs, run.json, log.jsonl and adapters; never the base-model weights).
A failed run is recorded and the next one still starts.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import time

ROOT = pathlib.Path("/content/ft")
STATUS = ROOT / "status.json"


def status(**kv) -> None:
    current = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    current.update(kv, updated=time.time())
    STATUS.write_text(json.dumps(current, indent=1))


def sh(cmd: list[str], log: pathlib.Path, timeout: int | None = None) -> int:
    with log.open("ab") as out:
        out.write(("$ " + " ".join(cmd) + "\n").encode())
        out.flush()
        return subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=timeout).returncode


def main() -> None:
    jobs = json.loads((ROOT / "jobs.json").read_text())
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    status(state="running", phase="setup", runs={})
    env_log = out / "env.log"
    sh(["nvidia-smi"], env_log)
    for spec in jobs["pip"]:
        if "uninstall" in spec:  # e.g. the image's torchao 0.10 breaks peft 0.21 at LoRA injection
            rc = sh([sys.executable, "-m", "pip", "uninstall", "-y", *spec["uninstall"]], env_log, 300)
            status(**{f"pip-uninstall:{' '.join(spec['uninstall'])}": rc})
            continue
        rc = sh([sys.executable, "-m", "pip", "install", "-q", *spec["packages"]], env_log, spec.get("timeout", 900))
        status(**{f"pip:{' '.join(spec['packages'])}": rc})
    sh([sys.executable, "-m", "pip", "list", "--format=freeze"], env_log)
    results = {}
    for run in jobs["runs"]:
        name = run["name"]
        status(phase=f"run:{name}")
        started = time.time()
        cmd = [sys.executable, str(ROOT / "finetune.py"), *run["args"], "--data", str(ROOT / "data"), "--out", str(out / name)]
        rc = sh(cmd, out / f"{name}.log")
        results[name] = dict(returncode=rc, seconds=round(time.time() - started, 1))
        status(runs=results)
    archive = ROOT / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname="out")
    status(state="complete", phase="done", archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - report any failure to the host
        status(state="failed", error=repr(exc))
        raise
