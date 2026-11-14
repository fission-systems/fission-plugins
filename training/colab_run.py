"""Host side of an M2a Colab run: package, allocate one named GPU, run, poll, download, always stop.

Uses the local `colab` CLI with credentials already on this machine (nothing is uploaded but the training script, the
D0 train/val files and the job list). Allocating a GPU spends Colab compute units: run it only after deciding to.

  --plan probe   one short run per candidate (speed and memory, kernels installed or not); no result is a score
  --plan curve   the M2a grid: --models x --sizes, one run each, in order (long)
  --package-only build the tarball and job list and print them; no GPU

usage: colab_run.py --data <d0 dir> --output <new dir> --session <name> --plan probe [--gpu L4] [--budget-min 60]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {"coder1.5b": "Qwen/Qwen2.5-Coder-1.5B", "qwen3.5-2b": "Qwen/Qwen3.5-2B-Base", "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B-Base"}
DATA_FILES = ("train.jsonl", "val.jsonl", "val-masked.jsonl", "manifest.json")
PIP_BASE = {"packages": ["peft"], "timeout": 600}
PIP_FLA = {"packages": ["flash-linear-attention"], "timeout": 900}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(cmd: list, timeout: int = 90, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, timeout=timeout, **kw)


def build_jobs(plan: str, models: list[str], sizes: list[str], seed: int, fla: bool) -> dict:
    pip = [PIP_BASE] + ([PIP_FLA] if fla else [])
    runs = []
    if plan == "probe":
        for key in models:
            runs.append(dict(name=f"probe-{key}", args=["--model", MODELS[key], "--functions", "300", "--max-steps", "8",
                                                        "--eval-every", "1000", "--val-examples", "30", "--grad-ckpt", "--seed", str(seed)]))
    else:
        for key in models:
            for size in sizes:
                args = ["--model", MODELS[key], "--seed", str(seed), "--grad-ckpt"]
                if size != "all":
                    args += ["--functions", size]
                runs.append(dict(name=f"{key}-f{size}-s{seed}", args=args))
    return dict(pip=pip, runs=runs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--plan", choices=["probe", "curve"], required=True)
    p.add_argument("--gpu", choices=["A100", "L4", "T4", "G4"], default="L4")
    p.add_argument("--models", default="coder1.5b,qwen3.5-0.8b,qwen3.5-2b")
    p.add_argument("--sizes", default="1000,3000,all")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-fla", action="store_true", help="do not install flash-linear-attention (measures the slow fallback)")
    p.add_argument("--budget-min", type=int, default=60)
    p.add_argument("--package-only", action="store_true")
    a = p.parse_args()

    data = a.data.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    models = a.models.split(",")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown models {unknown}; choose from {sorted(MODELS)}")
    jobs = build_jobs(a.plan, models, a.sizes.split(","), a.seed, fla=not a.no_fla)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=1) + "\n")

    files = {"finetune.py": HERE / "finetune.py", "jobs.json": out / "jobs.json"}
    for name in DATA_FILES:
        files[f"data/{name}"] = data / name
    for name, path in files.items():
        if path.is_symlink() or not path.is_file():
            sys.exit(f"payload must be regular files: {name}")
    payload = dict(files={n: sha(pth) for n, pth in files.items()}, plan=a.plan, gpu=a.gpu, jobs=jobs, credentials_uploaded=False,
                   script_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip())
    (out / "payload.json").write_text(json.dumps(payload, indent=1) + "\n")
    archive = out / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
        tar.add(out / "payload.json", arcname="payload.json")
    print(f"archive {archive} ({archive.stat().st_size / 2**20:.1f} MiB), {len(jobs['runs'])} runs", flush=True)
    if a.package_only:
        print(json.dumps(jobs, indent=1))
        return

    session, started, allocated = a.session, time.monotonic(), False
    try:
        checked(["colab", "new", "-s", session, "--gpu", a.gpu], timeout=180)
        allocated = True
        checked(["colab", "upload", "-s", session, archive, "/content/ft-source.tar.gz"], timeout=300)
        checked(["colab", "upload", "-s", session, HERE / "colab_driver.py", "/content/ft-driver.py"], timeout=90)
        (out / "unpack.py").write_text(
            "import tarfile, pathlib\npathlib.Path('/content/ft').mkdir(exist_ok=True)\n"
            "tarfile.open('/content/ft-source.tar.gz').extractall('/content/ft', filter='data')\nprint('unpacked')\n")
        (out / "launch.py").write_text(
            "import subprocess, sys\nlog = open('/content/ft-driver.log', 'w')\n"
            "p = subprocess.Popen([sys.executable, '/content/ft-driver.py'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n"
            "print({'pid': p.pid})\n")
        (out / "poll.py").write_text(
            "import json, pathlib\ns = pathlib.Path('/content/ft/status.json')\n"
            "r = json.loads(s.read_text()) if s.exists() else {'state': 'starting'}\n"
            "for f in sorted(pathlib.Path('/content/ft/out').glob('*/log.jsonl')):\n"
            "    lines = f.read_text().splitlines()\n    r.setdefault('tails', {})[f.parent.name] = lines[-1][:200] if lines else ''\n"
            "print(json.dumps(r))\n")
        for step in ("unpack", "launch"):
            for attempt in range(4):
                try:
                    done = subprocess.run(["colab", "exec", "-s", session, "-f", str(out / f"{step}.py")], capture_output=True, text=True, timeout=90)
                    if done.returncode == 0:
                        print(done.stdout.strip(), flush=True)
                        break
                except subprocess.TimeoutExpired:
                    pass
                if attempt == 3:
                    raise RuntimeError(f"{step} did not succeed on the session")
                time.sleep(5)
        last = None
        while time.monotonic() - started < a.budget_min * 60:
            try:
                result = subprocess.run(["colab", "exec", "-s", session, "-f", str(out / "poll.py")], capture_output=True, text=True, timeout=60)
                text = result.stdout
                status = json.loads(text[text.index("{") : text.rindex("}") + 1])
            except (subprocess.TimeoutExpired, ValueError):
                time.sleep(10)
                continue
            key = json.dumps({k: v for k, v in status.items() if k != "updated"}, sort_keys=True)
            if key != last:
                print(json.dumps(status), flush=True)
                last = key
            if status.get("state") in ("complete", "failed"):
                checked(["colab", "download", "-s", session, "/content/ft/results.tar.gz", out / "results.tar.gz"], timeout=600)
                if status["state"] == "complete":
                    if sha(out / "results.tar.gz") != status["archive_sha256"]:
                        raise ValueError("result archive digest mismatch")
                    with tarfile.open(out / "results.tar.gz") as tar:
                        tar.extractall(out, filter="data")
                    return
                raise RuntimeError(status.get("error", "remote job failed"))
            time.sleep(30)
        raise TimeoutError(f"exceeded the {a.budget_min}-minute budget")
    finally:
        if allocated:
            for remote, name in (("/content/ft-driver.log", "driver.log"), ("/content/ft/status.json", "final-status.json")):
                try:
                    subprocess.run(["colab", "download", "-s", session, remote, str(out / name)], timeout=60, capture_output=True)
                except subprocess.TimeoutExpired:
                    print("diagnostic download timed out: " + name, flush=True)
            checked(["colab", "stop", "-s", session], timeout=90)


if __name__ == "__main__":
    main()
