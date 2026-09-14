from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "t8star/Auk-Comfy"
VARIANT_DIRECTORIES = {
    "flash": ["AuK-Flash", "Qwen2.5-Omni-3B"],
    "base": ["AuK", "Qwen2.5-Omni-3B"],
    "all": ["AuK-Flash", "AuK", "Qwen2.5-Omni-3B"],
}


def default_model_root() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "auk"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(model_root: Path, directories: list[str], verify_hashes: bool) -> None:
    manifest = json.loads(Path(__file__).with_name("MODEL_MANIFEST.json").read_text(encoding="utf-8"))["models"]
    problems: list[str] = []
    for directory in directories:
        for filename, details in manifest[directory]["files"].items():
            path = model_root / directory / filename
            if not path.is_file():
                problems.append(f"missing: {path}")
            elif path.stat().st_size != int(details["size"]):
                problems.append(f"wrong size: {path}")
            elif verify_hashes and sha256(path) != details["sha256"]:
                problems.append(f"wrong SHA-256: {path}")
    if problems:
        raise RuntimeError("Model verification failed:\n" + "\n".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download native AuK ComfyUI models")
    parser.add_argument("--variant", choices=VARIANT_DIRECTORIES, default="flash")
    parser.add_argument("--model-root", type=Path, default=default_model_root())
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument("--verify-sha256", dest="verify_sha256", action="store_true")
    verification.add_argument("--skip-sha256", dest="verify_sha256", action="store_false")
    parser.set_defaults(verify_sha256=True)
    args = parser.parse_args()
    model_root = args.model_root.expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    directories = VARIANT_DIRECTORIES[args.variant]
    patterns = [pattern for directory in directories for pattern in (f"{directory}/*", f"{directory}/assets/*")]
    manifest = json.loads(Path(__file__).with_name("MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    snapshot_download(
        repo_id=REPOSITORY,
        revision=manifest["repository_revision"],
        local_dir=model_root,
        allow_patterns=patterns,
    )
    verify(model_root, directories, args.verify_sha256)
    print(f"AuK models are ready in {model_root}")


if __name__ == "__main__":
    main()
