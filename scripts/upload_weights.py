"""Upload a trained checkpoint (adapter/ and head.pt) to a Hugging Face Hub model repo.

    python scripts/upload_weights.py --src ~/Documents/ActionRank/checkpoints/genrec/joint_s1 \
        --repo <user>/aethis-helm-weights --private

Needs a write token (`hf auth login` or HF_TOKEN). Prints the plan and asks for confirmation unless --yes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
HEAD_FILE = "head.pt"


def plan(src: Path) -> list[tuple[Path, str]]:
    """(local file, path in repo) pairs; raises if the checkpoint is incomplete."""
    files = [(src / "adapter" / name, f"adapter/{name}") for name in ADAPTER_FILES] + [(src / HEAD_FILE, HEAD_FILE)]
    missing = [str(local) for local, _ in files if not local.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint at {src} is incomplete; missing {missing}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="checkpoint dir containing adapter/ and head.pt")
    parser.add_argument("--repo", required=True, help="Hub repo id, e.g. user/aethis-helm-weights")
    parser.add_argument("--private", action="store_true", help="create the repo as private (if it does not exist)")
    parser.add_argument("--revision", default="main", help="branch to upload to")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    files = plan(args.src.expanduser())
    visibility = "private" if args.private else "PUBLIC"
    print(f"repo: {args.repo} ({visibility} if created), revision {args.revision}")
    for local, remote in files:
        print(f"  {local}  ->  {remote}  ({local.stat().st_size / 1e6:.1f} MB)")
    if not args.yes and input("upload? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in files]
    info = api.create_commit(args.repo, operations=operations, revision=args.revision,
                             commit_message="Add Aethis Helm span-head checkpoint (adapter + head.pt)")
    print(f"uploaded: {info.commit_url}")


if __name__ == "__main__":
    main()
