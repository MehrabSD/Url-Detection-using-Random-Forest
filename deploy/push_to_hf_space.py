"""Publish the demo page as a Hugging Face Space (free CPU hardware, Docker SDK).

One-time setup:
    .venv\\Scripts\\python.exe -m pip install -r deploy\\requirements-deploy.txt
    .venv\\Scripts\\hf.exe auth login        # paste a token with "write" permission
                                             # (huggingface.co -> Settings -> Access Tokens)
Then, and after every retrain:
    .venv\\Scripts\\python.exe deploy\\push_to_hf_space.py                 # <you>/malicious-url-detector
    .venv\\Scripts\\python.exe deploy\\push_to_hf_space.py --space you/name --private
    .venv\\Scripts\\python.exe deploy\\push_to_hf_space.py --dry-run       # list the files, upload nothing

What is uploaded is exactly what the Dockerfile copies (the model, the JSON
reports, src/ and web/) plus deploy/space_card.md as the Space's README.md.
Everything goes up in one commit, and files a previous upload left behind that
are no longer part of the app are deleted in the same commit. Hugging Face then
builds the Docker image and starts the server; the first build takes a few minutes.

Nothing about the model is changed here: the Space serves the same model file
and reports as the local demo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_CARD = Path(__file__).resolve().parent / "space_card.md"
DEFAULT_NAME = "malicious-url-detector"
KEEP_IN_SPACE = {".gitattributes"}  # created by Hugging Face; leave it alone

# path in the Space -> path in this project (must match the COPY lines of the Dockerfile)
FIXED_FILES = {
    "Dockerfile": "Dockerfile",
    "requirements-serve.txt": "requirements-serve.txt",
    "models/malicious_url_rf.joblib": "models/malicious_url_rf.joblib",
    "reports/metrics.json": "reports/metrics.json",
    "reports/external_eval.json": "reports/external_eval.json",
    "reports/red_team.json": "reports/red_team.json",
    "data/processed/prepare_report.json": "data/processed/prepare_report.json",
}


def app_files() -> dict[str, Path]:
    files = {space: ROOT / local for space, local in FIXED_FILES.items()}
    for folder, pattern in (("src", "*.py"), ("web", "**/*")):
        for path in sorted((ROOT / folder).glob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = path
    return files


def check_model_loads(files: dict[str, Path]) -> None:
    """Fail here, not in a remote build log, if the bundle cannot be served."""
    missing = [space for space, path in files.items() if not path.exists()]
    if missing:
        sys.exit("Missing files (run the pipeline first, see README.md): " + ", ".join(missing))
    sys.path.insert(0, str(ROOT / "src"))
    import joblib
    import sklearn

    pinned = next(line.split("==")[1].strip() for line in
                  (ROOT / "requirements-serve.txt").read_text().splitlines()
                  if line.startswith("scikit-learn=="))
    saved_with = joblib.load(ROOT / "models" / "malicious_url_rf.joblib").get("sklearn_version")
    if saved_with and saved_with != pinned:
        sys.exit(f"The model was saved with scikit-learn {saved_with} but requirements-serve.txt "
                 f"pins {pinned}. Make them match, or the Space cannot load the model.")
    if sklearn.__version__ != pinned:
        print(f"note: this venv has scikit-learn {sklearn.__version__}; the Space installs {pinned}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload the demo page to a Hugging Face Space.")
    ap.add_argument("--space", help=f"owner/name of the Space (default: <your user>/{DEFAULT_NAME})")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded and stop")
    args = ap.parse_args()

    files = app_files()
    check_model_loads(files)
    total_mb = sum(p.stat().st_size for p in files.values()) / 1e6
    if args.dry_run:
        for space, path in sorted(files.items()):
            print(f"  {path.stat().st_size / 1e6:8.2f} MB  {space}")
        print(f"  {len(files) + 1} files ({total_mb:.0f} MB) incl. README.md from {SPACE_CARD.name}")
        return 0

    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, get_token
    except ImportError:
        sys.exit("Install the uploader first: .venv\\Scripts\\python.exe -m pip install "
                 "-r deploy\\requirements-deploy.txt")
    if not get_token():
        sys.exit("No Hugging Face token found. Run `.venv\\Scripts\\hf.exe auth login` and paste a "
                 "token with write permission (huggingface.co -> Settings -> Access Tokens), "
                 "or set the HF_TOKEN environment variable.")

    api = HfApi()
    space_id = args.space or f"{api.whoami()['name']}/{DEFAULT_NAME}"
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=args.private,
                    exist_ok=True)

    operations = [CommitOperationAdd("README.md", SPACE_CARD)]
    operations += [CommitOperationAdd(space, path) for space, path in sorted(files.items())]
    current = set(api.list_repo_files(space_id, repo_type="space"))
    stale = current - set(files) - {"README.md"} - KEEP_IN_SPACE
    operations += [CommitOperationDelete(path) for path in sorted(stale)]

    print(f"Uploading {len(files) + 1} files ({total_mb:.0f} MB) to {space_id}"
          + (f", removing {len(stale)} old ones" if stale else "") + " ...", flush=True)
    api.create_commit(space_id, operations, repo_type="space",
                      commit_message="Update the demo page from the project folder")

    print(f"\nDone. Space:  https://huggingface.co/spaces/{space_id}")
    print("Hugging Face now builds the image and starts the server (a few minutes the first")
    print("time); the Space page shows the build log and then the demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
