"""Download the pinned LitheVoice models and llama.cpp runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("models.json")

IS_WINDOWS = os.name == "nt"
# Upstream ships Windows binaries as .zip and Linux binaries as .tar.gz, and
# only Windows gets a prebuilt CUDA archive.
ASSET_KEY = "assets" if IS_WINDOWS else "linux_assets"
SERVER_NAME = "llama-server.exe" if IS_WINDOWS else "llama-server"


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_name(entry: str | dict) -> str:
    return entry if isinstance(entry, str) else entry["filename"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, entry: dict, label: str) -> None:
    expected_size = entry.get("size")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"{label} has size {path.stat().st_size:,}; expected {expected_size:,}"
        )
    expected_hash = entry.get("sha256")
    if expected_hash:
        print(f"  verifying {label}...")
        actual = sha256(path)
        if actual.lower() != expected_hash.lower():
            raise RuntimeError(
                f"{label} failed SHA256 verification: {actual} != {expected_hash}"
            )


def download_http(url: str, target: Path, expected_hash: str, force: bool) -> Path:
    if target.exists() and not force and sha256(target).lower() == expected_hash.lower():
        print(f"  using {target.name}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    start = part.stat().st_size if part.exists() and not force else 0
    if force and part.exists():
        part.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "LitheVoice-setup"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    print(f"  downloading {target.name}" + (f" (resuming at {start:,} bytes)" if start else ""))
    with urllib.request.urlopen(request, timeout=60) as response:
        append = start > 0 and getattr(response, "status", None) == 206
        if not append:
            start = 0
        total_header = response.headers.get("Content-Length")
        total = start + int(total_header) if total_header else None
        mode = "ab" if append else "wb"
        completed = start
        last_report = time.monotonic()
        with part.open(mode) as handle:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
                completed += len(block)
                now = time.monotonic()
                if now - last_report >= 2:
                    if total:
                        print(f"    {completed / 1024**2:,.0f} / {total / 1024**2:,.0f} MiB")
                    else:
                        print(f"    {completed / 1024**2:,.0f} MiB")
                    last_report = now

    actual = sha256(part)
    if actual.lower() != expected_hash.lower():
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"{target.name} failed SHA256 verification: {actual} != {expected_hash}"
        )
    part.replace(target)
    return target


def _write_member(source, bin_dir: Path, name: str, executable: bool) -> None:
    target = bin_dir / name
    with target.open("wb") as dest:
        shutil.copyfileobj(source, dest, length=4 * 1024 * 1024)
    if executable and not IS_WINDOWS:
        target.chmod(target.stat().st_mode | 0o111)


def extract_flat(archive: Path, bin_dir: Path) -> None:
    """Flatten an archive's files into bin_dir.

    The upstream layouts nest binaries beside their shared libraries
    (``build/bin/`` on Linux, the archive root on Windows), so flattening keeps
    llama-server next to the libraries it resolves through its own directory.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name}")
    if archive.name.endswith((".tar.gz", ".tgz")):
        # The Linux archives resolve their SONAMEs through symlinks
        # (libllama-common.so.0 -> libllama-common.so.0.0.9867), so the links
        # have to be recreated or llama-server will not load. They are made in
        # a second pass because a link can precede its target in the archive.
        links: list[tuple[str, str]] = []
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                name = Path(member.name).name
                if not name:
                    continue
                if member.issym() or member.islnk():
                    links.append((name, Path(member.linkname).name))
                    continue
                if not member.isfile():
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    continue
                with source:
                    # Keep the upstream executable bit, and force it on for the
                    # binaries we are about to run.
                    _write_member(source, bin_dir, name,
                                  bool(member.mode & 0o111) or name.startswith("llama-"))
        for name, target in links:
            link_path = bin_dir / name
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(target)
        return
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if not name:
                continue
            with bundle.open(member) as source:
                _write_member(source, bin_dir, name, name.startswith("llama-"))


def download_llama(
    manifest: dict,
    backend: str,
    llama_dir: Path,
    downloads_dir: Path,
    force: bool,
) -> None:
    spec = manifest["llama_cpp"]
    table = spec.get(ASSET_KEY, {})
    assets = table.get(backend)
    if not assets:
        if not IS_WINDOWS and backend == "cuda":
            raise RuntimeError(
                "llama.cpp publishes no prebuilt Linux CUDA binary. Use "
                "--backend vulkan (works on NVIDIA), --backend cpu, or build "
                "llama.cpp with -DGGML_CUDA=ON and point LITHEVOICE_LLAMA_DIR "
                "at that build.")
        supported = ", ".join(sorted(table)) or "none"
        raise RuntimeError(
            f"unsupported llama.cpp backend on this platform: {backend} "
            f"(available: {supported})")

    marker = llama_dir / "backend.json"
    if marker.exists() and not force:
        installed = json.loads(marker.read_text(encoding="utf-8"))
        if installed.get("backend") != backend:
            raise RuntimeError(
                f"llama.cpp is configured for {installed.get('backend')}; rerun with --force "
                f"to replace it with {backend}"
            )

    downloads = downloads_dir / "llama.cpp"
    archives: list[Path] = []
    for asset in assets:
        url = (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{spec['tag']}/{asset['filename']}"
        )
        archives.append(
            download_http(url, downloads / asset["filename"], asset["sha256"], force)
        )

    bin_dir = llama_dir / "bin"
    if force and bin_dir.exists():
        shutil.rmtree(bin_dir)
    for archive in archives:
        extract_flat(archive, bin_dir)
    server = bin_dir / SERVER_NAME
    if not server.is_file():
        raise RuntimeError(f"{server} was not present in the downloaded archive")

    llama_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "tag": spec["tag"],
                "commit": spec["commit"],
                "backend": backend,
                "assets": [asset["filename"] for asset in assets],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_entries(root: Path, entries: list[str | dict], label: str) -> None:
    for entry in entries:
        if isinstance(entry, str):
            continue
        verify(root / entry["filename"], entry, f"{label}/{entry['filename']}")


def download_hugging_face_models(
    manifest: dict,
    models_dir: Path,
    include_gpu_stt: bool,
    force: bool,
    include_lite: bool = False,
) -> None:
    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download, snapshot_download

    llm = manifest["llm"]
    llm_dir = models_dir / "gemma_4_e2b"
    llm_dir.mkdir(parents=True, exist_ok=True)
    print("Gemma 4 E2B")
    for entry in llm["files"]:
        path = Path(
            hf_hub_download(
                repo_id=llm["repo_id"],
                filename=entry["filename"],
                revision=llm["revision"],
                local_dir=llm_dir,
                force_download=force,
            )
        )
        verify(path, entry, entry["filename"])

    parakeet = manifest["parakeet"]
    cpu_dir = models_dir / "parakeet-int8"
    print("Parakeet CPU int8")
    snapshot_download(
        repo_id=parakeet["repo_id"],
        revision=parakeet["revision"],
        local_dir=cpu_dir,
        allow_patterns=[file_name(item) for item in parakeet["cpu_files"]],
        force_download=force,
    )
    verify_entries(cpu_dir, parakeet["cpu_files"], "parakeet-int8")
    (cpu_dir / ".complete").write_text(parakeet["revision"] + "\n", encoding="ascii")

    if include_gpu_stt:
        gpu_dir = models_dir / "parakeet-fp32"
        print("Parakeet GPU fp32")
        snapshot_download(
            repo_id=parakeet["repo_id"],
            revision=parakeet["revision"],
            local_dir=gpu_dir,
            allow_patterns=[file_name(item) for item in parakeet["gpu_files"]],
            force_download=force,
        )
        verify_entries(gpu_dir, parakeet["gpu_files"], "parakeet-fp32")
        (gpu_dir / ".complete").write_text(parakeet["revision"] + "\n", encoding="ascii")

    kokoro = manifest["kokoro"]
    kokoro_patterns = [
        "config.json",
        kokoro["model"]["filename"],
        *[f"voices/{voice}.pt" for voice in kokoro["voices"]],
    ]
    print("Kokoro and curated voices")
    kokoro_snapshot = Path(
        snapshot_download(
            repo_id=kokoro["repo_id"],
            revision=kokoro["revision"],
            allow_patterns=kokoro_patterns,
            force_download=force,
        )
    )
    verify(kokoro_snapshot / kokoro["model"]["filename"], kokoro["model"], "Kokoro")

    if include_lite:
        lite = manifest["kokoro_onnx"]
        print("Kokoro ONNX (torch-free profile)")
        lite_path = Path(
            hf_hub_download(
                repo_id=lite["repo_id"],
                filename=lite["file"]["filename"],
                revision=lite["revision"],
                force_download=force,
            )
        )
        verify(lite_path, lite["file"], "Kokoro ONNX")

    speaker = manifest["speaker"]
    print("WeSpeaker speaker embeddings")
    speaker_path = Path(
        hf_hub_download(
            repo_id=speaker["repo_id"],
            filename=speaker["file"]["filename"],
            revision=speaker["revision"],
            force_download=force,
        )
    )
    verify(speaker_path, speaker["file"], "WeSpeaker")

    smart_turn = manifest["smart_turn"]
    print("Smart Turn v3.2")
    smart_path = Path(
        hf_hub_download(
            repo_id=smart_turn["repo_id"],
            filename=smart_turn["file"]["filename"],
            revision=smart_turn["revision"],
            force_download=force,
        )
    )
    verify(smart_path, smart_turn["file"], "Smart Turn")

    print("Silero VAD")
    from silero_vad import load_silero_vad

    load_silero_vad()


def print_plan(manifest: dict, backend: str, include_gpu_stt: bool) -> None:
    print("Pinned download plan:")
    print(f"  Gemma: {manifest['llm']['repo_id']}@{manifest['llm']['revision']}")
    print(f"  Parakeet: CPU int8" + (" + optional fp32" if include_gpu_stt else ""))
    print(f"  Kokoro: {len(manifest['kokoro']['voices'])} curated voices")
    print(f"  Smart Turn: {manifest['smart_turn']['file']['filename']}")
    print(f"  llama.cpp: {manifest['llama_cpp']['tag']} ({backend})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--llama-dir", type=Path, default=PROJECT_ROOT / "llama.cpp")
    parser.add_argument("--downloads-dir", type=Path, default=PROJECT_ROOT / "downloads")
    parser.add_argument("--backend", choices=["cpu", "cuda", "vulkan", "hip"], default="cpu")
    parser.add_argument("--include-gpu-stt", action="store_true")
    parser.add_argument("--include-lite", action="store_true",
                        help="also fetch the torch-free ONNX synthesiser")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if sys.maxsize <= 2**32:
        parser.error("64-bit Python is required")
    manifest = load_manifest()
    print_plan(manifest, args.backend, args.include_gpu_stt)
    if args.dry_run:
        return 0

    args.models_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_models:
        download_hugging_face_models(
            manifest, args.models_dir.resolve(), args.include_gpu_stt,
            args.force, include_lite=args.include_lite
        )
    if not args.skip_llama:
        print(f"llama.cpp {manifest['llama_cpp']['tag']} ({args.backend})")
        download_llama(
            manifest,
            args.backend,
            args.llama_dir.resolve(),
            args.downloads_dir.resolve(),
            args.force,
        )
    print("Downloads complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
