"""Export Git-selected source files only; preserve working trees and simulator pins.

Run on the original server with a NEW destination root. This tool does not create
remote repositories, upload anything or inspect authentication-token files.
"""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


SOURCES = {
    "n15": Path("/home/yoon/vla_finetune/DEAS-Isaac-GR00T"),
    "n17": Path("/home/yoon/vla_finetune/Isaac-GR00T"),
}
EXCLUDED_DIRS = {"demo_data", "media", "datasets", "outputs", "wandb", ".venv", ".git"}
EXCLUDED_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".parquet", ".mp4", ".hdf5", ".h5", ".log", ".pem", ".key", ".whl"}
SECRET_PATTERNS = [
    re.compile(rb"\bhf_[A-Za-z0-9]{25,}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?:WANDB_API_KEY|HF_TOKEN|GITHUB_TOKEN|api_key|access_token)\s*[:=]\s*['\"][A-Za-z0-9_-]{25,}", re.I),
]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if any((destination / name).exists() for name in SOURCES):
        raise SystemExit("Refusing to overwrite existing n15/n17 snapshot directories")
    manifest = {"format_version": 1, "sources": {}, "excluded_dirs": sorted(EXCLUDED_DIRS),
                "excluded_suffixes": sorted(EXCLUDED_SUFFIXES)}
    submodules = configparser.ConfigParser()
    for name, source in SOURCES.items():
        target = destination / name
        target.mkdir(parents=True)
        record = {"head": git(source, "rev-parse", "HEAD").decode().strip(),
                  "remote": git(source, "remote", "get-url", "origin").decode().strip(),
                  "files": {}, "excluded": [], "submodules": []}
        gitlinks = {}
        for entry in git(source, "ls-files", "--stage", "-z").split(b"\0"):
            if entry.startswith(b"160000 "):
                metadata, path = entry.split(b"\t", 1)
                gitlinks[os.fsdecode(path)] = metadata.split()[1].decode()
        modules = configparser.ConfigParser()
        if gitlinks:
            modules.read(source / ".gitmodules")
        files = git(source, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        for raw in sorted(set(files.split(b"\0")) - {b""}):
            relative = Path(os.fsdecode(raw))
            if str(relative) in gitlinks:
                section = next(s for s in modules if modules[s].get("path") == str(relative))
                path = f"{name}/{relative}"
                url = modules[section]["url"]
                submodules[f'submodule "{path}"'] = {"path": path, "url": url}
                record["submodules"].append({"path": path, "commit": gitlinks[str(relative)], "url": url})
                continue
            if (relative.parts[0] in EXCLUDED_DIRS or "media" in relative.parts or relative.suffix.lower() in EXCLUDED_SUFFIXES
                    or (relative.name.startswith(".env") and relative.name != ".env.example")):
                record["excluded"].append(str(relative))
                continue
            origin = source / relative
            output = target / relative
            if not origin.exists() and not origin.is_symlink():
                record["excluded"].append(str(relative))
                continue
            if origin.is_symlink():
                link = os.readlink(origin)
                if Path(link).is_absolute() or not origin.resolve().is_relative_to(source.resolve()):
                    raise ValueError(f"Unsafe external symlink: {name}/{relative}")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.symlink_to(link)
                record["files"][str(relative)] = {"symlink": link}
                continue
            data = origin.read_bytes()
            if len(data) >= 50 * 1024 * 1024:
                raise ValueError(f"Unexpected large source file: {name}/{relative}")
            if any(pattern.search(data) for pattern in SECRET_PATTERNS):
                raise ValueError(f"Potential credential detected (content withheld): {name}/{relative}")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, output)
            digest = hashlib.sha256(data).hexdigest()
            if hashlib.sha256(output.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed while exporting: {name}/{relative}")
            record["files"][str(relative)] = {"sha256": digest, "bytes": len(data)}
        manifest["sources"][name] = record
        print(f"{name}: {len(record['files'])} source files, {len(record['excluded'])} excluded")
    with (destination / ".gitmodules").open("w") as handle:
        submodules.write(handle)
    (destination / "SOURCE_SNAPSHOT.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environments = destination / "environments"
    environments.mkdir(exist_ok=True)
    for filename in ("activate_gr00t.sh", "storage_env.sh"):
        shutil.copy2(Path("/home/yoon/vla_finetune") / filename, environments / filename)
    for name, env in (("n15", "deas-gr00t-n1.5"), ("n17", "gr00t-n1.7")):
        python = f"/raid/yoon/vla_finetune/envs/{env}/bin/python"
        packages = subprocess.check_output([python, "-c", "import importlib.metadata as m; print('\\n'.join(sorted({d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name']})))"])
        (environments / f"{name}-packages.txt").write_bytes(packages)


if __name__ == "__main__":
    main()
