import hashlib
import random
from pathlib import Path
from typing import Any

import httpx

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.infra.snapshots import fetch_snapshot
from auditor.infra.snapshots.data import fetch_data_files
from auditor.schemas import RepositoryRef

SHA = "c" * 40


class Hosts:
    def __init__(self) -> None:
        self.routes: dict[str, httpx.Response] = {}
        self.requested: list[str] = []

    def json(self, url: str, body: Any) -> None:
        self.routes[url] = httpx.Response(200, json=body)

    def bytes(self, url: str, body: bytes) -> None:
        self.routes[url] = httpx.Response(200, content=body)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(str(request.url))
        return self.routes.get(str(request.url), httpx.Response(404))


async def _no_sleep(seconds: float) -> None:
    return None


def settings_for(tmp_path: Path, **values: object) -> Settings:
    return Settings.model_validate(
        {
            "cache_dir": tmp_path / "cache",
            "scratch_dir": tmp_path / "scratch",
            "http_max_attempts": 1,
            **values,
        }
    )


def client(hosts: Hosts, settings: Settings) -> HttpClient:
    return HttpClient(
        settings,
        transport=httpx.MockTransport(hosts),
        sleep=_no_sleep,
        rng=random.Random(0),
        use_default_cache=False,
    )


def github(hosts: Hosts, files: dict[str, bytes]) -> None:
    api = "https://api.github.com/repos/o/r"
    hosts.json(f"{api}/commits/main", {"sha": SHA})
    tree = [{"path": p, "type": "blob", "mode": "100644", "size": len(b)} for p, b in files.items()]
    hosts.json(f"{api}/git/trees/{SHA}?recursive=1", {"tree": tree, "truncated": False})
    for path, body in files.items():
        hosts.bytes(f"https://raw.githubusercontent.com/o/r/{SHA}/{path}", body)


TRAIN = b"text,label\na,0\nb,1\n"
TEST = b"text,label\na,0\n"


async def test_data_files_are_listed_but_not_in_the_snapshot(tmp_path: Path) -> None:
    hosts = Hosts()
    github(hosts, {"train.py": b"x=1", "data/train.csv": TRAIN, "data/test.csv": TEST})
    settings = settings_for(tmp_path)
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/o/r", commit="main"), http, settings
        )
    assert [f.path for f in snapshot.manifest.files] == ["train.py"]
    assert [(d.path, d.size) for d in snapshot.manifest.data_files] == [
        ("data/test.csv", len(TEST)),
        ("data/train.csv", len(TRAIN)),
    ]
    assert not any("data/" in url for url in hosts.requested)


async def test_data_files_are_downloaded_into_content_addressed_blobs(tmp_path: Path) -> None:
    hosts = Hosts()
    github(hosts, {"data/train.csv": TRAIN, "data/test.csv": TEST})
    settings = settings_for(tmp_path)
    ref = RepositoryRef(url="https://github.com/o/r", commit="main")
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(ref, http, settings)
        fetched = await fetch_data_files(
            snapshot, ["data/train.csv", "data/test.csv"], http, settings
        )
    train = fetched.blobs["data/train.csv"]
    assert train.local.read_bytes() == TRAIN
    assert train.sha256 == hashlib.sha256(TRAIN).hexdigest()
    assert train.local.is_relative_to(tmp_path / "scratch" / "data-blobs")
    assert fetched.skipped == {}


async def test_data_limits_skip_large_files(tmp_path: Path) -> None:
    hosts = Hosts()
    github(hosts, {"data/train.csv": TRAIN, "data/test.csv": TEST})
    settings = settings_for(tmp_path, data_analysis_max_file_bytes=len(TEST))
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/o/r", commit="main"), http, settings
        )
        fetched = await fetch_data_files(
            snapshot, ["data/train.csv", "data/test.csv"], http, settings
        )
    assert set(fetched.blobs) == {"data/test.csv"}
    assert fetched.skipped == {"data/train.csv": "larger than the data file limit"}


async def test_git_lfs_pointers_are_not_data(tmp_path: Path) -> None:
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 999999\n"
    hosts = Hosts()
    github(hosts, {"data/train.csv": pointer})
    settings = settings_for(tmp_path)
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/o/r", commit="main"), http, settings
        )
        fetched = await fetch_data_files(snapshot, ["data/train.csv"], http, settings)
    assert fetched.blobs == {}
    assert "Git LFS" in fetched.skipped["data/train.csv"]


async def test_hugging_face_lfs_files_are_verified_and_reused(tmp_path: Path) -> None:
    body = b"text,label\nhello,1\n"
    oid = hashlib.sha256(body).hexdigest()
    api = "https://huggingface.co/api/datasets/org/data"
    hosts = Hosts()
    hosts.json(f"{api}/revision/main", {"sha": SHA})
    hosts.json(
        f"{api}/tree/{SHA}?recursive=true",
        [{"type": "file", "path": "train.csv", "size": len(body), "lfs": {"oid": oid}}],
    )
    file_url = f"https://huggingface.co/datasets/org/data/resolve/{SHA}/train.csv"
    hosts.bytes(file_url, body)
    settings = settings_for(tmp_path)
    ref = RepositoryRef(url="https://huggingface.co/datasets/org/data")
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(ref, http, settings)
        (entry,) = snapshot.manifest.data_files
        assert entry.sha256 == oid
        first = await fetch_data_files(snapshot, ["train.csv"], http, settings)
        downloads = hosts.requested.count(file_url)
        second = await fetch_data_files(snapshot, ["train.csv"], http, settings)
    assert first.blobs["train.csv"].sha256 == oid
    assert second.blobs["train.csv"].local == first.blobs["train.csv"].local
    assert hosts.requested.count(file_url) == downloads == 1


async def test_local_data_is_read_in_place(tmp_path: Path) -> None:
    repo = tmp_path / "repo" / "data"
    repo.mkdir(parents=True)
    (repo / "train.csv").write_bytes(TRAIN)
    settings = settings_for(tmp_path)
    async with client(Hosts(), settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url=(tmp_path / "repo").resolve().as_uri()), http, settings
        )
        fetched = await fetch_data_files(snapshot, ["data/train.csv"], http, settings)
    assert fetched.blobs["data/train.csv"].local == (repo / "train.csv").resolve()
