import asyncio
import hashlib
import io
import random
import zipfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.infra.snapshots import (
    DownloadLimits,
    SnapshotUnavailableError,
    fetch_snapshot,
)
from auditor.schemas import RepositoryRef

SHA = "a" * 40
Route = httpx.Response | Callable[[httpx.Request], httpx.Response]


class FakeHosts:
    """Answers exact URLs; anything else is a 404. Records every URL requested."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.requested: list[str] = []

    def json(self, url: str, body: Any, headers: dict[str, str] | None = None) -> None:
        self.routes[url] = httpx.Response(200, json=body, headers=headers)

    def bytes(self, url: str, body: bytes) -> None:
        self.routes[url] = httpx.Response(200, content=body)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requested.append(url)
        route = self.routes.get(url)
        if route is None:
            return httpx.Response(404, json={"message": "Not Found"})
        return route(request) if callable(route) else route


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


def client(hosts: FakeHosts, settings: Settings) -> HttpClient:
    return HttpClient(
        settings,
        transport=httpx.MockTransport(hosts),
        sleep=_no_sleep,
        rng=random.Random(0),
        use_default_cache=False,
    )


def github(hosts: FakeHosts, tree: list[dict[str, Any]], truncated: bool = False) -> None:
    api = "https://api.github.com/repos/owner/repo"
    hosts.json(f"{api}/commits/main", {"sha": SHA})
    hosts.json(f"{api}/git/trees/{SHA}?recursive=1", {"tree": tree, "truncated": truncated})


def blob(path: str, size: int, mode: str = "100644") -> dict[str, Any]:
    return {"path": path, "type": "blob", "mode": mode, "size": size}


def raw(hosts: FakeHosts, path: str, body: bytes) -> None:
    hosts.bytes(f"https://raw.githubusercontent.com/owner/repo/{SHA}/{path}", body)


async def test_github_snapshot_has_only_core_files(tmp_path: Path) -> None:
    hosts = FakeHosts()
    github(
        hosts,
        [
            blob("train.py", 20),
            {"path": "configs", "type": "tree", "mode": "040000"},
            blob("configs/base.yaml", 9),
            blob("README.md", 6),
            blob("data/train.csv", 5_000),
            blob("big.json", 2_000_000),
            blob("link.py", 8, mode="120000"),
            {"path": "third_party/lib", "type": "commit", "mode": "160000"},
            blob(".github/workflows/ci.yml", 10),
        ],
    )
    raw(hosts, "train.py", b"lr = 3e-4\nbatch = 32\n")
    raw(hosts, "configs/base.yaml", b"lr: 3e-4\n")
    raw(hosts, "README.md", b"# Repo")
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/owner/repo", commit="main"), http, settings
        )

    manifest = snapshot.manifest
    assert manifest.revision == SHA
    assert manifest.requested_ref == "main"
    assert snapshot.source_id == f"https://github.com/owner/repo@{SHA}"
    assert [f.path for f in manifest.files] == ["train.py", "configs/base.yaml", "README.md"]
    assert manifest.files[0].sha256 == hashlib.sha256(b"lr = 3e-4\nbatch = 32\n").hexdigest()
    assert snapshot.read_text("train.py") == "lr = 3e-4\nbatch = 32\n"
    assert snapshot.root.is_relative_to(tmp_path / "scratch")
    assert {s.path: s.reason for s in manifest.skipped} == {
        "data/train.csv": "not a core file type",
        "big.json": "larger than the per-file limit",
        "link.py": "symbolic link",
        "third_party/lib": "git submodule",
        ".github/workflows/ci.yml": "excluded directory",
    }
    assert manifest.incomplete == ()
    assert not any("data/train.csv" in url for url in hosts.requested)
    assert (snapshot.root.parent / "snapshot.json").is_file()


async def test_a_pinned_snapshot_is_reused_without_requests(tmp_path: Path) -> None:
    hosts = FakeHosts()
    github(hosts, [blob("train.py", 3)])
    raw(hosts, "train.py", b"x=1")
    settings = settings_for(tmp_path)
    ref = RepositoryRef(url="https://github.com/owner/repo", commit=SHA)

    async with client(hosts, settings) as http:
        first = await fetch_snapshot(ref, http, settings)
        count = len(hosts.requested)
        second = await fetch_snapshot(ref, http, settings)

    assert len(hosts.requested) == count
    assert second.manifest == first.manifest


async def test_files_listed_as_empty_are_not_downloaded(tmp_path: Path) -> None:
    # raw.githubusercontent.com sometimes answers 501 for an empty file.
    hosts = FakeHosts()
    github(hosts, [blob("train.py", 3), blob("pkg/__init__.py", 0)])
    raw(hosts, "train.py", b"x=1")
    hosts.routes[f"https://raw.githubusercontent.com/owner/repo/{SHA}/pkg/__init__.py"] = (
        httpx.Response(501)
    )
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/owner/repo", commit="main"), http, settings
        )

    assert [f.path for f in snapshot.manifest.files] == ["train.py", "pkg/__init__.py"]
    assert snapshot.read_text("pkg/__init__.py") == ""
    assert not any("__init__" in url for url in hosts.requested)


async def test_download_limits_mark_the_snapshot_incomplete(tmp_path: Path) -> None:
    hosts = FakeHosts()
    github(hosts, [blob("train.py", 20), blob("eval.py", 20)])
    raw(hosts, "eval.py", b"e" * 20)
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/owner/repo", commit="main"),
            http,
            settings,
            limits=DownloadLimits(max_total_bytes=30),
        )

    assert [f.path for f in snapshot.manifest.files] == ["eval.py"]
    assert snapshot.manifest.incomplete == ("total download limit reached",)


async def test_truncated_listing_marks_the_snapshot_incomplete(tmp_path: Path) -> None:
    hosts = FakeHosts()
    github(hosts, [], truncated=True)
    settings = settings_for(tmp_path)
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/owner/repo", commit="main"), http, settings
        )
    assert snapshot.manifest.incomplete == ("the host truncated the file listing",)


async def test_missing_repository_is_unavailable_not_an_error(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    async with client(FakeHosts(), settings) as http:
        with pytest.raises(SnapshotUnavailableError, match="HTTP 404"):
            await fetch_snapshot(
                RepositoryRef(url="https://github.com/owner/gone", commit="main"), http, settings
            )


async def test_a_rate_limited_repository_is_not_asked_for_again(tmp_path: Path) -> None:
    from auditor.infra.http import HttpStatusError

    settings = settings_for(tmp_path)
    hosts = FakeHosts()
    hosts.routes["https://api.github.com/repos/owner/repo/commits/main"] = httpx.Response(
        403, json={"message": "API rate limit exceeded"}, headers={"x-ratelimit-remaining": "0"}
    )
    ref = RepositoryRef(url="https://github.com/owner/repo", commit="main")
    async with client(hosts, settings) as http:
        # Ten code tasks of one paper used to send ten identical requests.
        for _ in range(3):
            with pytest.raises(HttpStatusError, match="HTTP 403"):
                await fetch_snapshot(ref, http, settings)
    assert len(hosts.requested) == 1


async def test_a_server_error_is_asked_for_again(tmp_path: Path) -> None:
    from auditor.infra.http import HttpStatusError

    settings = settings_for(tmp_path)
    hosts = FakeHosts()
    hosts.routes["https://api.github.com/repos/owner/repo/commits/main"] = httpx.Response(502)
    ref = RepositoryRef(url="https://github.com/owner/repo", commit="main")
    async with client(hosts, settings) as http:
        for _ in range(2):
            with pytest.raises(HttpStatusError, match="HTTP 502"):
                await fetch_snapshot(ref, http, settings)
    assert len(hosts.requested) == 2


async def test_github_token_is_sent_only_to_the_api(tmp_path: Path) -> None:
    hosts = FakeHosts()
    seen: dict[str, str | None] = {}

    def commits(request: httpx.Request) -> httpx.Response:
        seen["api"] = request.headers.get("authorization")
        return httpx.Response(200, json={"sha": SHA})

    def file(request: httpx.Request) -> httpx.Response:
        seen["raw"] = request.headers.get("authorization")
        return httpx.Response(200, content=b"x=1")

    github(hosts, [blob("a.py", 3)])
    hosts.routes["https://api.github.com/repos/owner/repo/commits/main"] = commits
    hosts.routes[f"https://raw.githubusercontent.com/owner/repo/{SHA}/a.py"] = file
    settings = settings_for(tmp_path, github_token="ghp_secret")
    async with client(hosts, settings) as http:
        await fetch_snapshot(
            RepositoryRef(url="https://github.com/owner/repo", commit="main"), http, settings
        )
    assert seen == {"api": "Bearer ghp_secret", "raw": None}


async def test_gitlab_files_of_unknown_size_stop_at_the_limit(tmp_path: Path) -> None:
    hosts = FakeHosts()
    api = "https://gitlab.com/api/v4/projects/group%2Fsub%2Fproj"
    hosts.json(f"{api}/repository/commits/v1.0", {"id": SHA})
    tree = f"{api}/repository/tree?ref={SHA}&recursive=true&per_page=100"
    hosts.json(
        f"{tree}&page=1",
        [
            {"path": "train.py", "type": "blob", "mode": "100644"},
            {"path": "link.py", "type": "blob", "mode": "120000"},
        ],
        headers={"x-next-page": "2"},
    )
    hosts.json(
        f"{tree}&page=2",
        [{"path": "huge.json", "type": "blob", "mode": "100644"}],
        headers={"x-next-page": ""},
    )
    hosts.bytes(f"{api}/repository/files/train.py/raw?ref={SHA}", b"lr=1")

    async def endless() -> AsyncIterator[bytes]:
        for _ in range(10_000):
            yield b"{" * 100

    hosts.routes[f"{api}/repository/files/huge.json/raw?ref={SHA}"] = lambda request: (
        httpx.Response(200, content=endless())
    )
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://gitlab.com/group/sub/proj", commit="v1.0"),
            http,
            settings,
            limits=DownloadLimits(max_file_bytes=1_000),
        )

    assert snapshot.manifest.revision == SHA
    assert [f.path for f in snapshot.manifest.files] == ["train.py"]
    assert {s.path: s.reason for s in snapshot.manifest.skipped} == {
        "link.py": "symbolic link",
        "huge.json": "larger than the per-file limit",
    }


async def test_hugging_face_skips_large_file_storage_and_follows_pages(tmp_path: Path) -> None:
    hosts = FakeHosts()
    api = "https://huggingface.co/api/models/org/model"
    hosts.json(f"{api}/revision/main", {"sha": SHA})
    page_two = f"{api}/tree/{SHA}?recursive=true&cursor=abc"
    hosts.json(
        f"{api}/tree/{SHA}?recursive=true",
        [
            {"type": "file", "path": "config.json", "size": 30},
            {"type": "file", "path": "model.safetensors", "size": 10, "lfs": {"size": 10**9}},
            {"type": "directory", "path": "src"},
        ],
        headers={"link": f'<{page_two}>; rel="next"'},
    )
    hosts.json(page_two, [{"type": "file", "path": "src/modeling.py", "size": 5}])
    hosts.bytes(
        f"https://huggingface.co/org/model/resolve/{SHA}/config.json", b'{"hidden_size": 768}'
    )
    hosts.bytes(f"https://huggingface.co/org/model/resolve/{SHA}/src/modeling.py", b"x = 1")
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://huggingface.co/org/model"), http, settings
        )

    assert snapshot.manifest.revision == SHA
    assert [f.path for f in snapshot.manifest.files] == ["config.json", "src/modeling.py"]
    assert {s.path: s.reason for s in snapshot.manifest.skipped} == {
        "model.safetensors": "stored in large file storage",
    }


async def test_hugging_face_dataset_keeps_split_sizes(tmp_path: Path) -> None:
    hosts = FakeHosts()
    api = "https://huggingface.co/api/datasets/org/data"
    hosts.json(f"{api}/revision/main", {"sha": SHA})
    hosts.json(f"{api}/tree/{SHA}?recursive=true", [])
    size = {
        "size": {
            "splits": [{"split": "train", "num_rows": 800}, {"split": "test", "num_rows": 200}]
        }
    }
    hosts.json("https://datasets-server.huggingface.co/size?dataset=org%2Fdata", size)
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://huggingface.co/datasets/org/data"), http, settings
        )

    assert snapshot.manifest.metadata["dataset_size"] == size


async def test_kaggle_dataset_lists_data_files_but_downloads_only_core_ones(
    tmp_path: Path,
) -> None:
    hosts = FakeHosts()
    api = "https://www.kaggle.com/api/v1/datasets"
    hosts.json(
        f"{api}/view/owner/data",
        {"title": "Data", "currentVersionNumber": 3, "licenseName": "CC0"},
    )
    hosts.json(
        f"{api}/list/owner/data?datasetVersionNumber=3",
        {
            "datasetFiles": [
                {"name": "train.csv", "totalBytes": 5_000, "columns": [{"name": "label"}]},
                {"name": "dataset-metadata.json", "totalBytes": 12, "columns": []},
            ]
        },
    )
    signed = "https://storage.googleapis.com/kaggle-data-sets/1/meta.json?sig=x"
    hosts.routes[f"{api}/download/owner/data/dataset-metadata.json?datasetVersionNumber=3"] = (
        httpx.Response(302, headers={"location": signed})
    )
    hosts.bytes(signed, b'{"id": "x"}\n')
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://www.kaggle.com/datasets/owner/data"), http, settings
        )

    assert snapshot.manifest.revision == "3"
    assert [f.path for f in snapshot.manifest.files] == ["dataset-metadata.json"]
    assert snapshot.manifest.metadata["files"] == [
        {"name": "train.csv", "bytes": 5_000, "columns": ["label"]},
        {"name": "dataset-metadata.json", "bytes": 12, "columns": []},
    ]
    assert snapshot.manifest.metadata["title"] == "Data"


def anonymous(hosts: FakeHosts, options: dict[str, Any]) -> str:
    api = "https://anonymous.4open.science/api/repo/abc"
    hosts.json(f"{api}/options", options)
    hosts.json(
        f"{api}/files/",
        [
            {"name": "main.py", "path": "", "size": 5, "sha": "aa"},
            {"name": "cfg", "path": ""},
            {"name": "node_modules", "path": ""},
        ],
    )
    hosts.json(
        f"{api}/files/?path=cfg", [{"name": "a.yaml", "path": "cfg", "size": 4, "sha": "bb"}]
    )
    hosts.bytes(f"{api}/file/main.py", b"x=1\n")
    hosts.bytes(f"{api}/file/cfg/a.yaml", b"a: 1")
    return api


async def test_anonymous_github_is_crawled_and_pinned_by_its_listing(tmp_path: Path) -> None:
    hosts = FakeHosts()
    api = anonymous(hosts, {"download": True, "truncatedFolders": [], "anonymizedAt": "2026-05-07"})
    settings = settings_for(tmp_path)
    ref = RepositoryRef(url="https://anonymous.4open.science/r/abc")

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(ref, http, settings)
        downloads = sum("/file/" in url for url in hosts.requested)
        again = await fetch_snapshot(ref, http, settings)

    assert [f.path for f in snapshot.manifest.files] == ["main.py", "cfg/a.yaml"]
    assert snapshot.manifest.revision.startswith("listing-")
    assert f"{api}/files/?path=node_modules" not in hosts.requested
    assert snapshot.manifest.metadata["anonymizedAt"] == "2026-05-07"
    # The listing is fetched again (the host can change), but unchanged files are not.
    assert sum("/file/" in url for url in hosts.requested) == downloads
    assert again.manifest.revision == snapshot.manifest.revision


async def test_anonymous_github_reports_truncated_folders(tmp_path: Path) -> None:
    hosts = FakeHosts()
    anonymous(hosts, {"download": True, "truncatedFolders": ["data"]})
    settings = settings_for(tmp_path)
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://anonymous.4open.science/r/abc"), http, settings
        )
    assert snapshot.manifest.incomplete == ("the host truncated these folders: data",)


async def test_anonymous_github_respects_disabled_downloads(tmp_path: Path) -> None:
    hosts = FakeHosts()
    anonymous(hosts, {"download": False})
    settings = settings_for(tmp_path)
    async with client(hosts, settings) as http:
        with pytest.raises(SnapshotUnavailableError, match="disabled downloads"):
            await fetch_snapshot(
                RepositoryRef(url="https://anonymous.4open.science/r/abc"), http, settings
            )


def make_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return buffer.getvalue()


async def test_zenodo_archive_is_unpacked_to_core_files(tmp_path: Path) -> None:
    hosts = FakeHosts()
    archive = make_zip(
        {
            "owner-repo-abc/train.py": b"lr = 0.1",
            "owner-repo-abc/configs/a.yaml": b"a: 1",
            "owner-repo-abc/data/train.csv": b"1,2",
            "owner-repo-abc/../../evil.py": b"boom",
        }
    )
    content = "https://zenodo.org/api/records/42/files/repo-v1.zip/content"
    hosts.json(
        "https://zenodo.org/api/records/42",
        {
            "id": 42,
            "doi": "10.5281/zenodo.42",
            "metadata": {"title": "repo v1", "version": "v1"},
            "files": [
                {
                    "key": "repo-v1.zip",
                    "size": len(archive),
                    "checksum": f"md5:{hashlib.md5(archive).hexdigest()}",
                    "links": {"self": content},
                }
            ],
        },
    )
    hosts.bytes(content, archive)
    settings = settings_for(tmp_path)

    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://zenodo.org/records/42"), http, settings
        )

    assert snapshot.manifest.revision == "42"
    assert [f.path for f in snapshot.manifest.files] == ["train.py", "configs/a.yaml"]
    assert {s.path: s.reason for s in snapshot.manifest.skipped} == {
        "data/train.csv": "not a core file type",
        "../../evil.py": "unsafe path",
    }
    assert snapshot.manifest.metadata["doi"] == "10.5281/zenodo.42"


async def test_zenodo_archive_with_a_wrong_checksum_is_refused(tmp_path: Path) -> None:
    hosts = FakeHosts()
    archive = make_zip({"r/train.py": b"x"})
    content = "https://zenodo.org/api/records/42/files/r.zip/content"
    hosts.json(
        "https://zenodo.org/api/records/42",
        {
            "id": 42,
            "files": [
                {
                    "key": "r.zip",
                    "size": len(archive),
                    "checksum": "md5:0",
                    "links": {"self": content},
                }
            ],
        },
    )
    hosts.bytes(content, archive)
    settings = settings_for(tmp_path)
    async with client(hosts, settings) as http:
        with pytest.raises(SnapshotUnavailableError, match="checksum"):
            await fetch_snapshot(RepositoryRef(url="https://zenodo.org/records/42"), http, settings)


async def test_local_directory_is_copied_and_pinned_by_content(tmp_path: Path) -> None:
    repo = tmp_path / "patched"
    (repo / "configs").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "train.py").write_text("lr = 3e-4\n", encoding="utf-8")
    (repo / "configs" / "a.yaml").write_text("a: 1\n", encoding="utf-8")
    (repo / "data.csv").write_text("1,2\n", encoding="utf-8")
    (repo / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    try:
        (repo / "link.py").symlink_to(repo / "train.py")
        has_link = True
    except OSError:
        has_link = False
    settings = settings_for(tmp_path)
    ref = RepositoryRef(url=repo.resolve().as_uri())

    async with client(FakeHosts(), settings) as http:
        first = await fetch_snapshot(ref, http, settings)
        (repo / "train.py").write_text("lr = 1e-3\n", encoding="utf-8")
        second = await fetch_snapshot(ref, http, settings)

    assert sorted(f.path for f in first.manifest.files) == ["configs/a.yaml", "train.py"]
    reasons = {s.path: s.reason for s in first.manifest.skipped}
    assert reasons["data.csv"] == "not a core file type"
    assert ".git/config" not in {f.path for f in first.manifest.files}
    if has_link:
        assert reasons["link.py"] == "symbolic link"
    assert first.manifest.revision.startswith("content-")
    assert second.manifest.revision != first.manifest.revision
    assert first.read_text("train.py") == "lr = 3e-4\n"
    assert second.read_text("train.py") == "lr = 1e-3\n"


async def test_snapshot_paths_outside_the_manifest_are_refused(tmp_path: Path) -> None:
    hosts = FakeHosts()
    github(hosts, [blob("a.py", 3)])
    raw(hosts, "a.py", b"x=1")
    settings = settings_for(tmp_path)
    async with client(hosts, settings) as http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://github.com/owner/repo", commit="main"), http, settings
        )
    with pytest.raises(KeyError):
        snapshot.path("../snapshot.json")
    with pytest.raises(KeyError):
        snapshot.path("missing.py")


async def test_files_of_unknown_size_download_in_parallel_within_the_budget(
    tmp_path: Path,
) -> None:
    api = "https://gitlab.com/api/v4/projects/group%2Fproj"
    names = [f"f{i}.py" for i in range(6)]
    in_flight = 0
    most = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, most
        url = str(request.url)
        if url == f"{api}/repository/commits/main":
            return httpx.Response(200, json={"id": SHA})
        if "/repository/tree" in url:
            tree = [{"path": n, "type": "blob", "mode": "100644"} for n in names]
            return httpx.Response(200, json=tree, headers={"x-next-page": ""})
        in_flight += 1
        most = max(most, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, content=b"x" * 40)

    settings = settings_for(tmp_path)
    http = HttpClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
        rng=random.Random(0),
        use_default_cache=False,
    )
    async with http:
        snapshot = await fetch_snapshot(
            RepositoryRef(url="https://gitlab.com/group/proj", commit="main"),
            http,
            settings,
            limits=DownloadLimits(max_total_bytes=100),
        )

    assert most > 1
    # The budget is spent in priority order, whatever order the downloads finish in.
    assert [f.path for f in snapshot.manifest.files] == ["f0.py", "f1.py"]
    assert snapshot.manifest.incomplete == ("total download limit reached",)


async def test_an_unchanged_local_folder_is_not_copied_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auditor.infra.snapshots import hosts

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("lr = 3e-4\n", encoding="utf-8")
    calls = []
    original = hosts.LocalFolder.listing

    async def counted(self: hosts.LocalFolder, revision: str | None) -> hosts.Listing:
        calls.append(revision)
        return await original(self, revision)

    monkeypatch.setattr(hosts.LocalFolder, "listing", counted)
    settings = settings_for(tmp_path)
    ref = RepositoryRef(url=repo.resolve().as_uri())
    async with client(FakeHosts(), settings) as http:
        first = await fetch_snapshot(ref, http, settings)
        second = await fetch_snapshot(ref, http, settings)
        (repo / "train.py").write_text("lr = 1e-3\n", encoding="utf-8")
        third = await fetch_snapshot(ref, http, settings)
    assert second.manifest == first.manifest
    assert len(calls) == 2
    assert third.manifest.revision != first.manifest.revision
