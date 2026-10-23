"""Path containment for the screenshot server's file_path routes.

`/info`, `/page` and `/process` take a caller-supplied `file_path`. These tests
pin the containment rules from the caller's side (Flask test client, no socket
bound): files under an allowed root work, everything that tries to leave one is
refused, and `/info` cannot be used to tell whether an out-of-bounds path exists.

Only local temp files are touched -- no model is loaded, since every rejection
happens before any rendering or OCR.
"""

import os

import pytest
from PIL import Image

screenshot_app = pytest.importorskip("surya.scripts.screenshot_app")

FILE_PATH_ROUTES = ["/info", "/page", "/process"]


def _write_png(path, size=(32, 16)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


@pytest.fixture()
def upload_dir(tmp_path, monkeypatch):
    """Point the app's upload dir at an isolated tmp dir for the test."""
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr(screenshot_app, "UPLOAD_DIR", str(d))
    monkeypatch.delenv(screenshot_app.ALLOWED_DIRS_ENV, raising=False)
    return d


@pytest.fixture()
def client():
    screenshot_app.app.config.update(TESTING=True)
    return screenshot_app.app.test_client()


@pytest.fixture()
def outside_file(tmp_path):
    """A file that exists but is outside every allowed root."""
    return _write_png(str(tmp_path / "outside" / "secret.png"))


def _post(client, route, path):
    return client.post(route, json={"file_path": path, "page": 0})


def test_file_in_upload_dir_is_accepted(client, upload_dir):
    """The normal path -- an uploaded file -- still renders."""
    target = _write_png(str(upload_dir / "page.png"), size=(48, 20))

    info = _post(client, "/info", target)
    assert info.status_code == 200
    assert info.get_json()["page_count"] == 1

    page = _post(client, "/page", target)
    assert page.status_code == 200
    body = page.get_json()
    assert body["image_base64"].startswith("data:image/png;base64,")
    assert (body["width"], body["height"]) == (48, 20)


def test_bare_filename_resolves_inside_upload_dir(client, upload_dir):
    """A relative name is interpreted against the upload dir, not the cwd."""
    _write_png(str(upload_dir / "relative.png"))
    assert _post(client, "/info", "relative.png").status_code == 200


@pytest.mark.parametrize("route", FILE_PATH_ROUTES)
def test_absolute_path_outside_is_rejected(client, upload_dir, outside_file, route):
    """An absolute path to a real file elsewhere on the host is refused."""
    assert _post(client, route, outside_file).status_code == 400


@pytest.mark.parametrize("route", FILE_PATH_ROUTES)
def test_dotdot_escape_is_rejected(client, upload_dir, outside_file, route):
    """`..` cannot walk out of the upload dir."""
    escape = os.path.join(str(upload_dir), "..", "outside", "secret.png")
    assert os.path.exists(escape)  # the traversal target really is reachable on disk
    assert _post(client, route, escape).status_code == 400


@pytest.mark.parametrize("route", FILE_PATH_ROUTES)
def test_symlink_escape_is_rejected(client, upload_dir, outside_file, route):
    """A symlink *inside* the upload dir may not point out of it."""
    link = upload_dir / "link.png"
    os.symlink(outside_file, link)
    assert os.path.exists(link)  # the link resolves, so only containment can reject it
    assert _post(client, route, str(link)).status_code == 400


@pytest.mark.parametrize("route", FILE_PATH_ROUTES)
def test_prefix_similar_sibling_dir_is_rejected(client, upload_dir, route):
    """A sibling dir sharing the upload dir's name prefix is not inside it."""
    sneaky = _write_png(str(upload_dir) + "_evil/evil.png")
    assert sneaky.startswith(str(upload_dir))  # string-prefix check would pass this
    assert _post(client, route, sneaky).status_code == 400


def test_info_is_not_an_existence_oracle(client, upload_dir, outside_file):
    """Out-of-bounds paths look identical whether or not they exist."""
    missing = os.path.join(os.path.dirname(outside_file), "does_not_exist.png")
    assert os.path.exists(outside_file) and not os.path.exists(missing)

    present = _post(client, "/info", outside_file)
    absent = _post(client, "/info", missing)
    assert present.status_code == absent.status_code == 400
    assert present.get_json() == absent.get_json()


@pytest.mark.parametrize("route", FILE_PATH_ROUTES)
def test_rejection_does_not_echo_the_path(client, upload_dir, outside_file, route):
    """The error must not leak the resolved path back to the caller."""
    body = _post(client, route, outside_file).get_json()
    assert outside_file not in body["error"]
    assert os.path.dirname(outside_file) not in body["error"]


@pytest.mark.parametrize("route", FILE_PATH_ROUTES)
def test_missing_file_path_is_rejected(client, upload_dir, route):
    for value in ("", "   "):
        assert _post(client, route, value).status_code == 400


def test_extra_allowed_dir_can_be_opted_into(client, upload_dir, tmp_path, monkeypatch):
    """Operators can still serve a local corpus by naming it explicitly."""
    corpus = tmp_path / "corpus"
    target = _write_png(str(corpus / "doc.png"))

    assert _post(client, "/info", target).status_code == 400
    monkeypatch.setenv(screenshot_app.ALLOWED_DIRS_ENV, str(corpus))
    assert _post(client, "/info", target).status_code == 200


def test_default_bind_is_loopback(monkeypatch):
    """The unauthenticated server must not listen off-box unless asked to."""
    monkeypatch.delenv(screenshot_app.HOST_ENV, raising=False)
    captured = {}
    monkeypatch.setattr(screenshot_app.app, "run", lambda **kw: captured.update(kw))

    screenshot_app.main()
    assert captured["host"] == "127.0.0.1"

    monkeypatch.setenv(screenshot_app.HOST_ENV, "0.0.0.0")
    screenshot_app.main()
    assert captured["host"] == "0.0.0.0"
