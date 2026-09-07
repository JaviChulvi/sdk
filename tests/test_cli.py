# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Guard the new CLI's mutation-facing input and executable boundaries without live API calls."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from ultralytics_platform import Platform
from ultralytics_platform.cli import (
    MULTIPART_FILES,
    assignments,
    describe,
    main,
    open_inputs,
    parse,
)


def parsed(command, tokens, stack):
    resource, method = (part.replace("-", "_") for part in command.split())
    with Platform(api_key="") as client:
        target = getattr(getattr(client, resource), method)
        arguments = describe(target)
    return open_inputs(
        parse(assignments(tokens), arguments), arguments, MULTIPART_FILES.get(f"{resource}.{method}", []), stack
    )


def test_inventory():
    """Every public SDK operation can be described and accepts its CLI argument names."""
    with Platform(api_key="") as client:
        for name, resource in vars(client).items():
            if not name.startswith("_"):
                for method_name, method in inspect.getmembers(resource, inspect.ismethod):
                    if not method_name.startswith("_"):
                        arguments = describe(method)
                        signature = inspect.signature(method)
                        assert set(arguments) == signature.parameters.keys() - {"timeout", "extra_headers"}
                        signature.bind(**dict.fromkeys(arguments))
    assert main(["--help"]) == main(["cloud", "--help"]) == main(["cloud", "datasets", "images", "--help"]) == 0
    assert main(["cloud", "training", "start", "--help"]) == main(["cloud", "exports", "create", "--help"]) == 0
    assert main(["cloud", "train", "--help"]) == main(["cloud", "export", "--help"]) == 2
    assert main(["login", "--help"]) == main(["logout", "--help"]) == 0
    assert main(["cloud", "datasets", "unknown=value"]) == 2
    assert "ultralytics" not in sys.modules


def test_values_and_files(tmp_path):
    """Preserve omitted/false/zero/null, strings, nested JSON, whole unions, and binary ownership."""
    with ExitStack() as stack:
        args = parsed(
            "models update",
            ["project=p", "model=m", "starred=False", "epochs=0", "color=null", "license=None"],
            stack,
        )
        assert args == {
            "project": "p",
            "model": "m",
            "starred": False,
            "epochs": 0,
            "color": None,
            "license": "None",
        }
        assert parsed("models update", ["project=p", "model=m", "starred"], stack)["starred"] is True
        args = parsed("models update", ["project=p", "model=m", "description=token=", "starred"], stack)
        assert args["description"] == "token=" and args["starred"] is True
        for text, expected in (("description=", ""), ("description=token=", "token=")):
            args = parsed("models update", ["project=p", "model=m", text, "starred=False"], stack)
            assert args["description"] == expected and args["starred"] is False
        assert parsed("models update", ["project=p", "model=m", "color=#fff"], stack)["color"] == "#fff"
        assert parsed("deployments update", ["deployment=production", 'body={"action":"stop"}'], stack)["body"] == {
            "action": "stop"
        }
        assert parsed("datasets preview-roboflow", ["api_key=@not-a-file"], stack)["api_key"] == "@not-a-file"
        assert parsed("models update", ["project=p", "model=m", "best_fitness=.25"], stack)["best_fitness"] == 0.25
        nested = {"labels": [False, 0, None, {"value": "None"}]}
        assert (
            parsed("models update", ["project=p", "model=m", f"metadata={json.dumps(nested)}"], stack)["metadata"]
            == nested
        )
        assert (
            parsed("datasets images", ["dataset=d", "include_total=false", "limit", "=", "0"], stack)["include_total"]
            == "false"
        )
        assert (
            parsed("projects create", ["project=p", "name", "=a name with spaces"], stack)["name"]
            == "a name with spaces"
        )
        request = tmp_path / "request.json"
        request.write_text('{"model":"yolo26n.pt","data":"ul://jane/datasets/coco8","epochs":100}', encoding="utf-8")
        args = parsed(
            "training start", ["model_id=m", f"train_args=@{request}", "capture_dataset_version=false"], stack
        )
        assert args["train_args"]["epochs"] == 100 and args["capture_dataset_version"] is False
        upload = tmp_path / "image with spaces.jpg"
        upload.write_bytes(b"image contents")
        body = json.dumps({"file": f"@{upload}", "conf": 0.25, "normalize": False})
        args = parsed("models predict", ["project=p", "model=m", f"body={body}"], stack)
        handle = args["body"]["file"]
        assert handle.read() == b"image contents" and args["body"]["normalize"] is False
    assert handle.closed


@pytest.mark.parametrize(
    "command,tokens",
    [
        ("models update", ["project=p", "model=m", "starred=0"]),
        ("models update", ["project=p", "model=m", "epochs=True"]),
        ("models update", ["project=p", "model=m", "epochs=0", "epochs=1"]),
        ("models update", ["project=p", "model=m", "name"]),
        ("models update", ["project=p", "model=m", "nonesuch=x"]),
        ("datasets images", ["--dataset", "d"]),
        ("datasets images", ["limit=2"]),
        ("datasets images", ["dataset=d", "limit=null"]),
        ("models predict", ["project=p", "model=m", "body=not-json"]),
    ],
)
def test_invalid_inputs(command, tokens):
    with ExitStack() as stack, pytest.raises(ValueError):
        parsed(command, tokens, stack)


def test_diagnostics():
    with ExitStack() as stack:
        with pytest.raises(ValueError, match="limit: expected integer") as error:
            parsed("datasets images", ["dataset=d", "limit=private-value"], stack)
        assert "private-value" not in str(error.value)


def test_cli_wire_and_auth(tmp_path):
    """Exercise the installed CLI and SDK over real loopback HTTP, with isolated YOLO settings."""
    requests = []

    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append((self.command, self.path, dict(self.headers), body))
            if self.headers.get("Authorization") == "Bearer invalid":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"sensitive upstream response")
                return
            invalid_training = self.path == "/api/training/start" and json.loads(body)["trainArgs"] == {"epochs": 100}
            payload = {"username": "jane"} if self.path == "/api/account/summary" else {"ok": True}
            if invalid_training:
                payload = {"error": "model and data are required"}
            self.send_response(422 if invalid_training else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {
        **os.environ,
        "YOLO_CONFIG_DIR": str(tmp_path),
        "ULTRALYTICS_API_KEY": "",
        "ULTRALYTICS_PLATFORM_URL": origin,
    }

    def run(*args, input=None):
        return subprocess.run(
            [sys.executable, "-m", "ultralytics_platform.cli", *args],
            env=env,
            input=input,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    try:
        assert run("login", "ul_" + "a" * 40).returncode == 0
        settings = tmp_path / "Ultralytics" / "settings.json"
        if os.name == "posix":
            assert settings.stat().st_mode & 0o777 == 0o600
        saved = json.loads(settings.read_text())
        saved["runs_dir"] = "custom runs"
        settings.write_text(json.dumps(saved))
        assert run("cloud", "datasets", "list", "limit=0").returncode == 0
        assert [item[1] for item in requests[-2:]] == ["/api/account/summary", "/api/datasets/jane?limit=0"]
        assert run("cloud", "datasets", "list", "owner=team", "limit=1").returncode == 0
        assert requests[-1][1] == "/api/datasets/team?limit=1"
        env["ULTRALYTICS_API_KEY"] = "environment-key"
        assert run("cloud", "datasets", "list").returncode == 0
        assert all(item[2]["Authorization"] == "Bearer environment-key" for item in requests[-2:])
        env["ULTRALYTICS_API_KEY"] = ""
        for resource, arguments, endpoint in (
            ("datasets", [], "/api/datasets/jane"),
            ("datasets", ["dataset", "=", "coco8"], "/api/datasets/jane/coco8"),
            ("projects", [], "/api/projects/jane"),
            ("projects", ["project=p"], "/api/projects/jane/p"),
            ("deployments", [], "/api/deployments/jane"),
            ("deployments", ["deployment=production"], "/api/deployments/jane/production"),
            ("models", ["project=p"], "/api/models/jane/p"),
            ("models", ["project=p", "model=m"], "/api/models/jane/p/m"),
            ("exports", ["project=p", "model=m"], "/api/models/jane/p/m/exports"),
            ("exports", ["project=p", "model=m", "export_id=e"], "/api/models/jane/p/m/exports/e"),
            ("storage-integrations", [], "/api/integrations/buckets"),
        ):
            count = len(requests)
            owner = [] if resource == "storage-integrations" else ["owner=jane"]
            assert run("cloud", resource, *owner, *arguments).returncode == 0
            assert len(requests) == count + 1 and requests[-1][:2] == ("GET", endpoint)
        count = len(requests)
        assert run("cloud", "datasets", "--help").returncode == 0
        for name in (["name=token="], ["name="], ["name", "="]):
            for flag in ("help", "--help", "-h"):
                help_result = run("cloud", "projects", "create", "project=p", *name, flag)
                assert help_result.returncode == 0 and "name (" in help_result.stdout
                assert "timeout (" not in help_result.stdout and "extra_headers (" not in help_result.stdout
        assert run("cloud", "datasets", "dataset=coco8", "--help").returncode == 0
        assert run("cloud", "training").returncode == 0
        assert run("cloud", "models", "model=m").returncode == 2
        assert run("cloud", "datasets", "dataset=coco8", "limit=1").returncode == 2
        assert run("cloud", "datasets", "owner=jane", "typo").returncode == 2
        assert len(requests) == count
        assert run("cloud", "datasets", "dataset=coco8").returncode == 0
        assert [item[1] for item in requests[-2:]] == ["/api/account/summary", "/api/datasets/jane/coco8"]
        for value in ("help", "--help", "-h", "a=b", "https://example.com/?a=b&c=d"):
            count = len(requests)
            assert run("cloud", "projects", "create", "project=p", f"name={value}").returncode == 0
            assert len(requests) == count + 1 and json.loads(requests[-1][3])["name"] == value
        assert (
            run(
                "cloud",
                "models",
                "predict",
                "owner=jane",
                "project=p",
                "model=m",
                "body=@-",
                input='{"source":"https://example.com/a.jpg","normalize":false}',
            ).returncode
            == 0
        )
        assert b"false" in requests[-1][3]
        image = tmp_path / "image with spaces.jpg"
        image.write_bytes(b"binary image contents")
        assert (
            run(
                "cloud",
                "deployments",
                "predict",
                "deployment=production",
                f"body={json.dumps({'file': '@' + str(image), 'normalize': False})}",
            ).returncode
            == 0
        )
        assert requests[-1][1] == "/api/deployments/jane/production/predict"
        assert b"binary image contents" in requests[-1][3]
        count = len(requests)
        assert (
            run(
                "cloud", "models", "update", "project=p", "model=m", "metadata=@-", "train_args=@-", input="{}"
            ).returncode
            == 2
        )
        assert len(requests) == count
        assert (
            run(
                "cloud",
                "training",
                "start",
                "model_id=model-id",
                "gpu_type=rtx-4090",
                'train_args={"model":"yolo26n.pt","data":"ul://jane/datasets/coco8","epochs":100}',
            ).returncode
            == 0
        )
        assert len(requests) == count + 1
        submitted = json.loads(requests[-1][3])
        assert requests[-1][1] == "/api/training/start"
        assert submitted["modelId"] == "model-id" and "captureDatasetVersion" not in submitted
        failed = run("cloud", "training", "start", "model_id=m", 'train_args={"epochs":100}')
        assert failed.returncode == 1 and "HTTP 422): model and data are required" in failed.stderr
        assert len(requests) == count + 2
        count = len(requests)
        assert run("cloud", "exports", "create", "owner=jane", "project=p", "model=m", "format=onnx").returncode == 0
        assert len(requests) == count + 1 and requests[-1][1] == "/api/models/jane/p/m/exports"
        assert (
            run("cloud", "exports", "retrieve", "owner=jane", "project=p", "model=m", "export_id=job").returncode == 0
        )
        assert len(requests) == count + 2 and requests[-1][1].endswith("/exports/job")
        failed = run("login", "invalid")
        assert failed.returncode == 1 and "sensitive" not in failed.stderr
        assert json.loads(settings.read_text())["api_key"] == saved["api_key"]
        assert run("logout").returncode == 0
        assert json.loads(settings.read_text()) == {**saved, "api_key": ""}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
