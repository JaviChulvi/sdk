<a href="https://www.ultralytics.com"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🔌 Ultralytics Platform API Python SDK

[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics) [![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com) [![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://www.reddit.com/r/Ultralytics/)

Typed synchronous and asynchronous Python clients generated from the [Ultralytics Platform API](https://platform.ultralytics.com) contract with [Ultralytics OpenAPI](https://github.com/ultralytics/openapi). The [interactive API reference](https://platform.ultralytics.com/api/docs) documents every resource and includes Python examples.

## 🐍 Python

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics-platform?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics-platform/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics-platform)](https://clickpy.clickhouse.com/dashboard/ultralytics-platform) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics-platform?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics-platform/)

Install the standalone [`ultralytics-platform`](https://pypi.org/project/ultralytics-platform/) package from PyPI in a [**Python >=3.11**](https://www.python.org/) environment. It has one lightweight runtime dependency (`httpx`) and does not install the larger `ultralytics` package:

```bash
uv pip install ultralytics-platform
```

Pass your [API key](https://platform.ultralytics.com/settings?tab=api-keys) directly as shown below. Alternatively, omit `api_key` to use `ULTRALYTICS_API_KEY` or the Platform key saved by `yolo login`. Both clients use explicit credentials first, then the environment, then saved settings. Pass `api_key=""` to disable authentication. `yolo logout` removes the saved key; it does not unset an environment variable. The SDK reads the existing Ultralytics settings directory, including `YOLO_CONFIG_DIR` and Linux `XDG_CONFIG_HOME`, without importing or installing `ultralytics`.

```python
from ultralytics_platform import Platform

with Platform(api_key="YOUR_API_KEY") as client:
    response = client.datasets.list("your_username")
```

The asynchronous client exposes the same resource tree:

```python
import asyncio

from ultralytics_platform import AsyncPlatform


async def main():
    async with AsyncPlatform(api_key="YOUR_API_KEY") as client:
        response = await client.datasets.list("your_username")


asyncio.run(main())
```

The package includes typed responses, multipart uploads, retries for temporary failures, structured API errors, custom HTTP clients, and context-manager cleanup.

## Unified `ul` CLI

This package installs `ul`. Local commands lazily delegate to the existing YOLO parser, so install `ultralytics` in the same environment to use them. Cloud commands and authentication work without the ML package.

```bash
ul login                         # securely prompt for a Platform API key
ul logout                        # clear the shared saved key, not environment variables
ul train model=yolo26n.pt data=coco8.yaml epochs=100
ul predict model=yolo26n.pt source="image with spaces.jpg"

ul cloud --help                  # list every generated API command
ul cloud datasets images --help # describe an operation's arguments
ul cloud datasets list
ul cloud datasets images dataset=coco8 limit=20
ul cloud models list project=inspection
ul cloud datasets list owner=another-user
ul cloud datasets                         # list your datasets
ul cloud datasets dataset=coco8           # retrieve one dataset
ul cloud models project=inspection        # list models in a project
ul cloud models project=inspection model=experiment # retrieve one model
```

Cloud syntax follows YOLO: `key=value`, not `--key value`. Command names use hyphens (`storage-integrations`, `signed-url`); argument names match SDK Python keywords (`train_args`, `from_`, `owner_body`). An omitted **path** `owner` is resolved once from the effective credential's account username. Explicit owners always win. No profiles, cached usernames, inferred projects, or inferred clone destinations are introduced. Optional creation owners retain the API's own defaults.

Only supplied values reach SDK methods: omission, `False`, `0`, and nullable `None`/`null` remain distinct. Booleans are case-insensitive and may be bare; schema-declared strings remain strings (including `license=None`). Nested objects, arrays, and whole union bodies use JSON. Structured inputs also accept `@request.json` or `@-` for stdin; only one argument may consume stdin. Binary fields require `@path` and are opened only when the contract declares them as binary:

```bash
ul cloud training start model_id=MODEL_ID train_args=@train.json
ul cloud models predict project=inspection model=experiment body='{"file":"@image.jpg","conf":0.25}'
ul cloud models predict project=inspection model=experiment body=@-
```

Each resource command makes **one operation call**, plus an account lookup if owner inference is needed. It prints the complete response envelope as JSON, text, or bytes. Pagination is explicit: use the endpoint's own `limit`, `offset`, `cursor`, or `page_token` arguments; nothing implicitly fetches all pages. Serialization, credentials, transport, retries, and API errors remain owned by the SDK. The CLI reads SDK signatures and annotations to check input types, scalar choices, and required arguments; the API validates nested JSON bodies, including required fields. It does not interpret JSON Schema at runtime.

For resources with a GET `list` operation, the operation may be omitted. A matching GET `retrieve` operation is selected when its item identifier is supplied. For example, `project` scopes a model list, while `model` identifies one model. This applies to datasets, projects, deployments, models, and exports; storage integrations support implicit listing only. Explicit verbs remain available. Missing retrieval arguments fail without falling back to listing. Resources without these defaults show help, and `--help` never makes an API request. Writes always require an explicit operation.

### Submit and inspect cloud jobs

Use generated API operations directly. Each invocation submits, retrieves, or cancels one operation and exits; there are no cloud workflow shortcuts, polling loops, or automatic artifact downloads.

```bash
ul cloud training start model_id=MODEL_ID gpu_type=rtx-4090 train_args=@train.json
ul cloud models training project=inspection model=experiment
ul cloud models delete-training project=inspection model=experiment
ul cloud deployments predict deployment=production body='{"file":"@image.jpg"}'
ul cloud exports create project=inspection model=experiment format=onnx
ul cloud exports retrieve project=inspection model=experiment export_id=EXPORT_ID
ul cloud exports delete project=inspection model=experiment export_id=EXPORT_ID
```

Cloud execution may incur charges and never falls back to local execution. Inspect the response for job status or artifact URLs. Exit codes are 0 for a successful API call (not necessarily a completed job), 1 for API/network errors, 2 for local input errors, and 130 for interruption. Missing nested fields are reported by the API with exit code 1 and its error message; missing required command arguments fail locally with exit code 2. Interrupting the CLI does not cancel a submitted job; use the explicit cancellation operation.

Local `ul train` remains local even with `ul://` inputs. The existing YOLO callbacks continue to own automatic tracking, cancellation, retries, and checkpoint promotion. `yolo` and `ultralytics` are unchanged.

`ULTRALYTICS_PLATFORM_URL` can select another API origin for CLI invocations. Credentials use the same environment-first and saved-settings fallback as the SDK and `yolo login`; `ul login` validates a key before changing only `api_key` in those settings. A Unix system command may also be named `ul`: activate your Python environment to select this executable, or use `python -m ultralytics_platform.cli` unambiguously.

## 🧩 One Contract, Typed Python

The [Ultralytics Platform API](https://platform.ultralytics.com) contract is the single source of truth for the generated client:

```text
OpenAPI contract
    └── Python SDK # ultralytics-platform
```

The [source repository](https://github.com/ultralytics/sdk) pins the consumed contract and generated output so API changes remain deterministic and reviewable. Generated SDK files should never be edited manually; update the contract, consumer configuration, [package README source](https://github.com/ultralytics/sdk/blob/main/README.python.md), or [generator](https://github.com/ultralytics/openapi) and regenerate.

## 🛠️ Validation

[CI](https://github.com/ultralytics/sdk/actions) regenerates the Python SDK to detect contract mismatch or generated drift. It also formats and lints Python, compiles the package, builds its wheel, installs it through the package boundary, and exercises representative synchronous and asynchronous requests.

## 💡 Contribute

[Ultralytics](https://www.ultralytics.com) thrives on community collaboration, and we deeply value your contributions! Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for details on how you can get involved. We also encourage you to share your feedback through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge thank you 🙏 to all our contributors!

API shape changes belong in the service OpenAPI contract; generated files should not be edited directly.

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/sdk/graphs/contributors)

## 📄 License

- **AGPL-3.0 License**: The generated SDK is licensed under the [AGPL-3.0 License](https://spdx.org/licenses/AGPL-3.0-only.html).
- **Enterprise License**: Commercial licensing is available separately through [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📫 Contact

For bug reports or feature suggestions related to this SDK, please submit an issue via [GitHub Issues](https://github.com/ultralytics/sdk/issues). Join our [Discord](https://discord.com/invite/ultralytics), [Reddit](https://www.reddit.com/r/Ultralytics/), or [Community Forums](https://community.ultralytics.com) for discussions and support!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
