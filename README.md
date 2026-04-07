# ecommerce_text_removal_system_v2_mvp

Clean MVP v2 mainline for e-commerce Chinese text removal.

## Current MVP flow
1. OCR detects text groups.
2. Protect phase 1 runs before user selection.
3. User selects `selected_group_ids`.
4. Execution mask is built from OCR groups + dilation - protect mask.
5. Connected mask regions are processed one by one.
6. Per region:
   - simple background -> simple fill
   - medium complexity -> LaMa backend
   - complex region -> reject
7. Hard gate decides whether the region is accepted.
8. Accepted regions are composited back into the image.

## Important scope
- Protect phase 1 is intentionally limited to:
  - product-body auto protect
  - lightweight colorful badge candidate protect
- Analyze and diagnostics distinguish protect sources instead of treating protect as one opaque blob.
- User-provided protect mask is frozen out of the v2 main flow for now.
- This repo does not implement a full automatic protect-recognition system.
- Medium regions REQUIRE a real LaMa backend by default.
- If no backend is available, regions are rejected honestly unless `allow_lama_fallback` is explicitly enabled.
- OCR now reports `ocr_provider` and `ocr_degraded` explicitly:
  - Google Vision success -> `google_vision`, `False`
  - Heuristic path -> `heuristic`, `True`
- Each returned OCR group now includes `text_reliable`:
  - `true` when the active OCR provider is Google Vision
  - `false` when the system is running on heuristic fallback

## Known environment recommendations
- Main project: currently run on this machine with Python `3.14.3` in `.venv`.
- LaMa thin service: currently run on this machine with Python `3.11.0` in `.venv-lama`.
- Keep them in separate virtual environments:
  - main project can use the newer interpreter
  - `simple-lama-inpainting` in the LaMa service currently behaves better in a dedicated Python 3.11 environment
  - this avoids reintroducing the `numpy 1.26.4` / wheel mismatch seen under Python 3.13

## Main project environment
Create the main virtual environment from the repo root in Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your machine does not have the `py -3.14` launcher entry, use an equivalent installed interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Main project environment variables
The main project reads `.env` automatically through `env_bootstrap.py`.

Create or update `.env` in the repo root:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\google-service-account.json
LAMA_EXTERNAL_URL=http://127.0.0.1:8009
```

Notes:
- `GOOGLE_APPLICATION_CREDENTIALS` is required if you want real Google Vision OCR.
- If that variable is missing or the client package is unavailable, OCR falls back to the heuristic path.
- `LAMA_EXTERNAL_URL` should point to the LaMa service base URL only. The adapter appends `/restore` itself.

## Run the main project
Example CLI run from the repo root:

```powershell
.\.venv\Scripts\Activate.ps1
python .\run_auto.py `
  --image .\demo\inputs\4A9706E2-0E6C-4BCD-AC85-C68B5866B6B3.jpeg `
  --output .\demo\outputs\out.png `
  --overlay .\demo\outputs\overlay.png `
  --diagnostics .\demo\outputs\diag.json `
  --debug-dir .\demo\outputs\debug
```

If the LaMa service is already running and you do not want to rely on `.env`, you can pass the service URL explicitly:

```powershell
python .\run_auto.py `
  --image .\demo\inputs\4A9706E2-0E6C-4BCD-AC85-C68B5866B6B3.jpeg `
  --output .\demo\outputs\out.png `
  --external-url http://127.0.0.1:8009
```

Optional modes:

```powershell
python .\run_auto.py ... --allow-lama-fallback
python .\run_auto.py ... --no-require-lama
```

## LaMa service environment
Create a separate LaMa service virtual environment with Python 3.11:

```powershell
py -3.11 -m venv .venv-lama
.\.venv-lama\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lama.txt
```

Do not reuse the main `.venv` for this service.

Python version guidance for the LaMa service:
- Recommended: Python `3.11`
- Not recommended in the current setup: Python `3.13` and `3.14`
- Reason: `simple-lama-inpainting` pulled a `numpy 1.26.4` dependency path here, and the Python 3.13 rebuild failed because the required wheel chain was not available in that setup

## Run the LaMa service
Start the thin local LaMa HTTP service from the repo root:

```powershell
.\.venv-lama\Scripts\Activate.ps1
python .\run_lama_server.py
```

Optional service variables:

```powershell
$env:LAMA_SERVER_HOST = "127.0.0.1"
$env:LAMA_SERVER_PORT = "8009"
python .\run_lama_server.py
```

When the service is reachable at `http://127.0.0.1:8009`, set this in the main project environment:

```powershell
$env:LAMA_EXTERNAL_URL = "http://127.0.0.1:8009"
```

The adapter sends requests to `http://127.0.0.1:8009/restore`.
