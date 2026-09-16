# Install LumenMorph

LumenMorph is a local research application. The supported browser workspace
consists of a Python backend and a React frontend running on the same machine.
It does not require an account, cloud service, or network access after the
dependencies have been installed.

## Prerequisites

- Git.
- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for the locked Python environment.
- Node.js 22 LTS or newer and npm for the browser workspace.

On Linux, OpenCV may require system graphics/video libraries supplied by the
distribution. Install the usual OpenGL and video packages if importing `cv2`
fails. Use your operating system's package manager rather than copying a
distribution-specific command from another platform.

## Clone and install

```bash
git clone --recurse-submodules --depth 1 https://github.com/icirauqui/LumenMorph.git
cd LumenMorph
uv sync --project backend --locked
npm --prefix frontend ci
```

The backend environment is created at `backend/.venv`. `uv.lock` fixes the
Python dependency resolution. Do not replace it with an unpinned installation
when reproducing a reported result.

Check the installation:

```bash
backend/.venv/bin/python -m backend.src.cli --help
backend/.venv/bin/python -m pytest -q
npm --prefix frontend run lint
npm --prefix frontend test
```

On Windows, invoke the environment's `python.exe` under `backend/.venv/Scripts`
instead of `backend/.venv/bin/python`.

## Run the supported browser workspace

Terminal 1, from the repository root:

```bash
backend/.venv/bin/python -m backend.src.cli api --host 127.0.0.1 --port 8765
```

Terminal 2:

```bash
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

Open <http://127.0.0.1:5173>. The status indicator should become **online**.
The backend serves the API only; Vite serves the browser interface during
development.

## Common problems

### `uv sync` cannot find Python 3.12

Install Python 3.12 or ask uv to provision it:

```bash
uv python install 3.12
uv sync --project backend --locked
```

### The browser says `offline`

Confirm the backend is running at `http://127.0.0.1:8765`. The browser's API
address can be changed in the left-hand **Backend** panel if you intentionally
run the backend on another local port.

### The preview is slow

Use a lower preview resolution or lower radial/longitudinal sample counts while
editing. A preview is a design aid; it is not the final dataset render.
