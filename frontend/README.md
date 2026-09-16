# LumenMorph browser workspace

This is LumenMorph's supported interactive interface for designing tubular
synthetic sequences, inspecting live previews, and launching local generation
jobs. It communicates only with the local Python API; it is not a hosted web
application.

From the repository root, install and run it with:

```bash
npm --prefix frontend ci
backend/.venv/bin/python -m backend.src.cli api --host 127.0.0.1 --port 8765
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

Open <http://127.0.0.1:5173>. See the repository
[installation guide](../docs/INSTALLATION.md) and
[tube workflow guide](../docs/tube_user_guide.md).

## Quality checks

```bash
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run build
```

`dist/` is generated output and is deliberately not committed.
