# LumenMorph backend

The Python backend contains LumenMorph's generator, validator, command-line
interface, benchmark tools and local FastAPI service. Run commands from the
repository root so that `backend` is importable as a Python package.

Install the locked environment:

```bash
uv sync --project backend --locked
```

Run the command-line interface:

```bash
backend/.venv/bin/python -m backend.src.cli --help
backend/.venv/bin/python -m backend.src.cli generate --config configs/tube_default.json --out /tmp/lumenmorph_tube --single-sequence
backend/.venv/bin/python -m backend.src.cli validate --dataset /tmp/lumenmorph_tube
```

Start the local API for the browser workspace:

```bash
backend/.venv/bin/python -m backend.src.cli api --host 127.0.0.1 --port 8765
```

The API can generate files at a requested local output path. Keep it bound to
`127.0.0.1` unless you have secured the network environment yourself. See the
repository [architecture](../docs/ARCHITECTURE.md) and
[reproducibility guidance](../docs/REPRODUCIBILITY.md).
