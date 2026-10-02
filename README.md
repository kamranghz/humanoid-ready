# HumanoidReady

*From human video to physically verified, humanoid-ready motion.*

Python package: **hready** (conda env: `hready`, Windows-native — see `docs/decisions.md`).

## Setup

```bash
conda activate hready
pip install -e ".[dev]"
cp configs/paths.example.yaml configs/paths.yaml
```

Edit `configs/paths.yaml` if your data root differs from `D:/projects/hready_data`.

## Tests

```bash
python -c "import hready; print(hready.__version__)"
```

Project spec and checklist: [AGENTS.md](AGENTS.md).
