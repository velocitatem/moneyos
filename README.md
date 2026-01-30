# MoneyOS (Money Operating System)

A naive quote-control simulator for **dynamic pricing** and **market making** experiments.
MoneyOS models pricing as a closed loop of **Quote > Arrival > Execution > Position**, enabling
controlled experimentation with demand models, inventory constraints, and reward shaping.

_(built as a byproduct for inspiration of another project)_

## Why MoneyOS

- Modular building blocks (mechanisms, arrivals, execution, position, objectives)
- Preconfigured scenarios for retail pricing and market making
- Baseline policies + evaluation utilities for policy comparison
- Optional **Gymnasium** wrapper for RL training
- Optional **TensorBoard** logging helpers for rollouts

## Quickstart

Requirements: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U numpy
python lab/run_example.py
```

Optional extras:

```bash
pip install gymnasium tensorboardX
```

## Minimal usage

Retail dynamic pricing:

```python
from lab.config import make_retail_platform
from lab.experiments import rollout, fixed_price_policy

platform = make_retail_platform()
policy = fixed_price_policy(platform.instruments.refs)
result = rollout(platform, policy, n_steps=100)
print(result.total_pnl)
```

Market making:

```python
from lab.config import make_market_making_platform
from lab.experiments import rollout

platform = make_market_making_platform()
mm_policy = lambda obs, t: (platform.instruments.refs, 1.0)
result = rollout(platform, mm_policy, n_steps=200, seed=42)
print(result.total_pnl)
```

## Project layout

| Path | Purpose |
|------|---------|
| `lab/outlet` | Core simulation engine, domain types, mechanisms, objectives |
| `lab/population` | Demand arrivals, execution models, competitor/market dynamics |
| `lab/experiments` | Rollout utilities, baseline policies, off-policy helpers |
| `lab/observability` | Hooks + TensorBoard logging (`tensorboardX` required) |
| `lab/docs` | Sphinx documentation |

## Docs

Build the HTML docs:

```bash
make -C lab/docs html
```

## Notes

- The detailed simulator overview lives in `lab/README.md`.
- This repo currently runs from source (no packaging/publishing yet).

