<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# scenarios

## Purpose
TTA experiment scenarios — orchestrates iteration over multiple shifted dataset variants and yields per-step + averaged results. The four scenario types model different real-world test-time conditions: **Standard** (single static target), **Gradual** (smooth domain trajectory), **Continual** (sequence of discrete shifts, possibly stateful across them), and **Universal** (all-of-the-above superset).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: `BaseDataset`, `BaseScenario`, the four `Base{Standard,Gradual,Continual,Universal}TTAScenario` classes, plus concrete scenarios (`SHIFTContinuousScenarioForGradualTTA`, `CityScapesContinuousScenarioForGradualTTA`, `SHIFTDiscreteScenarioForContinualTTA`, `ACDCScenarioForContinualTTA`, `CityScapesDiscreteScenarioForContinualTTA`). |
| `base.py` | `ScenarioType` enum, `BaseScenario` (dict subclass keyed by shift step). `play(script, index, **kwargs)` is the iteration driver: builds a `DataLoader` per dataset, calls `script(key.value, loader, loader_len, **kwargs)`, accumulates per-step results, and finally yields an `"avg"` aggregate. |
| `continual.py` | Concrete CTTA scenarios over SHIFT discrete, ACDC, and CityScapes discrete splits. |
| `gradual.py` | Concrete gradual-TTA scenarios over SHIFT and CityScapes continuous splits. |
| `standard.py` / `universal.py` | Empty stubs — placeholders for upcoming Standard/Universal scenarios. |

## For AI Agents

### Working In This Directory
- A scenario *is a dict*: keys are shift descriptors (enum values), values are `BaseDataset` instances. `__init__` walks `self.order` (defaults to `cls.DEFAULT`), constructs each per-key dataset via either `subset_type=` or `corruption_type=` (decided by introspecting `cls.dataset_class` signature), and stores them.
- `play(script, index)` is a generator: yields `(result, index)` after each step *and* one final time after appending the `"avg"` aggregate. `index` defaults to `["Trial"]` (single model); pass a longer list when running parallel models so per-trial dicts line up.
- The `script` callable returns a per-batch result `dict` (or list of dicts for parallel evaluation). Reuse `Evaluator.evaluate` from `ttadapters.utils.validator` as the canonical script — its signature already matches.
- Stub files (`standard.py`, `universal.py`, etc.) are empty by design. Add scenarios there rather than inventing new files unless a fundamentally new scenario type is needed.

### Testing Requirements
- Smoke-test new scenarios by constructing with `force_download=False` against a cached dataset root, then calling `next(scenario.play(...))` once and asserting the result dict has the expected keys.

### Common Patterns
- `exclude_list` lets callers skip specific shift keys without subclassing.
- `transform` / `target_transform` / `transforms` are passed through to the underlying dataset class — preserve that pass-through when adding new scenarios.

## Dependencies

### Internal
- `..` — `BaseDataset`. Concrete scenarios import dataset classes from `ttadapters.datasets`.

### External
- `torch.utils.data.DataLoader`, `tqdm`, `inspect`, `enum`.

<!-- MANUAL: -->
