# TNC

## Installation

```bash
bash scripts/setup.sh
```

## Evaluation

Outside of Gypsum:

```bash
bash scripts/run_eval.sh --config tnc_test --tag test
```

In Gypsum:

```bash
sbatch -p titanx-short --gres=gpu:1 --output=out/test.out scripts/run_eval.sh --config tnc_test --tag test
```
