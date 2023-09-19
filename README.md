# W&B command memo

## Install the W&B library

```bash
pip install wandb
```

```python
import wandb
```

## Organize your hyperparameters

```python
config = {"learning_rate": 0.001}
```

## Start the W&B run

```python
wandb.init(project="my-project", config=config)
```

## Model training here

## Log metrics over time to visualize performance


```python
wandb.log({"loss": loss})
```

## When working in a notebook, finish

```python
wandb.finish()
```