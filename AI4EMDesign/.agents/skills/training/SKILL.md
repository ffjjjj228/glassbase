---
name: training
description: Train surrogate models and RL policies in tmux with live TensorBoard monitoring
license: MIT
compatibility: opencode
---

## What I do

- Create a tmux session with split panes: training (left) and TensorBoard (right)
- Training runs in the background — survives SSH disconnects
- Automatically logs training output to `surrogate_output/<run_name>/train.log`
- After training completes, generate evaluation plots and experiment report

## How to use

### Start training

```bash
cd AI4EMDesign
source scripts/train.sh <run_name> [config_file]
```

Example:
```bash
source scripts/train.sh s2p_s11
source scripts/train.sh s2p_s11 conf/surrogate_s11.conf  # explicit config
```

### Monitor

```bash
tmux a -t train_<run_name>     # attach
# Ctrl+B D                       # detach (training keeps running)
# Ctrl+B Left/Right              # switch between training / TensorBoard panes
```

TensorBoard is accessible at `http://<host>:6006`.

### Directory structure

```
surrogate_output/<run_name>/
├── best_checkpoint.pt       # best model (lowest val loss)
├── checkpoint.pt            # latest model
├── train.log                # full training log (with tee)
├── events.out.*             # TensorBoard event files
├── eval_plots/              # evaluation comparison plots
│   ├── sample_0000.png
│   └── ...
└── experiment_report.md     # full experiment report
```

### After training completes

Run the evaluation script to generate plots + report:
```bash
cd AI4EMDesign
uv run python3 scripts/evaluate_s11.py --checkpoint surrogate_output/<run_name>/best_checkpoint.pt
```
