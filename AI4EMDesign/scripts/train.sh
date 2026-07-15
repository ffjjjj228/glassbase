#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: source train.sh <run_name> [config_file]

Start surrogate model training in tmux with live TensorBoard monitoring.

Arguments:
  run_name      Experiment name (e.g. s2p_s11, s4p_v2)
  config_file   Config path, default: conf/surrogate_<run_name>.conf

Examples:
  source train.sh s2p_s11
  source train.sh s2p_s11 conf/surrogate_s11.conf

After starting:
  tmux a -t train_<run_name>    # attach to monitor
  Ctrl+B D                       # detach (training keeps running)
  Ctrl+B Left/Right              # switch between panes
EOF
}

if [ $# -lt 1 ]; then
    usage
    return 1 2>/dev/null || exit 1
fi

RUN_NAME="$1"
CONF="${2:-conf/surrogate_${RUN_NAME}.conf}"
SESSION="train_${RUN_NAME}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[ERROR] Session '$SESSION' already exists. Attach with: tmux a -t $SESSION"
    return 1 2>/dev/null || exit 1
fi

mkdir -p "$ROOT_DIR/surrogate_output/$RUN_NAME"

tmux new-session -d -s "$SESSION" -n "train"
tmux send-keys -t "$SESSION" "cd $ROOT_DIR" Enter
tmux send-keys -t "$SESSION" "PYTHONPATH=$ROOT_DIR uv run python3 src/surrogate_model_training/main.py --conf $CONF -o surrogate_output/$RUN_NAME 2>&1 | tee surrogate_output/$RUN_NAME/train.log" Enter

tmux split-window -h -t "$SESSION"
tmux send-keys -t "$SESSION" "sleep 5 && cd $ROOT_DIR && setsid uv run tensorboard --logdir surrogate_output/$RUN_NAME --host 0.0.0.0 --port 6006 2>&1 | tee /tmp/tb_${RUN_NAME}.log" Enter

echo "================================================================"
echo "  Started: $RUN_NAME"
echo "  Config:  $CONF"
echo "  Output:  surrogate_output/$RUN_NAME"
echo ""
echo "  Attach:  tmux a -t $SESSION"
echo "  Detach:  Ctrl+B D"
echo "  Panes:   Ctrl+B Left/Right"
echo "================================================================"
