#!/usr/bin/env bash
# Activate the dedicated RL virtual environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSIX_ACTIVATE="$SCRIPT_DIR/trading_rl_env/bin/activate"
WIN_ACTIVATE="$SCRIPT_DIR/trading_rl_env/Scripts/activate"

if [[ -f "$POSIX_ACTIVATE" ]]; then
	source "$POSIX_ACTIVATE"
elif [[ -f "$WIN_ACTIVATE" ]]; then
	source "$WIN_ACTIVATE"
else
	echo "Unable to locate trading_rl_env activation script" >&2
	exit 1
fi

echo "Activated trading_rl_env"
