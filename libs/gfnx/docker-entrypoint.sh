#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: docker run <image> <baseline_script.py> [args...]"
    exit 1
fi

SCRIPT_NAME=$1
shift

exec python "baselines/$SCRIPT_NAME" "$@"
