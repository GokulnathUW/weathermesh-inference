#!/bin/bash
# Installs WeatherMesh-3 dependencies, then inference pipeline dependencies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing WeatherMesh-3 packages..."
cd "$SCRIPT_DIR/WeatherMesh-3"
bash install.sh

echo "Installing inference pipeline packages..."
cd "$SCRIPT_DIR"
pip install -r requirements.txt

echo "Done."
