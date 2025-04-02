#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parent directory (CLAH_IA root)
CLAH_IA_ROOT="$(dirname "$SCRIPT_DIR")"

# Run ServiceSetup.py
echo "Running ServiceSetup.py..."

# Check if the script executed successfully
if python "$CLAH_IA_ROOT/CLAH_ImageAnalysis/utils/ServiceSetup.py"; then
  echo "Service setup completed successfully!"
else
  echo "Error: Service setup failed!"
  exit 1
fi
