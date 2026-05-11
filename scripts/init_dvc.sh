#!/usr/bin/env bash
set -euo pipefail

# Bootstrap minimal pour l'exercice 4.2.
# Nécessite dvc installé dans l'environnement appelant.

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc est introuvable. Installez-le puis relancez ce script." >&2
  exit 1
fi

dvc init
mkdir -p data/features
dvc add data/processed data/features

echo "DVC initialisé."
echo "Pensez à commit:"
echo "  git add .dvc .gitignore data/processed.dvc data/features.dvc"
