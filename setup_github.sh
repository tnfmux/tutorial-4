#!/usr/bin/env bash
# push inicial para o GitHub

set -e

GITHUB_USER="tnfmux"
REPO_NAME="ibovespa_sp500"

git init
mkdir -p data output/plots notebooks
touch data/.gitkeep output/plots/.gitkeep notebooks/.gitkeep

git add .
git commit -m "initial commit"
git branch -M main
git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
git push -u origin main

echo "repo em: https://github.com/${GITHUB_USER}/${REPO_NAME}"
