#!/bin/bash

if command -v uv &> /dev/null; then
  if [ $# -eq 0 ]; then
    uv run pytest -sv tests/test_pipeline.py 
    exit 0
  fi

  while getopts ":sc" opt; do
    case $opt in
      s)
        uv run pytest -sv tests/test_scoring.py
        ;;
      c)
        uv run pytest -sv tests/test_crawling.py
        ;;
      \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    esac
  done
else
  if [ $# -eq 0 ]; then
    python -m pytest -sv tests/test_pipeline.py 
    exit 0
  fi

  while getopts ":sc" opt; do
    case $opt in
      s)
        python -m pytest -sv tests/test_scoring.py
        ;;
      c)
        python -m pytest -sv tests/test_crawling.py
        ;;
      \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    esac
  done
fi