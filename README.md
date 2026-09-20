<div align="center">

# HackGrader

Hackathon devpost webscraper and code grader/checker pipeline. This aims to check for AI usage, code correctness, format and other parameters that will be used to create an additional metric towards participants scores.

</div>

---

## Getting Started

### Prerequisites

---

The project spins up and hosts a model locally for the analysis phase. HackGrader uses a custom model trained off AI generated code to better predict such code in given hackathon repositories. Although the project primarily uses the model, it will work with a base [qwen coder](https://ollama.com/search?q=qwen+coder) model. [ollama](https://ollama.com/), regardless, is a necessary requirement. Additionally, you will need either [uv](https://github.com/astral-sh/uv) or pip to install packages for the scraper. The core hackGrader code is built around [python](https://www.python.org/) versions ```>=3.14```, while the model code works on python version ```3.12```.

### Installation

---

Pull the code locally

```bash
# clone the repo
git clone https://github.com/Google-Developers-Student-Club-LSU/HackGrader.git

# cd into the project
cd hackGrader
```

Follow steps in [model.md](model/README.md) or Install a qwen coder model

```bash
# pull in the qwen 2.5, 7b parameter model
ollama pull qwen2.5-coder:7b

# pull in the qwen 3, 30b parameter model
ollama pull qwen3-coder:30b
```

#### Python pip

---

Create your virtual environment

```bash
# Windows 10/11
python -m venv .venv

# Linux/MacOS 
python3 -m venv .venv
```

Source into your virtual environment

```bash
# Windows 10/11
. .venv/bin/activate.ps1

# Linux/MacOS 
source .venv/bin/activate
```

Install the packages

```bash
pip install -r requirements.txt
```

#### UV

---

Create the virtual environment

```bash
uv init
```

install the packages

```bash
uv sync
```

### Running

---

Run the pipeline with and pass in a devpost url as an argument for the script

```bash
# standard run
python src/hackGrader/__init__.py https://example.devpost.com

# uv run
uv run hackgrader https://example.devpost.com
```

### Testing

---

Testing the crawling, scoring, or entire pipeline can be done entirely through the test scripts.

The following flags may be used to test various components.

- -c : devpost crawling testing
- -s : llm scoring testing

If no flags are provided, then the entire pipeline will be tested.

```bash
# example script execution
./run_test -c
```

### Generating a Report

---

A PDF report can be generated once the core scoring pipeline has completed and output a .json file.

```sh
uv run src/format.py
```