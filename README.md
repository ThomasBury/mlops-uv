
# Mastering Python Project Management with UV: MLOps

[![Docs Build](https://img.shields.io/github/actions/workflow/status/OWNER/mlops-uv/docs.yaml?label=docs&branch=main)](https://github.com/OWNER/mlops-uv/actions/workflows/docs.yaml)
[![Published Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://OWNER.github.io/mlops-uv/)

> ℹ️ Replace `OWNER` in the badge and docs links with your GitHub username or organization.

## How to use

You have two options to follow along with this guide:

1. Build the project from scratch by manually setting up the structure and copy-pasting the provided code base (src and tests folders).

2. Clone the repository, install dependencies using the command `uv sync`, and run the commands explained below directly to:

    * Execute the test suite
    * Build the Docker image
    * Modify and test GitHub Actions
  

Ship it like it's hot! 🚢🔥

## Project metadata

- **Canonical project name**: `acebet` (from `[project].name` in `pyproject.toml`)
- **Package (import) name**: `acebet` (from `src/acebet/`)
- **Version source**: dynamic version from `acebet.__version__` via `[tool.setuptools.dynamic]`
- **License file path**: `LICENSE` (from `[project].license.file`)

## Introduction
MLOps (Machine Learning Operations) is all about bringing DevOps principles into machine learning, making **model deployment, versioning, and monitoring more efficient**. However, managing dependencies, ensuring reproducibility, and streamlining deployments can be a major headache for ML/DS teams.

That's where **UV** comes in—a fast, modern package manager that simplifies **dependency management, build processes, and CI/CD** for Python projects.

In this article, we'll explore how **UV** can enhance MLOps workflows through **AceBet**, a mock-up **FastAPI app** that predicts the winner of an ATP match (for demonstration purposes only—don’t bet your savings on it!). We'll cover:

- Setting up a **UV-based MLOps project**
- Managing **dependencies and lockfiles**
- Automating **CI/CD with GitHub Actions**
- **Building and deploying with Docker**

### **Prerequisites**
Make sure to read:
- [Part 1: UV Basics](https://bury-thomas.medium.com/mastering-python-project-management-with-uv-part1-its-time-to-ditch-poetry-c2590091d90a)
- [Part 2: Advanced UV Features](https://bury-thomas.medium.com/mastering-python-project-management-with-uv-part-2-deep-dives-and-advanced-use-1e2540e6f4a6)

---

## **📦 Initializing an MLOps Project with UV**
When working on an **MLOps project**, structuring your codebase properly is crucial. We'll start by setting up a **packaged application** using UV:

```bash
uv init --package acebet
```

A **packaged application** follows the **src-based structure**, where the source code is contained within a **dedicated package directory (`src/acebet`)**. This approach is beneficial for:

✅ **Large applications with multiple modules**  
✅ **Projects that need to be distributed** (e.g., PyPI packages, CLI tools)  
✅ **Better namespace isolation**, preventing import conflicts  
✅ **Improved testability and modularity**  

### **Example Directory Structure**
```
acebet/
├── src/
│   ├── acebet/
│   │   ├── __init__.py
│   │   ├── data_prep.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   └── utils.py
├── tests/
│   ├── test_model.py
├── pyproject.toml
└── README.md
```

This structure ensures:
✔ **Encapsulation**: The application is a **proper Python package**, avoiding accidental name conflicts.  
✔ **Reusability**: Can be **installed via `pip install .` or published to PyPI**.  
✔ **Cleaner Imports**: Enforces absolute imports (`from acebet.utils import foo`) instead of relative imports.  
✔ **Better CI/CD Support**: Easier to **package and distribute** in Docker, PyPI, or GitHub Actions.  

### **Regular App vs. Packaged App**
👉 **For quick scripts or internal projects?** Use a **regular application**.  
👉 **For scalable, maintainable, and deployable projects?** Use a **packaged application**.  

Since **AceBet is a full MLOps project**, **we’ll use a packaged application**.

---

## **🔧 Managing Dependencies with UV**
### **Installing Core Dependencies**
Once your project is initialized, install the necessary dependencies for **developing AceBet**, including **FastAPI** and **machine learning libraries** like Scikit-learn:

```bash
uv add fastapi scikit-learn pandas lightgbm
```

UV will **automatically resolve versions** and install the required packages.

### **Creating a Lockfile for Reproducibility**
One of UV's key advantages is ensuring **dependency reproducibility** with a **lockfile**. This guarantees that all environments (**local, staging, production**) use the **same** dependency versions.

Once you're satisfied with the **initial codebase**, generate a **lockfile**:

```bash
uv lock
```

Or, if you want to sync **all dependencies** in one go:

```bash
uv sync
```

This process ensures **version consistency across environments**, which is an **essential practice in MLOps**.

---

## **🛠 Adding Testing Dependencies & Running Tests**
Testing is **just as important** as model accuracy in MLOps. UV provides **three different ways** to manage **testing tools** (e.g., `pytest`):

| **Method**                 | **Command**                    | **Use Case** |
|----------------------------|--------------------------------|-------------|
| **Adding as Dev Dependency** | `uv add --dev pytest`         | When `pytest` is part of your **Python project** |
| **Running Temporarily**     | `uvx pytest tests`            | When you **only need to run it occasionally** |
| **Installing Persistently** | `uv tool install pytest`      | When you need `pytest` in **Docker** or as a global CLI |

For **MLOps best practices**, we **add testing dependencies**:

```bash
uv add --dev pytest
```

To **run tests**:

```bash
uv run pytest tests
```

**👉 Best Practice:** Explicitly list **all required dependencies** in `pyproject.toml` for **consistent test environments**.

---

## **🚀 Automating CI/CD with GitHub Actions**
A **CI/CD pipeline** ensures your **models and applications** remain **production-ready**. UV makes **GitHub Actions** setup **seamless**.

A simple **workflow** to **run tests on every commit to `main`**:

```yaml
name: Testing
on:
  push:
    branches:
      - "main"
jobs:
  uv-example:
    name: Python
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install UV
        uses: astral-sh/setup-uv@v5
      - name: Install the project
        run: uv sync --all-extras --dev
      - name: Run tests
        run: uv run pytest tests
```

✅ **Installs UV**  
✅ **Syncs dependencies**  
✅ **Runs unit tests using Pytest**  

---

## **🐳 Building a Docker Image with UV**
A **well-built Docker image** ensures consistent deployment across environments. UV simplifies **containerization**.

### **Dockerfile for AceBet**
```dockerfile
FROM python:3.12-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container
COPY . /app

# Set working directory
WORKDIR /app

# Install dependencies
RUN uv sync --frozen --no-cache

# Run the FastAPI app
CMD ["/app/.venv/bin/fastapi", "run", "src/acebet/app/main.py", "--port", "80", "--host", "0.0.0.0"]
```

For **production-ready builds**, use a **multi-stage Docker build** to keep the image **lightweight**.

## **✅ Canonical (Known-Good) Commands**
The command sets below were verified against this repository layout (`src/acebet/app/main.py`, `tests/test_acebet.py`, and `src/acebet/train/train.py`).

### 1) API startup + health check (local)
```bash
# from repository root
uv sync
uv run fastapi run src/acebet/app/main.py --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/
```

### 2) API startup + health check (container)
```bash
# from repository root
docker build -t acebet:local .
docker run --rm -p 8000:80 acebet:local
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:8000/
```

### 3) Test run
```bash
uv run pytest tests
```

### 4) Model training (module exposed in repo)
```bash
uv run python -m acebet.train.train
```

### 5) Prediction endpoint call (token + authenticated request)
```bash
# 1) fetch token
TOKEN=$(curl -fsS -X POST http://127.0.0.1:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=johndoe&password=secret' | python -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

# 2) call prediction endpoint
curl -fsS -X POST http://127.0.0.1:8000/predict/ \
  -H "Authorization: Bearer ${TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"p1_name":"Fognini F.","p2_name":"Jarry N.","date":"2018-03-04","testing":true}'
```

---

## **🌟 Why UV for MLOps?**
| **Feature**                 | **UV** 🚀 (Rust-based) | **Poetry** 🛠️ (Python-based) |
|-----------------------------|------------------------|------------------------------|
| **⏩ Performance & Speed**   | ⚡ **8-10x faster** than pip & pip-tools (80-115x with cache). Ideal for **CI/CD**. | 🐢 Slower dependency resolution and package installation. Can be a bottleneck in large projects. |
| **🔧 Dependency Management** | Uses a **global cache** to avoid redundant installations. Faster and more efficient. | Uses a **custom resolver**, but can be slow for large projects. |
| **📦 Environment Handling** | **Manages Python versions** natively (no need for pyenv). Creates fast, lightweight virtual environments. | Supports virtual environments but **requires external tools** for Python version management. |
| **🐳 Docker Efficiency** | ✅ **Smaller images** & faster builds. **Simplifies deployment** by combining Python & dependency management. | ❌ **Larger image footprint** due to reliance on multiple tools. Longer build times. |
| **🚀 CI/CD Pipelines** | ✅ **Faster builds** due to Rust-based optimizations. Reduces install time in GitHub Actions, Docker, and cloud environments. | ❌ **Slower CI/CD performance** due to Python-based dependency resolution. |
| **🔄 Migration & Ecosystem** | ✅ **Follows PEP standards closely**, making migration easier. **Less tightly integrated**, offering flexibility. | ❌ More **opinionated ecosystem**, making migration or integration with existing tools more complex. |
| **🔑 Authentication & Config** | ✅ Simplifies authentication using **environment variables**. Ensures **cross-platform consistency**. | ❌ Configuration can be complex, requiring additional setup for cross-platform consistency. |
| **📜 Unified Tooling** | ✅ **All-in-one tool**: Handles package management, virtual environments, and Python versions. **No need for extra tools**. | ❌ **Depends on multiple tools** (e.g., `pyenv` for Python versioning), increasing setup complexity. |
| **🏗️ Build & Deployment** | ✅ **Optimized for modern workflows**. Generates smaller wheels and installs faster in **Docker, Kubernetes, and cloud deployments**. | ❌ Traditional package builds, **not as optimized** for modern DevOps/MLOps pipelines. |

### **🎯 Key Takeaways**
- If you need **speed, lightweight builds, and a streamlined DevOps workflow**, **UV is the better choice**. 🚀  
- If you prefer **a well-established but slower tool with more integrated features**, **Poetry remains viable**. 🛠️  
- **For MLOps & CI/CD**, **UV's speed and efficiency make it the preferred option**. 💡  

---

## **🎯 Conclusion**
By integrating **UV** into your **MLOps workflow**, you get a **fast, reproducible, and efficient setup** for managing **dependencies, testing, and deployment**.

With **AceBet**, we demonstrated how to:
✔️ **Initialize a UV-based project**  
✔️ **Manage dependencies & lockfiles**  
✔️ **Automate testing with GitHub Actions**  
✔️ **Build Docker images for deployment**  

**Give UV a try**—it might just **replace Pip and Poetry** in your workflow! 🚀  

**Happy Coding!**


## Documentation

- Local docs guide: [`docs/getting-started.md`](docs/getting-started.md)
- Local preview command: `zensical serve`
- Docs contribution flow: update Markdown in `docs/`, run `zensical build --clean`, preview with `zensical serve`, then open a PR.
