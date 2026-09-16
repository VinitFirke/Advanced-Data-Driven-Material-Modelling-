# Lightweight image for the ROM identification CLI + Streamlit demo app.
# No FEniCSx/dolfinx here (that stack is conda-only, ~GB-scale) -- the ROM
# identification path only needs numpy/scipy/scikit-learn, so this builds fast
# and is what you'd actually deploy (Streamlit Cloud, HF Spaces, a container
# host, ...). To reproduce the full FOM path instead, use the conda env in
# environment.yml directly rather than Docker.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY data/sample/ data/sample/
COPY app/ app/

RUN pip install --no-cache-dir ".[app]"

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
