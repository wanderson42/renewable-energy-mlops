# The exact base already used by the operational and reproduction servers.
FROM ghcr.io/mlflow/mlflow@sha256:6af326440af64565c2ea5d1d28c9ee355f70a8d4af5b76e35938365e1ce2e428

WORKDIR /opt/energy-mlops-mlflow
COPY requirements.txt verify_runtime.py ./

# Preserve all other packages from the immutable base. Do not resolve newer
# transitive dependencies or compile native libraries during the build.
RUN python3 -m pip install --no-cache-dir --no-deps --only-binary=:all: -r requirements.txt \
    && python3 -m pip check \
    && python3 verify_runtime.py > verified-runtime.json \
    && python3 -m pip list --format=json > installed-packages.json

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_INDEX=1

# The chart will supply backend/artifact settings when this image is adopted.
ENTRYPOINT []
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
