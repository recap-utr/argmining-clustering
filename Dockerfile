# https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
# https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3/.devcontainer/Dockerfile
# https://github.com/nautobot/nautobot/blob/develop/docker/Dockerfile

ARG POETRY_VERSION=1.1.14
ARG PYTHON_VERSION=3.9

FROM python:${PYTHON_VERSION}

ENV PYTHONUNBUFFERED=1 \
    POETRY_HOME="/poetry" \
    PATH="/poetry/bin:${PATH}"

WORKDIR /app

RUN apt update && \
    apt install -y graphviz && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sSL https://install.python-poetry.org | python - \
    && poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root \
    && poetry run python -m spacy download en_core_web_lg \
    && poetry run python -m spacy download en_core_web_trf
