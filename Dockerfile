# The base image is build from .devcontainer/Dockerfile
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

ADD judges /auto-judge/judges
ADD pyproject.toml /auto-judge/

WORKDIR /auto-judge

RUN uv pip install --system -e .[all]

