# The base image is build from https://github.com/trec-auto-judge/auto-judge-starter-kit/blob/main/.devcontainer/Dockerfile
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

ADD judges /auto-judge/judges
ADD pyproject.toml /auto-judge/
ADD llm-config.yml /auto-judge/

WORKDIR /auto-judge

# Install into the base image's /venv (its PATH runs /venv/bin) — a --system
# or ambiguous install would be invisible at runtime.
RUN . /venv/bin/activate && uv pip install -e .[all]

# git metadata for provenance (tira's runtime stats look for a repo at ./)
ADD .git /auto-judge/.git

