# The base image is build from https://github.com/trec-auto-judge/auto-judge-starter-kit/blob/main/.devcontainer/Dockerfile
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

ADD judges /auto-judge/judges
ADD pyproject.toml /auto-judge/
ADD llm-config.yml /auto-judge/

WORKDIR /auto-judge

RUN uv pip install -e .[all]
# only needed so that we have an additional connection to the git metadata
ADD .git /auto-judge/

