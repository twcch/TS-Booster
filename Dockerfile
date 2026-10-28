# syntax=docker/dockerfile:1.4

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS tslib

WORKDIR /workspace

ARG http_proxy
ARG https_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Same four ordered steps as README "Installation" (requirements/reqs_1..4.txt).
# Step 1 is a no-op here: the base image already ships torch 2.6.0+cu124.
COPY requirements/ requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/reqs_1.txt && \
    pip install -r requirements/reqs_2.txt

# mamba-ssm (cxx11abiFALSE, torch 2.6, wheel chosen by the image's Python version) (models/tslib_models/Mamba.py)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/reqs_3.txt

# uni2ts (--no-deps) (models/tslib_models/Moirai.py)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements/reqs_4.txt

COPY . .

CMD ["bash"]
