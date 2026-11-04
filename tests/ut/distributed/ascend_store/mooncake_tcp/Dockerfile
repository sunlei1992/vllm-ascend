FROM python:3.12-slim-trixie

RUN apt-get update -qq \
    && apt-get install -y -qq libcurl4 libibverbs1 rdma-core librdmacm1 libnuma1 liburing2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mooncake-transfer-engine-non-cuda

WORKDIR /app
