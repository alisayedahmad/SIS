# Reproducible environment with GDAL/GEOS for rasterio & geopandas
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev \
        libgl1-mesa-glx libglib2.0-0 \
        build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

# Pre-install wheels then project
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /workspace

# Default: show help
CMD ["python", "-m", "src.cli.train", "--help"]
