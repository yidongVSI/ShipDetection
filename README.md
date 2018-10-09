# ShipDetection
Scripts for ship detection

## Require
```python
pytorch pandas PIL numpy matplotlib
```

## Basic Usage
expt notebook is used to illustrate usage of vaious implemented modules

## Run maskrcnn on docker
1. Prerequist
nvidia-driver on host machine
docker
nvidia-docker
docker-compose (optional)

2. Build image
```bash
cd ./docker
docker build -t maskrcnn:latest .
```

3. Run
Running Jupyter Notebook
```
docker run -it -p 8888:8888 -p 6006:6006 -v ~/:/host maskrcnn:latest jupyter notebook --allow-root /host
```
