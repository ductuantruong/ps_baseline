# Prepepration

copy .dockerignore and Dockerfile to the main dir

```
cp .dockerignore Dockerfile ../
```

# Build docker image

## build an image

```
docker build -t abxregistry.azurecr.io/ps_baseline:latest -f Dockerfile .
```

## upload to azure registry (optional)
```
docker login -u <push-token> -p <token> abxregistry.azurecr.io
docker push abxregistry.azurecr.io/ps_baseline:latest
```

# Run the system

1. Create a new dir with any name (e.g ps_baseline_sys) wiht the following structure. Get docker-compose.yml under ./docker dir.

.
├── ckpt
	└── e17_devEER1.163_devmAP0.614.pth
├── docker-compose.yml
├── input
├── output
└── README.md

2. Run the ps_baseline docker image

```
docker-compose -f docker-compose.yml up
```

3. Copy test files under ./input dir. The output results will be generated under ./output dir.


