#!/usr/bin/bash

### Gather and compress the demo application files
# tar -czvf demo-app.tar.gz app.py requirements.txt config.yml

### Build the Docker image
docker build -t demo-app .

### Run the Docker container
docker run --rm -v $(pwd):/app -p 8501:8501 demo-app