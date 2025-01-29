#!/bin/sh

docker compose up -d 
docker exec -it ollama ollama pull llama3.2:1b
docker exec -it ollama ollama serve