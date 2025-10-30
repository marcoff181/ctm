#!/bin/bash

IMAGES_DIR="./challenge_images"
for img in "$IMAGES_DIR"/*.bmp; do
    python crispy_embedder.py 5.0 "$(basename "$img")"
done