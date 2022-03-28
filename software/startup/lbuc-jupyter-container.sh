#!/usr/bin/env bash

DOCKER=podman
PORT=8888
IMAGE=localhost/lbuc-jupyter:latest

$DOCKER run -p $PORT:8888 -v $(pwd)/:/home/sage/hostnotebooks $IMAGE
