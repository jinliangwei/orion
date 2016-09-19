#!/usr/bin/env bash

THIRD_PARTY_HOME="/home/jinliang/orion.git/third_party/" \
./configure --enable-debug=no --enable-sanitizer=address --enable-perfcount=no
