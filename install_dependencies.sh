#!/usr/bin/env bash

sudo apt-get update
sudo apt-get --ignore-missing -y install \
    git \
    g++-5 \
    uuid-dev \
    libnuma-dev \
    valgrind \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libsnappy-dev \
    libgtest-dev \
    google-perftools \
    julia

if [ ! -f /usr/lib/libcblas.so ]; then
    echo "link libcblas!!"
    sudo ln -s /usr/lib/libcblas.so.3 /usr/lib/libcblas.so
fi