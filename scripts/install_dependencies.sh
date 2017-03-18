#!/usr/bin/env bash

# This installs dependencies on Ubuntu 16.04

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
    cmake \
    libgoogle-perftools-dev \
    libgl1-mesa-glx \
    openjdk-8-jdko \
    libprotobuf-dev \
    emacs \
    autoconf

if [ ! -f /usr/lib/libcblas.so ]; then
    echo "link libcblas!!"
    sudo ln -s /usr/lib/libcblas.so.3 /usr/lib/libcblas.so
fi

pushd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp *.a /usr/lib
popd

wget https://github.com/JuliaLang/julia/releases/download/v0.5.1/julia-0.5.1.tar.gz
tar xvzf julia-0.5.1.tar.gz
julia-0.5.1; make -j4; cd
