#!/usr/bin/env bash

sudo apt-get install g++-5 libnuma-dev libblas-dev liblapack-dev libatlas-base-dev -y
ls /usr/lib

if [ ! -f /usr/lib/libcblas.so ]; then
    echo "link libcblas!!"
    sudo ln -s /usr/lib/libcblas.so.3 /usr/lib/libcblas.so
fi

git clone git@github.com:jinliangwei/third_party.git

cd third_party; make third_party_core; make eigen
