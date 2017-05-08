#!/bin/bash -u

tcmalloc_lib="/usr/lib/libtcmalloc.so"

if [ "$#" != "1" ]; then
    echo "usage: $0 [julia-executable]"
    exit
fi

LD_PRELOAD=$tcmalloc_lib $1
