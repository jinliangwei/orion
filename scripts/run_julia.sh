#!/bin/bash -u

tcmalloc_lib="/usr/lib/libtcmalloc.so"

if [ "$#" != "2" ]; then
    echo "usage: $0 [julia-executable] [julia-script]"
    exit
fi

LD_PRELOAD=$tcmalloc_lib $1 $2
