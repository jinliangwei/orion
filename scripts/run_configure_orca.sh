#!/bin/bash -u
JULIA_HOME=/users/jinlianw/orion.git/julia-0.6.2/ \
	  JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
	  HADOOP_HOME=/users/jinlianw/hadoop-2.7.3/ \
	  ./configure \
	  --enable-sanitizer=none \
	  --enable-debug=no \
	  --enable-gprof=no \
	  --enable-googleprof=no
