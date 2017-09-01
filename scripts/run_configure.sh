#!/bin/bash -u
JULIA_HOME=/home/ubuntu/julia-0.5.1/ \
	  JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
	  HADOOP_HOME=/home/ubuntu/hadoop-2.7.3/ \
	  ./configure \
	  --enable-sanitizer=none
