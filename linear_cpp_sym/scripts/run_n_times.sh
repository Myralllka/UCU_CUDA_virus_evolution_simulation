#!/bin/zsh

for i in $(seq $1); do
 ./cmake-build-debug/linear_cpp_sym > "res/snap$i.txt" &
done;