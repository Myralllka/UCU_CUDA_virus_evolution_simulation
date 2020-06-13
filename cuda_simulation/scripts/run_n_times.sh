#!/bin/bash

start=0
step=40000
stop=26 # num of iters

deadstart=0
deadstep=0.05
deadstop=10 #num of iters

#dir="res/res0"
#mkdir -p $dir
for deadrate in $(seq $deadstop); do
  sed -i "s/patient_to_dead...*/patient_to_dead = $deadstart/g" ./simulation.conf
  for cycles in $(seq $stop); do
    dir="RES/res$deadstart/res$start"
    mkdir -p $dir
    sed -i "s/isol_place...*/isol_place = $start/g" ./simulation.conf
    for i in $(seq $1); do
       ./cmake-build-debug/linear_cpp_sym > "$dir/snap$i.txt" &
      sleep 1s
    done
    ((start += $step))
  done
  ((deadstart += $deadstep))
  echo "waiting$deadstart..."
  start=0
done
echo waiting...
wait
