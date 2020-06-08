#!/bin/zsh

start=0
step=50000
stop=10

dir="res/res0"
mkdir -p $dir
for cycles in $(seq $stop); do
  dir="res/res$start"
  mkdir -p $dir
  sed -i "s/isol_place...*/isol_place = $start/g" ./simulation.conf
  for i in $(seq $1); do
     ./cmake-build-debug/linear_cpp_sym > "$dir/snap$i.txt" &
    sleep 1s
  done
  ((start += $step))
  echo "waiting$start..."
done
echo waiting...
wait
