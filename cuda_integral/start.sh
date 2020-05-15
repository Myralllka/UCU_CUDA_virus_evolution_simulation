#!/usr/bin/env bash

while true; do
  case $1 in
    -c|--compile)
      comp=true;
      shift
      ;;
    -m|--make)
      make=true;
      shift
      ;;
    -h|--help)
      echo "Usage: ./start.sh [options]
  Options:
    -c    --compile       Compile before executing"
      exit 0;
    ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1 ;;
    :)
      echo "Option -$OPTARG requires an numerical argument." >&2
      exit 1 ;;
    *)
      break
      ;;
  esac
done


mkdir -p ./cmake-build-debug;
pushd ./cmake-build-debug  > /dev/null || exit 1


if [[ "$comp" = true || ! -e cuda_integral ]];
then
  echo Compiling...
  cmake -DCMAKE_BUILD_TYPE=Release -G"Unix Makefiles" ..;
  make;
fi;

if [[ "$make" = true ]];
then
  echo Compiling...
  make;
fi;

popd > /dev/null
optirun cmake-build-debug/cuda_integral execution.conf
