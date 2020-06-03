#!/bin/bash

while true; do
  case $1 in
    -c|--compile)
      comp=true;
      shift
      ;;
   -f|--file)
      file=true;
      file_name=$2
      shift 2
      ;;
    -h|--help)
      echo "Usage: ./start.sh [options]
  Options:
    -c    --compile       Compile before executing
    -f    --file          Specify file to redirect output"
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


if [[ "$comp" = true || ! -e linear_cpp_sym ]]; then
  echo Compiling...
  cmake -DCMAKE_BUILD_TYPE=Release -G"Unix Makefiles" ..;
  make;
fi;

popd > /dev/null

echo RUN
if [[ "$file" = true ]]; then
  echo "Output to -> ${file_name}"
  ./cmake-build-debug/linear_cpp_sym > ${file_name}
else
    ./cmake-build-debug/linear_cpp_sym
fi;



