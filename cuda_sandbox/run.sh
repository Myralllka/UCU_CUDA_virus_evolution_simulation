if [ ! -d cmake-build-debug ]
then
  mkdir cmake-build-debug
  cd cmake-build-debug
  cmake -G"Unix Makefiles" ../
  make
  cd ../
fi

optirun cmake-build-debug/cuda_sandbox
