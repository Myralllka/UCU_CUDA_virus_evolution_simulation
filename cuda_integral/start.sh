#!/bin/bash

config_filename=../execution.conf
while true; do
  case $1 in
    -c|--compile)
      compile=true;
      shift
      ;;
    -t|--threads)
      test -z "${OPTARG//[0-9]}"  || (echo "$1 is not a number" >&2 && exit 1)
      threads=$2
      shift 2
      ;;
    -f|--file)
      config_filename=$2
      shift 2
    ;;
    -D | --debug-build)
        debug_build=true
        shift
        ;;
    -d|--debug)
      run_debug_build=true;
      shift
    ;;
    -h|--help)
      echo "Usage: ./start.sh [options]
  Options:
    -c    --compile       Compile before executing
    -h    --help          Show help message
    -t    --threads       Number of threads
    -d    --debug         Debuging mode
    -f    --file          Path to the configuration file"
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

####################### Program Input Arguments process #####################
if [[ ! -f "$config_filename" ]]; then
  echo "No conf file: use -f [filename] to identify it.";
  exit 1;
fi
if [[ ! -z "$threads" ]]; then
  sed -i "s/flow_num...*/flow_num = $threads/g" execution.conf;
else
  sed -i "s/flow_num...*/flow_num = 1/g" execution.conf;
fi


#################################### Build ##################################
if [[ "$debug_build" == true ]]; then
  mkdir -p ./cmake-build-debug
  ehco -n "Entering "
  pushd ./cmake-build-debug || exit 1
  echo Compiling...
  cmake -lpng -DCMAKE_BUILD_TYPE=Debug -G"Unix Makefiles" .. || exit 1
  make || exit 1
  popd
else if [[ "$compile" == true  ]]; then
  mkdir -p ./cmake-build-release
  ehco -n "Entering "
  pushd ./cmake-build-release || exit 1
  echo Compiling...
  cmake -DCMAKE_BUILD_TYPE=Release -G"Unix Makefiles" .. || exit 1
  make || exit 1
  popd
fi; fi


###################################### Run ####################################
if [[ "$debug_build" == true -o "$run_debug_build" == true ]]; then
    optirun ./cmake-build-debug/mpi_heat_transfer "$config_filename" || exit 1
else
    optirun ./cmake-build-release/mpi_heat_transfer "$config_filename" || exit 1
fi
if [[ -z "$debug" ]]; then
  echo "Result is: $(cat tmp/array_result | head -n 1)"
fi;
