#!/bin/bash

exec_name="cuda_simulation"
config_filename=execution.conf
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
    -e|--executable)
        exec_name=$2
        shift 2
        ;;
    -h|--help)
      echo "Usage: ./start.sh [options]
  Options:
    -c    --compile       Compile before executing
    -t    --threads       Number of threads
    -d    --debug         Run executable with debug symbols
    -D    --debug-build   Build with debug
    -f    --file          Path to the configuration file
    -e    --executable    Executable target name"
    -h    --help          Show help message
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
fi
#################################### Build ##################################
if [[ "$debug_build" == true ]]; then
  mkdir -p ./cmake-build-debug
  echo -n "Entering "
  pushd ./cmake-build-debug || exit 1
  echo Compiling...
  cmake -DCMAKE_BUILD_TYPE=Debug -G"Unix Makefiles" .. || exit 1
  make || exit 1
  popd
else if [[ "$compile" == true  ]]; then
  mkdir -p ./cmake-build-release
  echo -n "Entering "
  pushd ./cmake-build-release || exit 1
  echo Compiling...
  cmake -DCMAKE_BUILD_TYPE=Release -G"Unix Makefiles" .. || exit 1
  make || exit 1
  popd
fi; fi

###################################### Run ####################################
if [[ "$debug_build" == true ]] || [[ "$run_debug_build" == true ]]; then
    echo "DEBUG RUN"
    optirun ./cmake-build-debug/${exec_name} "$config_filename" || exit 1
else
    optirun ./cmake-build-release/${exec_name} "$config_filename" || exit 1
fi
