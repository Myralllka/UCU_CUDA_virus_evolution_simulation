#include <iostream>


#define gpuErrorCheck(ans); { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

int main(int argc, char *argv[]) {
//    int deviceCount = 0;
//    gpuErrorCheck(cudaGetDeviceCount(&deviceCount));
//    int dev, driverVersion = 0, runtimeVersion = 0;
//    for (dev = 0; dev < deviceCount; ++dev) {
//        gpuErrorCheck(cudaSetDevice(dev));
//        cudaDeviceProp deviceProp;
//        gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, dev));
//        std::cout << "Name: "<< deviceProp.name << std::endl;
//        gpuErrorCheck(cudaDriverGetVersion(&driverVersion));
//        gpuErrorCheck(cudaRuntimeGetVersion(&runtimeVersion));
// deviceProp.name; deviceProp.totalGlobalMem;
// deviceProp.multiProcessorCount −− SM
// deviceProp.major, deviceProp.minor −− CC
// _ConvertSMVer2Cores(deviceProp.major,
// deviceProp.minor) −− CUDA cores per SM
// deviceProp.clockRate −− in Hz

// CUDA 5.0+
// deviceProp.memoryClockRate −− in Hz
// deviceProp.memoryBusWidth −− bits
// deviceProp.l2CacheSize
//
// deviceProp.totalConstMem
// deviceProp.sharedMemPerBlock
// deviceProp.regsPerBlock
// deviceProp.warpSize
// deviceProp.maxThreadsPerMultiProcessor
// deviceProp.maxThreadsPerBlock
// deviceProp.maxThreadsDim[0/1/2]
// deviceProp.maxGridSize[0/1/2]

//    std::cout << "Time = " << std::endl;

    return 0;
}
