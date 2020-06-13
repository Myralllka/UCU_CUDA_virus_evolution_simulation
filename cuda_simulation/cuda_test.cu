//
// Created by fenix on 6/13/20.
//
#include <iostream>
#include <cuda_assert.cuh>


int main(int argc, char *argv[]) {
    int deviceCount = 0;gpuErrorCheck(cudaGetDeviceCount(&deviceCount));
    int dev, driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev) {
        // choose device
        gpuErrorCheck(cudaSetDevice(dev));
        cudaDeviceProp deviceProp{};

        gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, dev));
        std::cout << "Name:\t\t\t\t\t\t\t" << deviceProp.name << std::endl;

        gpuErrorCheck(cudaDriverGetVersion(&driverVersion));gpuErrorCheck(cudaRuntimeGetVersion(&runtimeVersion));
        std::cout << "totalGlobalMem\t\t\t\t\t" << deviceProp.totalGlobalMem / 1'000'000 << " MiB" << std::endl;
        std::cout << "multiProcessorCount\t\t\t\t" << deviceProp.multiProcessorCount << std::endl; // −− SM
        std::cout << "minor\t\t\t\t\t\t\t" << deviceProp.minor << std::endl; // −− CC
        std::cout << "major\t\t\t\t\t\t\t" << deviceProp.major << std::endl;
//        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) //−− CUDA cores per SM

        std::cout << "\nclockRate\t\t\t\t\t\t" << deviceProp.clockRate / 1000 << " kHz" << std::endl;
        std::cout << "memoryClockRate\t\t\t\t\t" << deviceProp.memoryClockRate / 1000 << " kHz" << std::endl;
        std::cout << "memoryBusWidth\t\t\t\t\t" << deviceProp.memoryBusWidth << " b" << std::endl; // −− bits
        std::cout << "l2CacheSize\t\t\t\t\t\t" << deviceProp.l2CacheSize << std::endl;

        std::cout << "\ntotalConstMem\t\t\t\t\t" << deviceProp.totalConstMem << std::endl;
        std::cout << "sharedMemPerBlock\t\t\t\t" << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "regsPerBlock\t\t\t\t\t" << deviceProp.regsPerBlock << std::endl;
        std::cout << "warpSize\t\t\t\t\t\t" << deviceProp.warpSize << std::endl;
        std::cout << "maxThreadsPerMultiProcessor\t\t" << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "maxThreadsPerBlock\t\t\t\t" << deviceProp.maxThreadsPerBlock << std::endl;

        std::cout << "\nmaxThreadsDim 1\t\t\t\t\t" << deviceProp.maxThreadsDim[0] << std::endl;
        std::cout << "maxThreadsDim 2\t\t\t\t\t" << deviceProp.maxThreadsDim[1] << std::endl;
        std::cout << "maxThreadsDim 3\t\t\t\t\t" << deviceProp.maxThreadsDim[2] << std::endl;

        std::cout << "\nmaxGridSize 1\t\t\t\t\t" << deviceProp.maxGridSize[0] << std::endl;
        std::cout << "maxGridSize 2\t\t\t\t\t" << deviceProp.maxGridSize[1] << std::endl;
        std::cout << "maxGridSize 3\t\t\t\t\t" << deviceProp.maxGridSize[2] << std::endl;
    }
    return 0;
}
