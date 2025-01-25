
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = idx; i < size; i+=stride){
		y[i] = scale * x[i] + y[i];
	}

}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	
	size_t size = vectorSize * sizeof(float);
	float *device_x, *device_y, scale = 2.0f;

	// Initialize host
	std::vector<float> host_x(vectorSize);
	std::vector<float> host_y(vectorSize);
	std::vector<float> host_result(vectorSize);

	// Initialize vector values (vector.data() returns pointer of the first data in the vector)
	vectorInit(host_x.data(), vectorSize);
	vectorInit(host_y.data(), vectorSize);

	// Copy host_y data to host_result vector
	std::memcpy(host_result.data(), host_y.data(), size);

	// Malloc space for x and y in GPU, and use device_x, device_y pointers to point at them
	gpuAssert(cudaMalloc((void **)&device_x, size), __FILE__, __LINE__);
	gpuAssert(cudaMalloc((void **)&device_y, size), __FILE__, __LINE__);

	// Copy the data from host to GPU
	gpuAssert(cudaMemcpy( device_x, host_x.data(), size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
	gpuAssert(cudaMemcpy( device_y, host_result.data(), size, cudaMemcpyHostToDevice), __FILE__, __LINE__);

	// Kernel setup
	int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

	saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(device_x, device_y, scale, vectorSize);

	gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);

	// Computation finished, copy the result from GPU to host
	gpuAssert(cudaMemcpy(host_result.data(), device_y, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	int errorCount = verifyVector(host_x.data(), host_y.data(), host_result.data(), scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";


	gpuAssert(cudaFree(device_x), __FILE__, __LINE__);
    gpuAssert(cudaFree(device_y), __FILE__, __LINE__);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
