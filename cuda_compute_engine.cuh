#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_PTR float*
#define CUDA_BLOCK_SIZE 256

inline __global__ void addKernel(const float* x, const float* y, float* output)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	output[idx] = x[idx] + y[idx];
}
inline __global__ void addKernel_inplace(float* x, const float* y)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	x[idx]+= y[idx];
}

inline __global__ void subKernel(const float* x, const float* y, float* output)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	output[idx] = x[idx] - y[idx];
}
inline __global__ void subKernel_inplace(float* x, const float* y)
{
	int idx = threadIdx.x - blockIdx.x * blockDim.x;
	x[idx] -= y[idx];
}

inline __global__ void mulKernel(const float* x, const float* y, float* output)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	output[idx] = x[idx] * y[idx];
}
inline __global__ void mulKernel_inplace(float* x, const float* y)
{
	int idx = threadIdx.x * blockIdx.x * blockDim.x;
	x[idx] *= y[idx];
}

inline __global__ void divKernel(const float* x, const float* y, float* output)
{
	int idx = threadIdx.x * blockIdx.x * blockDim.x;
	output[idx] = x[idx] / y[idx];
}
inline __global__ void subKernel_inplace(float* x, const float* y)
{
	int idx = threadIdx.x - blockIdx.x * blockDim.x;
	x[idx] /= y[idx];
}

// more like a namespace but im lazy lmao
typedef struct CUDA_compute_engine
{
	static void add(float* x, float* y, float* out, size_t num_elements);

	// Stores
	static void add(float* x, float* y, size_t num_elements);

	//static void add(float* combined_array, size_t num_elements);


};
void CUDA_compute_engine::add(float* x, float* y, float* out, size_t num_elements)
{
	// Create buffers on the GPU
	CUDA_PTR CUDA_x;
	CUDA_PTR CUDA_y;
	CUDA_PTR CUDA_out;
	cudaMalloc(&CUDA_x, num_elements * sizeof(float));
	cudaMalloc(&CUDA_y, num_elements * sizeof(float));
	cudaMalloc(&CUDA_out, num_elements * sizeof(float));

	// Copy data to the GPU
	cudaMemcpy(CUDA_x, x, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_y, y, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	// Get the number of blocks and leftover threads
	int num_blocks = (num_elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
	int leftover_threads = num_elements - (num_blocks - 1) * CUDA_BLOCK_SIZE;

	addKernel << <num_blocks, CUDA_BLOCK_SIZE >> > (CUDA_x, CUDA_y, CUDA_out);

	size_t offset = (num_blocks - 1) * CUDA_BLOCK_SIZE;
	if (leftover_threads > 0)
	{
		addKernel << <1, leftover_threads >> > (CUDA_x + offset, CUDA_y + offset, CUDA_out + offset);
	}
	cudaMemcpy(out, CUDA_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(CUDA_x);
	cudaFree(CUDA_y);
	cudaFree(CUDA_out);

}

void CUDA_compute_engine::add(float* x, float* y, size_t num_elements)
{
	// Create buffers on the GPU
	CUDA_PTR CUDA_x;
	CUDA_PTR CUDA_y;
	cudaMalloc(&CUDA_x, num_elements * sizeof(float));
	cudaMalloc(&CUDA_y, num_elements * sizeof(float));

	// Copy data to the GPU
	cudaMemcpy(CUDA_x, x, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_y, y, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	// Get the number of blocks and leftover threads
	int num_blocks = (num_elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
	int leftover_threads = num_elements - (num_blocks - 1) * CUDA_BLOCK_SIZE;

	addKernel_inplace << <num_blocks, CUDA_BLOCK_SIZE >> > (CUDA_x, CUDA_y);

	size_t offset = (num_blocks - 1) * CUDA_BLOCK_SIZE;
	if (leftover_threads > 0)
	{
		addKernel_inplace << <1, leftover_threads >> > (CUDA_x + offset, CUDA_y + offset);
	}
	cudaMemcpy(x, CUDA_x, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(CUDA_x);
	cudaFree(CUDA_y);

}