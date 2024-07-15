


#include <iostream>
#include <random>
#include <iomanip> 
#include "cuda_compute_engine.cuh"

#define kilobyte 1000/ sizeof(float)
#define megabyte 1000 * kilobyte
#define gigabyte 1000 * megabyte

// Number of elements
#define TEST_SIZE gigabyte

typedef void (*void_func)(float* x, float* y, float* z);
typedef float(*test_func)(float* x, float* y, size_t index);



std::random_device rd;
std::mt19937 generator(rd()); // Mersenne Twister engine
std::uniform_real_distribution<> distribution(-100, 100); // Range from 1 to 100


void generate_array(float* buffer, size_t num_elements, size_t offset)
{
	#pragma omp parallel for
	// Assign some elements. Speed isnt a big deal as this is for debugging
	for (size_t i = 0; i < num_elements; i++)
	{
		buffer[i] = distribution(generator);
	}
}


void print_array(float* buffer, size_t num_elements)
{
	std::cout << std::fixed << std::setprecision(0);
	std::cout << "[ ";
	#pragma omp parallel for
	for (size_t i = 0; i < (num_elements - 1); i++)
	{
		std::cout << buffer[i] << ", ";
	}
	std::cout << buffer[num_elements - 1] << " ]\n";
}



void test_assert(float* x, float* y, float* out, size_t size, test_func test_function)
{
	std::cout << "Testing function located at " << reinterpret_cast<void*>(test_function) << '\n';

	float* expected_result = new float[TEST_SIZE];

	// Assign some elements. Speed isn't a big deal as this is for debugging
	#pragma omp parallel for
	for (size_t i = 0; i < size; i++)
	{
		expected_result[i] = test_function(x, y, i);
	}

	for (size_t i = 0; i < size; i++)
	{
		if (!(test_function(x, y, i) == out[i]) )
		{
			std::cerr << "TEST FAILED\n";
			std::cerr << "-------------------------------------------------------------------------------------------------------------------------------------\n";
			std::cerr << "X: \t" << x[i] << '\n';
			std::cerr << "Y: \t" << y[i] << '\n';
			std::cerr << "OUT: \t" << out[i] << '\n';
			std::cerr << "EXPECTED VALUE: \t" << expected_result[i] << '\n';
			std::cerr << "INDEX: \t" << i << '\n';
			std::cerr << "-------------------------------------------------------------------------------------------------------------------------------------\n";

			std::cerr << "BUFFERS:\n";
			std::cerr << "x:               ";
			print_array(x, TEST_SIZE);
			std::cerr << "y:               ";
			print_array(y, TEST_SIZE);
			std::cerr << "OUT:             ";
			print_array(out, TEST_SIZE);
			std::cerr << "EXPECTED OUTPUT: ";
			print_array(expected_result, TEST_SIZE);
			std::cerr << "-------------------------------------------------------------------------------------------------------------------------------------\n";


			exit(1);

		}
	}
	
}

void test(float* x, float* y, float* z, void_func void_function)
{
	std::cout << "Speed test of function " << reinterpret_cast<void*>(void_function) << "...";
	// ----------------------------------------------------
		// CUDA timing variables
	cudaEvent_t start, stop;
	float elapsedTime;

	// Create CUDA events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start timing
	cudaEventRecord(start);


	// Run the test
	void_function(x, y, z);

	// Stop timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Calculate elapsed time
	cudaEventElapsedTime(&elapsedTime, start, stop);
	// ----------------------------------------------------
	std::cout << "Test ran in  " << (int)elapsedTime << " milliseconds.\n";
}

void add_CUDA(float* x, float* y, float* z)
{
	CUDA_compute_engine::add(x, y, z, TEST_SIZE);
}

void add_CUDA2(float* x, float* y, float* z)
{
	CUDA_compute_engine::add(x, y, TEST_SIZE);
}

float add(float* x, float* y, size_t index)
{
	return x[index] + y[index];
}

int main()
{

	float* x = new float[TEST_SIZE];
	float* y = new float[TEST_SIZE];
	float* z = new float[TEST_SIZE];
	

	std::cout << "Done!\n";
	std::cout << "testing validity...";
	test_assert(x, y, z, TEST_SIZE, add);
	std::cout << "Done!\n\n";
	// print the output
	test(x, y, z, add_CUDA);
	test(x, y, z, add_CUDA2);

}