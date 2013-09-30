/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#ifndef KERNELS_CONFIG
#define KERNELS_CONFIG

#include <float.h>


//texture for labels associated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;

texture<float,1,cudaReadModeElementType> mainVecTexRef;
//needed for prediction and asynchronous transfers
texture<float,1,cudaReadModeElementType> mainVec2TexRef;



#define BLOCK_SIZE 256

#define BLOCK_SIZE_RED 128

#define WARP_SIZE 32

#define PREFETCH_SIZE 2

#define THREADS_ROW 4

#define VECDIM 1

#define maxNNZ 100



template<int TexSel> __device__ float fetchTex(int idx);

template<> __device__ float fetchTex<1>(int idx) { return tex1Dfetch(mainVecTexRef,idx); }
template<> __device__ float fetchTex<2>(int idx) { return tex1Dfetch(mainVec2TexRef,idx); }


//sets -1 for negative values and 1 for gather or equal than 0
//params:
// inputArray - array
// size - size of inputArray
extern "C" __global__ void setSignForPrediction(float * inputArray,const int size,const float RHO)
{
	__shared__ float shRHO;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(threadIdx.x==0)
		shRHO=RHO;

	if(idx<size)
	{	
		float val = inputArray[idx] - shRHO;
	
		//signbit returns 1 if val is negative, 0 otherwise
		//if val=-1,8 
		//signbit(val)==1,
		// -(1*2-1) = -1
		//if val==1.8
		//signbit(val)=0
		//-(0*2-1)=1

		inputArray[idx] = -(signbit(val)*2-1);
	}
}





extern "C" __global__ void reduce(const float* input, float* output, const int N)
{

	__shared__ float shVals[BLOCK_SIZE_RED];     
	
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	unsigned int gridSize = blockDim.x*2*gridDim.x;
	
	shVals[tid]=0;   
	float sum=0;

	while (i < N)
	{   
		sum+=input[i];
		
		// ensure we don't read out of bounds 
		if (i + BLOCK_SIZE_RED < N) {
			sum+=input[i+BLOCK_SIZE_RED];
		}
		i += gridSize;
	} 

	// each thread puts its local sum into shared memory 
	shVals[tid] = sum;
	__syncthreads();
	
	if (BLOCK_SIZE_RED >= 512) { 
		if (tid < 256) { 
			shVals[tid]=sum=sum+ shVals[tid+256]; 
		} 
		__syncthreads(); 
	}
	if (BLOCK_SIZE_RED >= 256) { 
		if (tid < 128) { 
			shVals[tid]=sum=sum+ shVals[tid+128]; 
		} 
		__syncthreads(); 
	}
	if (BLOCK_SIZE_RED >= 128) { 
		if (tid < 64) { 
			shVals[tid]=sum=sum+ shVals[tid+64]; 
		} 
		__syncthreads(); 
	}
	

	if (tid < 32)
	{
		 volatile float *smem = shVals;

		 smem[tid] = sum = sum + smem[tid + 32];
		 smem[tid] = sum = sum + smem[tid + 16];
		 smem[tid] = sum = sum + smem[tid +  8];
		 smem[tid] = sum = sum + smem[tid +  4];
		 smem[tid] = sum = sum + smem[tid +  2];
		 smem[tid] = sum = sum + smem[tid +  1];
	}
	
	// write result for this block to global mem 
	if (tid == 0) {
		output[blockIdx.x] = shVals[0];
	}
	
}


#endif /* KERNELS_CONFIG */