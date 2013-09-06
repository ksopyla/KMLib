/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#ifndef KERNELS_CONFIG
#define KERNELS_CONFIG



//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;

texture<float,1,cudaReadModeElementType> mainVecTexRef;
//texture<float,1,cudaReadModeElementType> mainVectorTexRef;

#define BLOCK_SIZE 256

#define WARP_SIZE 32

#define PREFETCH_SIZE 2

#define THREADS_ROW 4

#define VECDIM 1

#define maxNNZ 100


//rho for computing
//__constant__ float RHO=-2;
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



#endif /* KERNELS_CONFIG */