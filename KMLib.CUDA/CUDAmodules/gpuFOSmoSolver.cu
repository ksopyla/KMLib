/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/

CUDA function for SMO first order solver with 2 kernel columns computed at once
*/

#include <float.h>

__constant__ float C;

__constant__ float B[3]={0, 0, 0};
__constant__ float A[3]={0, 0, 0};

// minimal coeficient
__constant__ float COEF_EPS = 0.000001f;

// constat for kernel diagonal for index i
//__constant__ float QD_i;

// label for i-th example
//__constant__ float Y_i;

//texture for vector, which is used for matrix vector multiplication
//in SVM, when we have to compute many dot products (one vector with others)
texture<float,1,cudaReadModeElementType> mainVectorTexRef;


#define BLOCK_SIZE 128

#define BLOCK_SIZE_RED 128


//define NEG_INFINITY_F __int_as_float(0xff800000)



/*

	Do warp parallel reduction in order to find max value and its index
*/
__device__ void maxWarpReduce(volatile int *volShIdx,volatile float *volShVal,unsigned int tid)
{
		/*
		if (BLOCK_SIZE_RED >=  64) { if( volShVal[tid]< volShVal[tid+32]) {
							 volShVal[tid]=volShVal[tid+32]; volShIdx[tid]=volShIdx[tid+32];	} }
		if (BLOCK_SIZE_RED >=  32) { if( volShVal[tid]< volShVal[tid+16]) {
							 volShVal[tid]=volShVal[tid+16]; volShIdx[tid]=volShIdx[tid+16];	} }
		if (BLOCK_SIZE_RED >=  16) { if( volShVal[tid]< volShVal[tid+8]) {
							 volShVal[tid]=volShVal[tid+8]; volShIdx[tid]=volShIdx[tid+8];	} }
		if (BLOCK_SIZE_RED >=   8) { if( volShVal[tid]< volShVal[tid+4]) {
							 volShVal[tid]=volShVal[tid+4]; volShIdx[tid]=volShIdx[tid+4];	} }
		if (BLOCK_SIZE_RED >=   4) { if( volShVal[tid]< volShVal[tid+2]) {
							 volShVal[tid]=volShVal[tid+2]; volShIdx[tid]=volShIdx[tid+2];	} }
		if (BLOCK_SIZE_RED >=   2) { if( volShVal[tid]< volShVal[tid+1]) {
							 volShVal[tid]=volShVal[tid+1]; volShIdx[tid]=volShIdx[tid+1];	} }
*/

		if( volShVal[tid]< volShVal[tid+32]) {
							 volShVal[tid]=volShVal[tid+32]; volShIdx[tid]=volShIdx[tid+32];	} 
		if( volShVal[tid]< volShVal[tid+16]) {
							 volShVal[tid]=volShVal[tid+16]; volShIdx[tid]=volShIdx[tid+16];	} 
		if( volShVal[tid]< volShVal[tid+8]) {
							 volShVal[tid]=volShVal[tid+8]; volShIdx[tid]=volShIdx[tid+8];	} 
		if( volShVal[tid]< volShVal[tid+4]) {
							 volShVal[tid]=volShVal[tid+4]; volShIdx[tid]=volShIdx[tid+4];	}
		if( volShVal[tid]< volShVal[tid+2]) {
							 volShVal[tid]=volShVal[tid+2]; volShIdx[tid]=volShIdx[tid+2];  }
		if( volShVal[tid]< volShVal[tid+1]) {
							 volShVal[tid]=volShVal[tid+1]; volShIdx[tid]=volShIdx[tid+1];	}
}



/*
	Do parallel reduction for finding index "i" which maximize
	// i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes -y_j * grad(f)_j, j in I_down(\alpha)
*/
extern "C" __global__ void FindMaxI_MinJ(const float* y, 
									  const float* alpha, 
									  const float* grad,
									  int * idxReduce, 
									  float* gradReduce,
									  const int N)
{

	__shared__ float shValsI[BLOCK_SIZE_RED];     
	__shared__ int shIdxI[BLOCK_SIZE_RED];

	__shared__ float shValsJ[BLOCK_SIZE_RED];     
	__shared__ int shIdxJ[BLOCK_SIZE_RED];
	
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	//unsigned int i = blockIdx.x*BLOCK_SIZE_RED*2 + threadIdx.x;
	//unsigned int gridSize = BLOCK_SIZE_RED*2*gridDim.x;

	
	unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
	unsigned int gridSize = blockDim.x*2*gridDim.x;
	
	//stores 'i' index
	shValsI[tid]=-FLT_MAX;
	shIdxI[tid]=-1;
   
	//stores 'j' index
	shValsJ[tid]=-FLT_MAX;
	shIdxJ[tid]=-1;

	// we reduce multiple elements per thread.  The number is determined by the 
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	

	float maxGi=-FLT_MAX ,maxGj=-FLT_MAX ;
	float temp1Max=-FLT_MAX, temp2Max=-FLT_MAX ;
	float temp1Min=-FLT_MAX, temp2Min=-FLT_MAX ;
	int yi=0, yib=0;
	float alpha_i=0, alpha_ib=0;
	float grad_i=0, grad_ib=0;

	//N=k*2*blockDim
	while (i < N)
	{   
		yi=(int)y[i];
		

		alpha_i=alpha[i];
		yib = y[i+blockDim.x];
		alpha_ib=alpha[i+blockDim.x];
		grad_i = grad[i];
		grad_ib = grad[i+blockDim.x];

		temp1Max = (yi*alpha_i)<B[yi+1] ? -grad_i*yi:-FLT_MAX;
		temp2Max = (yib*alpha_ib)<B[yib+1] ? -grad_ib*yib:-FLT_MAX;
		
		temp1Min = (yi*alpha_i)>A[yi+1] ? grad_i*yi: -FLT_MAX;
		temp2Min = (yib*alpha_ib)>A[yib+1] ? grad_ib*yib:-FLT_MAX;

		
		maxGi<temp1Max ? shIdxI[tid]=i : 0;
		maxGi = fmaxf(maxGi, temp1Max );

		maxGi<temp2Max ? shIdxI[tid]=i+blockDim.x : 0;
		maxGi = fmaxf(maxGi, temp2Max );


		maxGj<temp1Min ? shIdxJ[tid]=i : 0;
		maxGj = fmaxf(maxGj, temp1Min );

		maxGj<temp2Min ? shIdxJ[tid]=i+blockDim.x : 0;
		maxGj = fmaxf(maxGj,temp2Min);

		
		i += gridSize;
	} 

	// each thread puts its local sum into shared memory 
	shValsI[tid] = maxGi;
	shValsJ[tid] = maxGj;
	__syncthreads();


	if (BLOCK_SIZE >= 512) { 
		if (tid < 256) { 
			if( shValsI[tid]< shValsI[tid+256]) {
							 shValsI[tid]=shValsI[tid+256]; shIdxI[tid]=shIdxI[tid+256];	}
			
			if( shValsJ[tid]< shValsJ[tid+256]) {
							 shValsI[tid]=shValsI[tid+256]; shIdxI[tid]=shIdxI[tid+256];	}
		
		} 
		__syncthreads(); 
	}
	if (BLOCK_SIZE >= 256) { 
		if (tid < 128) { 
			if( shValsI[tid]< shValsI[tid+128]) {
							 shValsI[tid]=shValsI[tid+128]; shIdxI[tid]=shIdxI[tid+128];	}
			if( shValsJ[tid]< shValsJ[tid+128]) {
							 shValsJ[tid]=shValsJ[tid+128]; shIdxJ[tid]=shIdxJ[tid+128];	}
		}
		__syncthreads(); 
	}
	
	if (BLOCK_SIZE >= 128) { 
		if (tid < 64) { 
			if( shValsI[tid]< shValsI[tid+64]) {
							 shValsI[tid]=shValsI[tid+64]; shIdxI[tid]=shIdxI[tid+64];	}


			if( shValsJ[tid]< shValsJ[tid+64]) {
							 shValsJ[tid]=shValsJ[tid+64]; shIdxJ[tid]=shIdxJ[tid+64];	}

		} 
		__syncthreads(); 
	}
	

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		//todo:its possible to merge it
		maxWarpReduce(shIdxI,shValsI,tid);
		maxWarpReduce(shIdxJ,shValsJ,tid);
	}
	
	// write result for this block to global mem 
	if (tid == 0) {
		gradReduce[blockIdx.x] = shValsI[0];
		idxReduce[blockIdx.x] = shIdxI[0];

		gradReduce[blockIdx.x+BLOCK_SIZE_RED] = shValsJ[0];
		idxReduce[blockIdx.x+BLOCK_SIZE_RED] = shIdxJ[0];
	}
	
}






/*

	Updates gradient 

	One threads process 4 gradients, inspired by Volkow http://www.cs.berkeley.edu/~volkov/volkov10-GTC.pdf
Qi - i-th kernel column
Qj - j-th kernel column
grad - gradient
diff_i  - alpha_i-old_alpha_i
diff_j  - alpha_j-old_alpha_j
N       - #rows
*/
extern "C" __global__ void UpdateGrad(const float* Qi, 
									  const float* Qj, 
									  float* grad,
									  float diff_i,
									  float diff_j,
									  const int N)
{
    int iblock = blockIdx.x; //+  gridDim.x*blockDim.x;
    int idx    = threadIdx.x+4*iblock*blockDim.x;
	//acumulators 
	float tempGrad[4];	
	float tempQi[4];	
	float tempQj[4];
	float alpha_i_diff=diff_i;
	float alpha_j_diff=diff_j;	
	//read 4 elements per thread int to register's
	for(int i=0;i<4;i++){
		/*
		tempGrad[i] = (idx+i*blockDim.x <N) ? grad[idx+i*blockDim.x]:0;
		tempQi[i]   = (idx+i*blockDim.x <N) ? Qi[idx+i*blockDim.x]:0;
		tempQj[i]   = (idx+i*blockDim.x <N) ? Qj[idx+i*blockDim.x]:0;
		*/
		if(idx+i*blockDim.x <N)
		{
			tempGrad[i] = grad[idx+i*blockDim.x]; 
			tempQi[i]=Qi[idx+i*blockDim.x]; 
			tempQj[i]=Qj[idx+i*blockDim.x];
		}
		//(idx+i*blockDim.x <N) ? (tempGrad[i] = grad[idx+i*blockDim.x]; tempQi[i]=Qi[idx+i*blockDim.x]; tempQj[i]=Qj[idx+i*blockDim.x]):0;
	}
	
	//do final computation
	for(int i=0;i<4;i++){
		(idx+i*blockDim.x <N) ? (grad[idx+i*blockDim.x]=tempGrad[i]+ alpha_i_diff*tempQi[i]+alpha_j_diff*tempQj[i]) :0;
		
	}
}