/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

/*
//texture for vector, which is used for matrix vector multiplication
//in SVM, when we have to compute many dot products (one vector with others)
texture<float,1,cudaReadModeElementType> mainVecTexRef;
//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define PREFETCH_SIZE 2
*/
#include <Config.h>

//texture for vector, which is used for evaluation procedure
//in SVM prediction, its stores one dense support vector
texture<float,1,cudaReadModeElementType> svTexRef;




/******************************************************************
 *
 *			Cuda Kernels for SVM kernels
 */

template<int TexSel> __device__ void SpMV_CSR(const float * vals,
	const int * colIdx, 
	const int * vecPointers,
	const int row_start,
	const int row_end,
	const int row,
	const int num_rows,
	volatile float* shDot);

template<int TexSel> __device__ void SpMV_CSR_nChi2(const float * vals,
	const int * colIdx, 
	const int * vecPointers,
	const int row_start,
	const int row_end,
	const int row,
	const int num_rows,
	volatile float* shChi);

//cuda kernel function for computing SVM RBF kernel, uses 
// CSR format for storing sparse matrix, labels are in texture cache, 
//Remarks: based on spmv_csr_vector_kernel from publication above
//Params:
//vals - array of vectors values
//idx  - array of vectors indexes in CSR format
//vecPointers -array of pointers(indexes) to idx and vals array to specific vectors
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors, stored in CSR matrix format, each vector is stored in one row of matrix
//mainVecIndex - main vector index, needed for retrieving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfCsrFormatKernel(const float * vals,
									   const int * colIdx, 
									   const int * vecPointers, 
									   const float* selfDot,
									   float * results,
									   const int num_rows,
									   const int mainVecIndex,
									   const float gamma)
{
	__shared__ float shDot[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ float shGamma;
	__shared__ int shMainVecIdx;
	__shared__ float shMainSelfDot;
	__shared__ float shLabel;
	
	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}	
	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < num_rows; row += num_warps){

		// use two threads to fetch vecPointers[row] and vecPointers[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];            //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = vecPointers[row+1];


		SpMV_CSR<1>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shDot);
		// first thread writes warp result
		if (thread_lane == 0){
			results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*shDot[threadIdx.x]));
		}
	}
}



//cuda kernel function for computing SVM RBF kernel, uses 
// CSR format for storing sparse matrix, labels are in texture cache, 
//Remarks: based on spmv_csr_vector_kernel from publication above
//Params:
//vals - array of vectors values
//idx  - array of vectors indexes in CSR format
//vecPointers -array of pointers(indexes) to idx and vals array to specific vectors
//results - array of results Linear Kernel
//num_rows -number of vectors, stored in CSR matrix format, each vector is stored in one row of matrix
//mainVecIndex - main vector index, needed for retrieving its label
extern "C" __global__ void nChi2_CSR(const float * vals,
	const int * colIdx, 
	const int * vecPointers, 
	float * results,
	const int num_rows,
	const int mainVecIndex)
{
	__shared__ float shDot[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	
	__shared__ int shMainVecIdx;
	__shared__ float shLabel;

	shDot[threadIdx.x]=0.0;

	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}	
	__syncthreads();


	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < num_rows; row += num_warps){

		// use two threads to fetch vecPointers[row] and vecPointers[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];            //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = vecPointers[row+1];
		
		SpMV_CSR_nChi2<1>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shDot);
		// first thread writes warp result
		if (thread_lane == 0){
			//results[row]= shDot[threadIdx.x];
			results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*shDot[threadIdx.x];
		}
	}
}


//cuda kernel function for computing SVM RBF kernel, uses 
// CSR format for storing sparse matrix, labels are in texture cache, 
//Remarks: based on spmv_csr_vector_kernel from publication above
//Params:
//vals - array of vectors values
//idx  - array of vectors indexes in CSR format
//vecPointers -array of pointers(indexes) to idx and vals array to specific vectors
//rowSum - array of precomputed self row sum
//results - array of results Linear Kernel
//num_rows -number of vectors, stored in CSR matrix format, each vector is stored in one row of matrix
//mainVecIndex - main vector index, needed for retrieving its label
//gamma - gamma parameter  
extern "C" __global__ void expChi2_CSR(const float * vals,
	const int * colIdx, 
	const int * vecPointers, 
	const float* rowSum,
	float * results,
	const int num_rows,
	const int mainVecIndex,
	const float gamma)
{
	__shared__ float shDot[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ float shGamma;
	__shared__ int shMainVecIdx;
	__shared__ float shMainSelfSum;
	__shared__ float shLabel;

	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;	
		shMainSelfSum = rowSum[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}	

	shDot[threadIdx.x]=0.0;
	__syncthreads();

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < num_rows; row += num_warps){

		// use two threads to fetch vecPointers[row] and vecPointers[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];            //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = vecPointers[row+1];
		
		float labelProd = tex1Dfetch(labelsTexRef,row)*shLabel;

		SpMV_CSR_nChi2<1>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shDot);
		// first thread writes warp result
		if (thread_lane == 0){
			float chi=rowSum[row]+shMainSelfSum-4*shDot[threadIdx.x];
			results[row]=labelProd*expf(-shGamma*chi);
		}
	}
}



/*
Computes dot product 


*/
template<int TexSel> __device__ void SpMV_CSR(const float * vals,
	const int * colIdx, 
	const int * vecPointers,
	const int row_start,
	const int row_end,
	const int row,
	const int num_rows,
	volatile float* shDot)
{

	const int thread_lane = threadIdx.x & (WARP_SIZE-1);
	// compute local sum
	float sum = 0;
	for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
	{
		sum += vals[jj] *fetchTex<TexSel>(colIdx[jj]); //tex1Dfetch(mainVecTexRef,colIdx[jj]);
	}
	shDot[threadIdx.x] = sum; 
	__syncthreads(); 

	// reduce local sums to row sum (warpsize 32)
	shDot[threadIdx.x] = sum = sum + shDot[threadIdx.x + 16];  
	shDot[threadIdx.x] = sum = sum + shDot[threadIdx.x +  8]; 
	shDot[threadIdx.x] = sum = sum + shDot[threadIdx.x +  4]; 
	shDot[threadIdx.x] = sum = sum + shDot[threadIdx.x +  2]; 
	shDot[threadIdx.x] = sum = sum + shDot[threadIdx.x +  1]; 

}


template<int TexSel> __device__ void SpMV_CSR_nChi2(const float * vals,
	const int * colIdx, 
	const int * vecPointers,
	const int row_start,
	const int row_end,
	const int row,
	const int num_rows,
	volatile float* shChi)
{

	const int thread_lane = threadIdx.x & (WARP_SIZE-1);
	// compute local sum
	float chi = 0;
	float val1=0, val2=0;
	int col = -1;
	for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
	{
		col = colIdx[jj];
		val2 = vals[jj];
		val1=fetchTex<TexSel>(col);
		chi+=(val1*val2)/(val1+val2+FLT_MIN);
		
	}

	shChi[threadIdx.x] =chi; 
	__syncthreads(); 

	// reduce local sums to row sum (warpsize 32)
	shChi[threadIdx.x] = chi = chi + shChi[threadIdx.x + 16];   
	shChi[threadIdx.x] = chi = chi + shChi[threadIdx.x +  8];  
	shChi[threadIdx.x] = chi = chi + shChi[threadIdx.x +  4];  
	shChi[threadIdx.x] = chi = chi + shChi[threadIdx.x +  2];  
	shChi[threadIdx.x] = chi = chi + shChi[threadIdx.x +  1];  

	__syncthreads(); 
}





/*******************************************************************************************/
/*									
 *								Evaluator CUDA Kernels
 *
 */

extern "C" __global__ void rbfCsrEvaluator(const float * vals,
	const int * colIdx, 
	const int * vecPointers, 
	const float* svSelfDot,
	const float* svAlpha,
	const float* svY,
	float * results,
	const int num_rows,
	const float vecSelfDot,
	const float gamma,
	const int texSel)
{
	__shared__ float shDot[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ float shGamma;
	__shared__ float shVecSelfDot;

	shDot[threadIdx.x]=0.0;	

	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfDot = vecSelfDot;
	}	

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < num_rows; row += num_warps){

		// use two threads to fetch vecPointers[row] and vecPointers[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];            //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = vecPointers[row+1];

		if(texSel==1){
			SpMV_CSR<1>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shDot);
		}
		else{
			SpMV_CSR<2>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shDot);
		}
		
		// first thread writes warp result
		if (thread_lane == 0){
			results[row]=svY[row]*svAlpha[row]*expf(-shGamma*(svSelfDot[row]+shVecSelfDot-2*shDot[threadIdx.x]));

		}
	}
}



extern "C" __global__ void expChiCsrEvaluator(const float * vals,
	const int * colIdx, 
	const int * vecPointers, 
	const float* svSelfSum,
	const float* svAlpha,
	const float* svY,
	float * results,
	const int num_rows,
	const float vecSelfSum,
	const float gamma,
	const int texSel)
{
	__shared__ float shChi[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ float shGamma;
	__shared__ float shVecSelfDot;

	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfDot = vecSelfSum;
	}	
	shChi[threadIdx.x]=0.0;	

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < num_rows; row += num_warps){

		// use two threads to fetch vecPointers[row] and vecPointers[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];            //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = vecPointers[row+1];

		if(texSel==1){
			SpMV_CSR_nChi2<1>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shChi);
		}
		else{
			SpMV_CSR_nChi2<2>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shChi);
		}

		// first thread writes warp result
		if (thread_lane == 0){
			float chi =svSelfSum[row]+shVecSelfDot-4*shChi[threadIdx.x];
			results[row]=svY[row]*svAlpha[row]*expf(-shGamma*chi);
		}
	}
}


extern "C" __global__ void nChi2CsrEvaluator(const float * vals,
	const int * colIdx, 
	const int * vecPointers, 
	const float* svAlpha,
	const float* svY,
	float * results,
	const int num_rows,
	const int texSel)
{
	__shared__ float shChi[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	shChi[threadIdx.x]=0.0;	

	for(int row = warp_id; row < num_rows; row += num_warps){

		// use two threads to fetch vecPointers[row] and vecPointers[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];            //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];            //same as: row_end   = vecPointers[row+1];

		if(texSel==1){
			SpMV_CSR_nChi2<1>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shChi);
		}
		else{
			SpMV_CSR_nChi2<2>(vals,colIdx,vecPointers,row_start,row_end,row,num_rows,shChi);
		}

		// first thread writes warp result
		if (thread_lane == 0){
			results[row]=svY[row]*svAlpha[row]*shChi[threadIdx.x];
		}
	}
}




//summary: cuda rbf kernel for evaluation, predicts new unseen elements using rbf SVM kernel,
// first elements matrix is in sparse CSR format, second (support vectors) matrix B is 
// in column major order (each kolumn is in dense format, in 'svTexRef' texture cache)
// you have to Launch this kernel as many times as support vectors, each time
// copy new support vector into texture cache
//params:
//AVals - values for first matrix
//AIdx - indexes for first matrix
//APtrs - pointers to next vector
//svLabels - support vector labels
//svAlphas - support vector alphas coef 
//selfDot - precomputed self linear product
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
//ColumnIndex - index of support vector in B matrix
//gamma - gamma prameter in RBF
extern "C" __global__ void rbfCSREvaluator_WholeDataSet(const float * AVals,
													  const int * AIdx, 
													  const int * APtrs, 
													  const float * svLabels,
													  const float * svAlphas,
													  const float* svSelfDot,
													  const float* elSelfDot,
													  float * result,
													  const int ARows,
													  const int BCols,
													  const float Gamma,
													  const int ColumnIndex)
{
	__shared__ float sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
	

	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	
	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	for(int row = warp_id; row < ARows; row += num_warps){
		// use two threads to fetch Ap[row] and Ap[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = APtrs[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

		// compute local sum
		float sum = 0;
		
		for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
		{	
			sum += AVals[jj] * tex1Dfetch(svTexRef,AIdx[jj]);
		}

		// reduce local sums to row sum (ASSUME: warpsize 32)
/*		
		sdata[threadIdx.x] = sum;
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
	   */
		volatile float* smem = sdata;
		smem[threadIdx.x] = sum; __syncthreads(); 
		smem[threadIdx.x] = sum = sum + smem[threadIdx.x + 16]; //__syncthreads(); 
		smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  8]; //__syncthreads();
		smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  4]; //__syncthreads();
		smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  2]; //__syncthreads();
		smem[threadIdx.x] = sum = sum + smem[threadIdx.x +  1]; //__syncthreads();


		// first thread writes warp result
		if (thread_lane == 0)
		{
			//remeber that we use result memory for stroing partial result
			//so the size of array is the same as number of elements
			result[row]+=expf(-Gamma*(elSelfDot[row]+svSelfDot[ColumnIndex]-2*smem[threadIdx.x]))*svLabels[ColumnIndex]*svAlphas[ColumnIndex];

			//results[row]+=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*sdata[threadIdx.x]));
			
		}
		
			
	}
}





 



