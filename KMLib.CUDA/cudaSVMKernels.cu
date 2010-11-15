
//texture for vector, which is used for matrix vector multiplication
//in SVM, when we have to compute many dot products (one vector with others)
texture<float,1,cudaReadModeElementType> mainVectorTexRef;

//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;


//texture for vector, which is used for evaluation procedure
//in SVM prediction, its stores one dense support vector
texture<float,1,cudaReadModeElementType> svTexRef;

#define BLOCK_SIZE 256

#define WARP_SIZE 32


/******************************************************************
 *
 *			Cuda Kernels for SVM kernels
 */

/*
Based on cuda kernels from 
"Efcient Sparse Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garlandy
December 11, 2008
*/
//
//cuda kernel funtion for computing SVM linear kernel, uses 
// CSR fromat for storing sparse matrix, labels and main vector are
//in texture cache
//Remarks: based on spmv_csr_vector_kernel from publication above
//Params:
//vals - array of vectors values
//idx  - array of vectros indexes in CSR fromat
//vecPointers -array of pointers(indexes) to idx and vals array to specific vectors
//results - array of results Linear Kernel
//num_rows - number of vectors, stored in CSR matrix format, each vector is stored in one row of matrix
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void linearCsrFormatKernel(const float * vals,
									   const int * idx, 
									   const int * vecPointers, 
									   float * results,
									   const int num_rows,
									   int mainVecIndex)
{
	__shared__ float sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	__shared__ int shMainVecIdx;
	if(threadIdx.x==0)
		shMainVecIdx=mainVecIndex;
	

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
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = vecPointers[row+1];

		// compute local sum
		float sum = 0;
		for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
			sum += vals[jj] * tex1Dfetch(mainVectorTexRef,idx[jj]);

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum;
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
	   

		

		// first thread writes warp result
		if (thread_lane == 0){
			//results[row] += sdata[threadIdx.x];
			//results[row] += sdata[threadIdx.x];
			//results[row] =tex1D(labelsTexRef,mainVecIndex)*tex1D(labelsTexRef,row) * sdata[threadIdx.x];
			//results[row] = tex1D(labelsTexRef,mainVecIndex)*sdata[threadIdx.x];
			//results[row] = tex1D(labelsTexRef,row);
			results[row] = tex1Dfetch(labelsTexRef,row)*tex1Dfetch(labelsTexRef,shMainVecIdx)*sdata[threadIdx.x];
		}

			
	}
}


//cuda kernel funtion for computing SVM linear kernel, uses 
// CSR fromat for storing sparse matrix, labels are in texture cache, main vector
//is in shared memeory
//Remarks: based on spmv_csr_vector_kernel from publication above
//Params:
//vals - array of vectors values
//idx  - array of vectros indexes in CSR fromat
//vecPointers -array of pointers(indexes) to idx and vals array to specific vectors
//results - array of results Linear Kernel
//num_rows - number of vectors, stored in CSR matrix format, each vector is stored in one row of matrix
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void linearCsrFormatKernelShared(const float * vals,
									   const int * idx, 
									   const int * vecPointers, 
									   float * results,
									   const int num_rows,
									   int mainVecIndex)
{
	__shared__ float sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
	
	extern __shared__ float shMainVec[];

	const int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
	const int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
	const int warp_id     = thread_id   / WARP_SIZE;                // global warp index
	const int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
	const int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

	int mainStart=vecPointers[mainVecIndex];
	int mainEnd=vecPointers[mainVecIndex+1];
	int diff=mainEnd-mainStart;
	if(thread_id<diff)
	{
		shMainVec[idx[mainStart+thread_id]]=vals[mainStart+thread_id];
	}
	__syncthreads(); 

	for(int row = warp_id; row < num_rows; row += num_warps){
		// use two threads to fetch vecPointers[row] and vecPointers[row+1]
		// this is considerably faster than the straightforward version
		if(thread_lane < 2)
			ptrs[warp_lane][thread_lane] = vecPointers[row + thread_lane];
		const int row_start = ptrs[warp_lane][0];                   //same as: row_start = vecPointers[row];
		const int row_end   = ptrs[warp_lane][1];                   //same as: row_end   = vecPointers[row+1];

		// compute local sum
		float sum = 0;
		for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
			sum += vals[jj] *  shMainVec[idx[jj]];

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum;
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
	   

		

		// first thread writes warp result
		if (thread_lane == 0){
			//results[row] += sdata[threadIdx.x];
			//results[row] += sdata[threadIdx.x];
			//results[row] =tex1D(labelsTexRef,mainVecIndex)*tex1D(labelsTexRef,row) * sdata[threadIdx.x];
			//results[row] = tex1D(labelsTexRef,mainVecIndex)*sdata[threadIdx.x];
			//results[row] = tex1D(labelsTexRef,row);
			results[row] = tex1Dfetch(labelsTexRef,row)*tex1Dfetch(labelsTexRef,mainVecIndex)*sdata[threadIdx.x];
		}

			
	}
	
}


//cuda kernel funtion for computing SVM RBF kernel, uses 
// CSR fromat for storing sparse matrix, labels are in texture cache, 
//Remarks: based on spmv_csr_vector_kernel from publication above
//Params:
//vals - array of vectors values
//idx  - array of vectros indexes in CSR fromat
//vecPointers -array of pointers(indexes) to idx and vals array to specific vectors
//selfDot - array of precomputed self linear product
//results - array of results Linear Kernel
//num_rows -number of vectors, stored in CSR matrix format, each vector is stored in one row of matrix
//mainVecIndex - main vector index, needed for retriving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfCsrFormatKernel(const float * vals,
									   const int * idx, 
									   const int * vecPointers, 
									   const float* selfDot,
									   float * results,
									   const int num_rows,
									   const int mainVecIndex,
									   const float gamma)
{
	__shared__ float sdata[BLOCK_SIZE + 16];                    // padded to avoid reduction ifs
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

		// compute local sum
		float sum = 0;
		for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
			sum += vals[jj] * tex1Dfetch(mainVectorTexRef,idx[jj]);

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum;
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();

		// first thread writes warp result
		if (thread_lane == 0){
			//results[row]=tex1Dfetch(labelsTexRef,row)*tex1Dfetch(labelsTexRef,shMainVecIdx)*expf(-shGamma*(selfDot[row]+selfDot[shMainVecIdx]-2*sdata[threadIdx.x]));
			
			results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*sdata[threadIdx.x]));
		}
	}
}



/*******************************************************************************************/
/*									
 *								Evaluator CUDA Kernels
 *
 */


//summary: cuda kernel for evaluation, predicts new unseen elements using linear SVM kernel,
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
//result - result matrix
//ARows - number of rows in first matrix
//BCols - number of cols in second matrix
//ColumnIndex - index of support vector in B matrix
extern "C" __global__ void linearCSREvaluatorDenseVector(const float * AVals,
									   const int * AIdx, 
									   const int * APtrs, 
									   const float * svLabels,
									   const float * svAlphas,
									   float * result,
									   const int ARows,
									   const int BCols,
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
			sum += AVals[jj] * tex1Dfetch(svTexRef,AIdx[jj]);

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum;
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
	   


		// first thread writes warp result
		if (thread_lane == 0)
		{
			//remeber that we use result memory for stroing partial result
			//so the size of array is the same as number of elements
			 result[row]+=sdata[threadIdx.x]*svLabels[ColumnIndex]*svAlphas[ColumnIndex];
			//row major order
			//result[row*BCols+ColumnIndex]= sdata[threadIdx.x];
			//column major order
			//result[ColumnIndex*ARows+row]= sdata[threadIdx.x];
		}
		
			
	}
}

