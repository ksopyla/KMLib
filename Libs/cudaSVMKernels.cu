/*

author: Krzysztof Sopyła

*/

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

#define PREFETCH_SIZE 2

#define VECDIM 597

#define maxNNZ 100

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
		{
			sum += vals[jj] * tex1Dfetch(mainVectorTexRef,idx[jj]);
			//__syncthreads();
		}

		// reduce local sums to row sum (ASSUME: warpsize 32)
/*		 old code not working on fermi card
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
		if (thread_lane == 0){
			//results[row]=tex1Dfetch(labelsTexRef,row)*tex1Dfetch(labelsTexRef,shMainVecIdx)*expf(-shGamma*(selfDot[row]+selfDot[shMainVecIdx]-2*sdata[threadIdx.x]));
			
			results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*smem[threadIdx.x]));
		}
	}
}



//cuda kernel funtion for computing SVM RBF kernel, uses 
// Ellpack-R fromat for storing sparse matrix, labels are in texture cache,  uses ILP - prefetch vector elements in registers
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfEllpackFormatKernel_ILP(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   float * results,
									   const int num_rows,
									   const int mainVecIndex,
									   const float gamma)
{
	

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
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<num_rows)
	{
		float dot=0;
		int maxEl = rowLength[row];
		
		int i=0;
		
		float preVals[PREFETCH_SIZE];
		int preColls[PREFETCH_SIZE];
		float preVecVals[PREFETCH_SIZE];
		
		//how many elements are the rest after division
		int rest = maxEl%PREFETCH_SIZE;
		int mainIter = ceilf( (maxEl+0.0)/PREFETCH_SIZE);
		for(i=0; i<mainIter;i++)
		{
			int subIter= min(maxEl-i*PREFETCH_SIZE,PREFETCH_SIZE);
			
			for(int j=0; j<subIter;j++)			
			{
				preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*num_rows+row];
				preVals[j]=vals[ (i*PREFETCH_SIZE+j)*num_rows+row];
			}			

			
			for(int j=0; j<subIter;j++)
			{
				preVecVals[j] = tex1Dfetch(mainVectorTexRef,preColls[j]);
			}

			for(int j=0; j<subIter;j++){
				dot+=preVals[j]*preVecVals[j];
				//dot+=preVals[j]*tex1Dfetch(mainVectorTexRef,preColls[j]);
			}
		}
		results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
	}	

}

//cuda kernel funtion for computing SVM RBF kernel, uses 
// Ellpack-R fromat for storing sparse matrix, labels are in texture cache,  uses ILP - prefetch vector elements in registers
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfEllpackFormatKernel_ILP_shared(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   float * results,
									   const int num_rows,
									   const int mainVecIndex,
									   const float gamma)
{
	

	__shared__ float shGamma;
	__shared__ int shMainVecIdx;
	__shared__ float shMainSelfDot;
	__shared__ float shLabel;

	
	__shared__ float shMainVec[VECDIM];
	
	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}

	for(int k=threadIdx.x;k<VECDIM;k+=blockDim.x)
		shMainVec[k]=tex1Dfetch(mainVectorTexRef,k);
	
	__syncthreads();
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<num_rows)
	{
		float dot=0;
		int maxEl = rowLength[row];
		
		int i=0;
		
		float preVals[PREFETCH_SIZE];
		int preColls[PREFETCH_SIZE];
		//float preVecVals[PREFETCH_SIZE];
		
		//how many elements are the rest after division
		int rest = maxEl%PREFETCH_SIZE;
		int mainIter = ceilf( (maxEl+0.0)/PREFETCH_SIZE);
		for(i=0; i<mainIter;i++)
		{
			int subIter= min(maxEl-i*PREFETCH_SIZE,PREFETCH_SIZE);
			
			for(int j=0; j<subIter;j++)			
			{
				preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*num_rows+row];
				preVals[j]=vals[ (i*PREFETCH_SIZE+j)*num_rows+row];
			}			

			for(int j=0; j<subIter;j++){
				dot+=preVals[j]*shMainVec[preColls[j]];
				//dot+=preVals[j]*tex1Dfetch(mainVectorTexRef,preColls[j]);
			}
		}
		results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
	}	

}
//cuda kernel funtion for computing SVM RBF kernel, uses 
// Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfEllpackFormatKernel_shared(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   float * results,
									   const int numRows,
									   const int mainVecIndex,
									   const float gamma)
{
	

	__shared__ float shGamma;
	__shared__ int shMainVecIdx;
	__shared__ float shMainSelfDot;
	__shared__ float shLabel;
	
	__shared__ float shMainVec[VECDIM];
	//volatile float *shMainVec =shMainVecAR;
	
	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}

	for(int k=threadIdx.x;k<VECDIM;k+=blockDim.x)
		shMainVec[k]=tex1Dfetch(mainVectorTexRef,k);
	
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	const int num_rows =numRows;
	if(row<num_rows)
	{
		int maxEl = rowLength[row];
		float labelProd = tex1Dfetch(labelsTexRef,row)*shLabel;
		
		float dot=0;
		int col=-1;
		float val=0;
		int i=0;
		for(i=0; i<maxEl;i++)
		{
			col=colIdx[num_rows*i+row];
			val= vals[num_rows*i+row];
			dot+=val*shMainVec[col];
		}

		//results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
		//val=-shGamma*(selfDot[row]+shMainSelfDot-2*dot);
		//val=-shGamma;
		//val=selfDot[row];
		//val=shMainSelfDot;
		//val=-2*dot;



		val =labelProd* __expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		//val = floorf(val*1000+0.5)/1000;
		//results[row]=labelProd*val;
		results[row]=val;
		
	}	

}



//cuda kernel funtion for computing SVM RBF kernel, uses 
// Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfEllpackFormatKernel(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   float * results,
									   const int numRows,
									   const int mainVecIndex,
									   const float gamma)
{
	

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
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	const int num_rows =numRows;
	if(row<num_rows)
	{
		int maxEl = rowLength[row];
		int labelProd = tex1Dfetch(labelsTexRef,row)*shLabel;
		float dot=0;
		
		int col=-1;
		float val=0;
		int i=0;
		for(i=0; i<maxEl;i++)
		{
			col=colIdx[num_rows*i+row];
			val= vals[num_rows*i+row];
			dot+=val*tex1Dfetch(mainVectorTexRef,col);
		}
		//results[row]=shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
		//results[row]=labelProd*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		//results[row]=labelProd*exp(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));

		val = expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		//val = floorf(val*10000+0.5)/10000;
		results[row]=labelProd*val;
		
		//results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
	}	

}


//cuda kernel funtion for computing SVM Chi-Square kernel, 
// K(x,y)= 1 - Sum( (xi-yi)^2/(xi+yi))
//
//uses Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void chiSquaredEllpackKernel(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   float * results,
									   const int numRows,
									   const int mainVecIndex)
{
	

	__shared__ int shMainVecIdx;
	__shared__ float shLabel;
	__shared__ int shMainCols[maxNNZ];
	__shared__ float shMainVals[maxNNZ];
	__shared__ int shMainNNZ;
	
	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
		shMainNNZ=  rowLength[mainVecIndex];
	}

	__syncthreads();
	
	for(int s=threadIdx.x;s<shMainNNZ;s+=blockDim.x){
		shMainCols[s] = colIdx[numRows*s+shMainVecIdx];
		shMainVals[s]=tex1Dfetch(mainVectorTexRef,shMainCols[s]);
	}
	
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	const int num_rows =numRows;
	if(row<num_rows)
	{
		int maxEl = rowLength[row];
		int labelProd = tex1Dfetch(labelsTexRef,row)*shLabel;
		float chi=0;
		
		int col1=-1;
		float val1=0;
		float val2=0;
		int i=0;
		int k=0;
		int prevCol=shMainCols[k];

		for(i=0; i<maxEl;i++)
		{
			col1=colIdx[num_rows*i+row];
			val1= vals[num_rows*i+row];
			val2 = tex1Dfetch(mainVectorTexRef,col1);
			
			chi+= (val1-val2)*(val1-val2)/(val1+val2);
			
			//vector in Ellpack format might miss some previous columns which are non zero in dens vector
			//we want to "catch up with main dense vector"
			
			while(k<shMainNNZ && prevCol<col1) //prevCol=cols[numRows*k+shMainVecIdx];
			{
				//it is sufficient to add only value of dense vector,
				//because sparse vector values in this position is zero
				chi+=tex1Dfetch(mainVectorTexRef,prevCol);
				k++;
				if(k<shMainNNZ)
					prevCol=shMainCols[k];
				
			}
			if(prevCol==col1){
				k++;//increase k, to move to first grather than col1 index
				if(k<shMainNNZ)
					prevCol=shMainCols[k];
			}
		}

		//add those values which left (were not added before)
		while(k<shMainNNZ){
			chi+=shMainVals[k++];		
			
			//prevCol=cols[numRows*k+shMainVecIdx];
			//chi+=tex1Dfetch(mainVectorTexRef,prevCol);
		}
		results[row]=labelProd*(1-2*chi);
	}	

}

//cuda kernel funtion for computing SVM Chi-Square kernel in its normalized version,
// K(x,y)= Sum( (xi*yi)/(xi+yi))
// uses Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void chiSquaredNormEllpackKernel(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   float * results,
									   const int numRows,
									   const int mainVecIndex)
{
	
	__shared__ float shLabel;
	
	if(threadIdx.x==0)
	{
		shLabel = tex1Dfetch(labelsTexRef,mainVecIndex);		
	}
	
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	const int num_rows =numRows;
	if(row<num_rows)
	{
		int maxEl = rowLength[row];
		float labelProd = tex1Dfetch(labelsTexRef,row)*shLabel;
		float chi=0;
		
		int col1=-1;
		float val1=0;
		float val2=0;
		int i=0;

		for(i=0; i<maxEl;i++)
		{
			col1=colIdx[num_rows*i+row];
			val1= vals[num_rows*i+row];
			val2 = tex1Dfetch(mainVectorTexRef,col1);
			
			chi+= (val1*val2)/(val1+val2);
			
		}
		results[row]=labelProd*chi;
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
extern "C" __global__ void rbfCSREvaluatorDenseVector(const float * AVals,
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
 



/*
Creates i-th dense vector from sparse matrix in Ellpack-R representation
vecVals - vector values, vectors are layout in matrix in Ellpack format
vecCols - vector columns indexes
vecLength - number of nonzero elements in each row
mainVector - output dense vector
mainVecIdx - main vector index, which vector should be created as dense
nrRows     - number of rows in matrix
vecDim     - vector dimensions
*/
extern "C" __global__ void makeDenseVectorEllpack(const float *vecVals,
											 const int *vecCols,
											 const int *vecLengths, 
											 float *mainVector,
											 const int mainVecIdx,
											 const int nrRows,
											 const int vecDim)
{
	__shared__ int shMaxNNZ;
	
	if(threadIdx.x==0)
	{
		shMaxNNZ =	vecLengths[mainVecIdx];
	}
	
	__syncthreads();

	int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);	
	if(thIdx < vecDim)
	{
		//set all vector values to zero
		mainVector[thIdx]=0.0;
	
		if(thIdx <shMaxNNZ){
			int col     = vecCols[thIdx*nrRows+mainVecIdx];
			float value = vecVals[thIdx*nrRows+mainVecIdx];
			
			//mainVector[thIdx]=col;
			mainVector[col]=value;
		}

	}//end if	
}//end func
