/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#include <float.h>

#include <Config.h>




//cuda kernel funtion for computing SVM Chi-Square kernel in its normalized version,
// K(x,y)= Sum( (xi*yi)/(xi+yi))
// this kernel is good for histogram type vectors, so each xi should be xi>=0, and whole vector should be l1 normalized
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
			
			chi+= (val1*val2)/(val1+val2+FLT_MIN);
			
		}
		results[row]=labelProd*chi;
	}	

}


/* EXP CHI2 kernel */

//cuda kernel funtion for computing SVM exp Chi-Square kernel,
// K(x,y)=exp( -gamma* Sum( (xi-yi)^2/(xi+yi)) =exp(-gamma (sum xi +sum yi -4*sum( (xi*yi)/(xi+yi)) ) )
// 
// uses Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//rowSum - array of precomputed row sums
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void expChi2EllpackKernel(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* rowSum,
									   float * results,
									   const int numRows,
									   const int mainVecIndex,
										const float gamma)
{

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
			
			chi+= (val1*val2)/(val1+val2+FLT_MIN);
			
		}
		chi=rowSum[row]+shMainSelfSum-4*chi;
		results[row]=labelProd*expf(-shGamma*chi);
	}	

}


/* EXP CHI2 kernel */

//cuda kernel funtion for computing SVM exp Chi-Square kernel,
// K(x,y)=exp( Sum( (xi-yi)^2/(xi+yi)) = sum xi +sum yi -4*sum( (xi*yi)/(xi+yi))
// with ILP technique
// uses Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//rowSum - array of precomputed row sums
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void expChi2EllpackKernel_ILP(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* rowSum,
									   float * results,
									   const int numRows,
									   const int mainVecIndex,
									   const float gamma)
{

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
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	const int num_rows =numRows;
	if(row<num_rows)
	{
		int maxEl = rowLength[row];
		float labelProd = tex1Dfetch(labelsTexRef,row)*shLabel;
				
		
		int col[2]={-1,-1};
		float val1[2]={0,0};
		float val2[2]={0,0};
		float sum[2] = {0, 0};
		int i=0;
		int sw=0;
		for(i=0; i<maxEl;i++)
		{
			sw= i%2;
			col[sw]=colIdx[num_rows*i+row];
			val1[sw]= vals[num_rows*i+row];
			val2[sw] = tex1Dfetch(mainVectorTexRef,col[sw]);
			
			sum[sw]+= (val1[sw]*val2[sw])/(val1[sw]+val2[sw]+FLT_MIN);
			
		}

		sum[0]=rowSum[row]+shMainSelfSum-4*(sum[0]+sum[1]);
		results[row]=labelProd*expf(-shGamma*sum[0]);
	}	

}


/******************* chi^2 kernels ********************/


//cuda kernel funtion for computing SVM Chi-Square kernel,
// K(x,y)=1-0.5*( Sum( (xi-yi)^2/(xi+yi)) =1-0.5* ( sum xi +sum yi -4*sum( (xi*yi)/(xi+yi)))
// 
// uses Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//rowSum - array of precomputed row sums
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void chi2EllpackKernel(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* rowSum,
									   float * results,
									   const int numRows,
									   const int mainVecIndex)
{

	__shared__ float shGamma;
	__shared__ int shMainVecIdx;
	__shared__ float shMainSelfSum;
	__shared__ float shLabel;
	
	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shMainSelfSum = rowSum[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
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
			
			chi+= (val1*val2)/(val1+val2+FLT_MIN);
			
		}
		chi=1-0.5f*(rowSum[row]+shMainSelfSum-4*chi);
		results[row]=labelProd*chi;
	}	

}


//!!! not optimal GPU utilization, only for testing
//cuda kernel funtion for computing SVM Chi-Square kernel, 
// K(x,y)= 1 -0.5* Sum( (xi-yi)^2/(xi+yi))
//
//uses Ellpack-R fromat for storing sparse matrix, labels are in texture cache
//
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
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
			
			chi+= (val1-val2)*(val1-val2)/(val1+val2+FLT_MIN);
			
			//vector in Ellpack format might miss some previous columns which are non zero in dense vector
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
		results[row]=labelProd*(1-0.5*chi);
	}	

}


