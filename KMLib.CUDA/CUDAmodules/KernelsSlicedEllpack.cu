/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#include <float.h>

#include <Config.h>


#define ThreadPerRow 4
#define LOG_THREADS 2 // LOG2(ThreadPerRow)
#define SliceSize 64

__device__ const int ROWS_B= SliceSize;

template<int TexSel> __device__ void SpMV_SliceEllpack(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shDot);


template<int TexSel> __device__ void SpMV_SERTILP(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int shSliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shDot);


template<int TexSel> __device__ void SpMV_SliceEllpack_nChi2(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shChi);

template<int TexSel> __device__ void SpMV_SERTILP_nChi2(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int shSliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shChi);

//extern __shared__  float sh_data[];

//Use sliced Ellpack format for computing rbf kernel
//vecVals - vectors values in Sliced Ellpack,
//vecCols - array containing column indexes for non zero elements
//vecLengths  - number of non zero elements in row
//sliceStart   - determine where particular slice starts and ends
//selfDot    - precomputed self dot product
//result  - for final result
//mainVecIdx - index of main vector
//nrows   - number of rows
//align	  - align
extern "C" __global__ void rbfSlicedEllpackKernel(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* selfDot,
	const float* vecLabels,
	float *result,
	const int mainVecIdx,
	const int nrRows,
	const float gamma, 
	const int align){

		//sh_data size = SliceSize*ThreadsPerRow*sizeof(float)
		//float* sh_cache = (float*)sh_data;
		__shared__  float shDot[ThreadPerRow*SliceSize];
		shDot[threadIdx.x]=0.0;	

		__shared__ int shMainVecIdx;
		__shared__ float shMainSelfDot;
		__shared__ float shLabel;
		__shared__ float shGamma;

		if(threadIdx.x==0)
		{
			shMainVecIdx=mainVecIdx;
			shMainSelfDot = selfDot[shMainVecIdx];
			shLabel = vecLabels[shMainVecIdx];
			shGamma=gamma;
		}
		__syncthreads();
		
		int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
		int txm = threadIdx.x %  ThreadPerRow;
		//map group of thread to row, in this case 4 threads are mapped to one row
		int row =  thIdx>> LOG_THREADS; // 

		if (row < nrRows){
			
			SpMV_SliceEllpack<1>(vecVals,vecCols,vecLengths,sliceStart,align,row,nrRows,shDot);
			if(txm == 0 ){
					result[row]=vecLabels[row]*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*shDot[threadIdx.x]));
				}

		}//if row<nrRows 
}//end func



//Use sliced Ellpack format for computing rbf kernel
//vecVals - vectors values in Sliced Ellpack,
//vecCols - array containing column indexes for non zero elements
//vecLengths  - number of non zero elements in row
//sliceStart   - determine where particular slice starts and ends
//selfDot    - precomputed self dot product
//result  - for final result
//mainVecIdx - index of main vector
//nrows   - number of rows
//ali	  - align
extern "C" __global__ void rbfSERTILP(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* selfDot,
	const float* vecLabels,
	float *result,
	const int mainVecIdx,
	const int nrRows,
	const float gamma, 
	const int align){


		//sh_data size = SliceSize*ThreadsPerRow*sizeof(float)
		//float* sh_cache = (float*)sh_data;
		__shared__  float shDot[ThreadPerRow*SliceSize];

		__shared__ int shMainVecIdx;
		__shared__ int shSliceStart;
		__shared__ float shMainSelfDot;
		__shared__ float shLabel;
		__shared__ float shGamma;

		if(threadIdx.x==0)
		{
			shMainVecIdx=mainVecIdx;
			shMainSelfDot = selfDot[shMainVecIdx];
			shLabel = vecLabels[shMainVecIdx];
			shGamma=gamma;
			shSliceStart=sliceStart[blockIdx.x];
		}
		__syncthreads();

		

		//map group of thread to row, in this case 4 threads are mapped to one row
		int row =  (blockIdx.x*blockDim.x+threadIdx.x)>> LOG_THREADS; 

		if (row < nrRows){
			
			SpMV_SERTILP<1>(vecVals,vecCols,vecLengths,shSliceStart,align,row,nrRows,shDot);
			if(threadIdx.x<ROWS_B){
				//results[row2]=row2;			
				unsigned int row2=blockIdx.x* ROWS_B+threadIdx.x;
				if(row2<nrRows){
				//result[row2]= shDotv[threadIdx.x];
					result[row2]=vecLabels[row2]*shLabel*expf(-shGamma*(selfDot[row2]+shMainSelfDot-2*shDot[threadIdx.x]));
				}
			}
		}//if row<nrRows 
}//end func







/****** nChi2 kernels *******************/
//Use sliced Ellpack format for computing normalized Chi2 kernel, vectors should be histograms
// and normalized according to l1 norm
// K(x,y)= Sum( (xi*yi)/(xi+yi))
//
//vecVals - vectors values in Sliced Ellpack,
//vecCols - array containing column indexes for non zero elements
//vecLengths  - number of non zero elements in row
//sliceStart   - determine where particular slice starts and ends
//result  - for final result
//mainVecIdx - index of main vector
//nrows   - number of rows
//ali	  - align
extern "C" __global__ void nChi2SlEllKernel(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* vecLabels,
	float *result,
	const int mainVecIdx,
	const int nrRows,
	const int align){

		//sh_data size = SliceSize*ThreadsPerRow*sizeof(float)
		//float* sh_cache = (float*)sh_data;
		__shared__  float shChi[ThreadPerRow*SliceSize];

		__shared__ int shMainVecIdx;
		__shared__ float shLabel;


		shChi[threadIdx.x]=0.0;

		if(threadIdx.x==0)
		{
			shMainVecIdx=mainVecIdx;
			shLabel = vecLabels[shMainVecIdx];
		}
		__syncthreads();

		int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
		int txm = threadIdx.x %  ThreadPerRow;
		//map group of thread to row, in this case 4 threads are mapped to one row
		int row =  thIdx>> LOG_THREADS; // 

		if (row < nrRows){
			
			SpMV_SliceEllpack_nChi2<1>(vecVals,vecCols,vecLengths,sliceStart,align,row,nrRows,shChi);

				if(txm == 0 ){
					result[row]=vecLabels[row]*shLabel*shChi[threadIdx.x];
				}
			
		}//if row<nrRows  
}//end func



extern "C" __global__ void nChi2SERTILP(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* vecLabels,
	float *result,
	const int mainVecIdx,
	const int nrRows,
	const int align){

		//sh_data size = SliceSize*ThreadsPerRow*sizeof(float)
		//float* sh_cache = (float*)sh_data;
		__shared__  float shChi[ThreadPerRow*SliceSize];

		
		__shared__ int shMainVecIdx;
		__shared__ float shLabel;
		__shared__ int shSliceStart;

		shChi[threadIdx.x]=0.0;

		if(threadIdx.x==0)
		{
			shMainVecIdx=mainVecIdx;
			shLabel = vecLabels[shMainVecIdx];
			shSliceStart=sliceStart[blockIdx.x];
		}

		__syncthreads();

		//int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
		//map group of thread to row, in this case 4 threads are mapped to one row
		int row =  (blockIdx.x*blockDim.x+threadIdx.x)>> LOG_THREADS; // 

		if (row < nrRows){

			SpMV_SERTILP_nChi2<1>(vecVals,vecCols,vecLengths,shSliceStart,align,row,nrRows,shChi);

			if(threadIdx.x<ROWS_B){
				//results[row2]=row2;			
				unsigned int row2=blockIdx.x* ROWS_B+threadIdx.x;
				if(row2<nrRows){
					result[row2]=vecLabels[row2]*shLabel*shChi[threadIdx.x];
				}
			}

		}//if row<nrRows  
}//end func



/************* ExpChi2 kernels *******************/

//Use sliced Ellpack format for computing ExpChi2 kernel matrix column
// K(x,y)=exp( -gamma* Sum( (xi-yi)^2/(xi+yi)) =exp(-gamma (sum xi +sum yi -4*sum( (xi*yi)/(xi+yi)) ) )
//vecVals - vectors values in Sliced Ellpack,
//vecCols - array containing column indexes for non zero elements
//vecLengths  - number of non zero elements in row
//sliceStart   - determine where particular slice starts and ends
//selfSum    - precomputed sum of each row
//result  - for final result
//mainVecIdx - index of main vector
//nrows   - number of rows
//ali	  - align
extern "C" __global__ void expChi2SlEllKernel(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* selfSum,
	const float* vecLabels,
	float *result,
	const int mainVecIdx,
	const int nrRows,
	const float gamma, 
	const int align){

		__shared__  float shChi[ThreadPerRow*SliceSize];

		__shared__ int shMainVecIdx;
		__shared__ float shMainSelfSum;
		__shared__ float shLabel;
		__shared__ float shGamma;

		if(threadIdx.x==0)
		{
			shMainVecIdx=mainVecIdx;
			shMainSelfSum = selfSum[shMainVecIdx];
			shLabel = vecLabels[shMainVecIdx];
			shGamma=gamma;
		}

		int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
		int txm = threadIdx.x %  ThreadPerRow;
		//map group of thread to row, in this case 4 threads are mapped to one row
		int row =  thIdx>> LOG_THREADS; // 

		if (row < nrRows){
			
			SpMV_SliceEllpack_nChi2<1>(vecVals,vecCols,vecLengths,sliceStart,align,row,nrRows,shChi);

			if(txm == 0 ){
				float chi = selfSum[row]+shMainSelfSum-4*shChi[threadIdx.x];
				result[row]=vecLabels[row]*shLabel*expf(-shGamma*chi);
			}
		}//if row<nrRows 
}//end func


extern "C" __global__ void expChi2SERTILP(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* selfSum,
	const float* vecLabels,
	float *result,
	const int mainVecIdx,
	const int nrRows,
	const float gamma, 
	const int align){

		//sh_data size = SliceSize*ThreadsPerRow*sizeof(float)
		//float* sh_cache = (float*)sh_data;
		__shared__  float shChi[ThreadPerRow*SliceSize];

		__shared__ int shMainVecIdx;
		__shared__ float shMainSelfSum;
		__shared__ float shLabel;
		__shared__ float shGamma;
		__shared__ int shSliceStart;

		shChi[threadIdx.x]=0.0;

		if(threadIdx.x==0)
		{
			shMainVecIdx=mainVecIdx;
			shMainSelfSum = selfSum[shMainVecIdx];
			shLabel = vecLabels[shMainVecIdx];
			shGamma=gamma;
			shSliceStart=sliceStart[blockIdx.x];
		}

		

		__syncthreads();

		int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
		//map group of thread to row, in this case 4 threads are mapped to one row
		int row =  thIdx>> LOG_THREADS; // 

		if (row < nrRows){

			SpMV_SERTILP_nChi2<1>(vecVals,vecCols,vecLengths,shSliceStart,align,row,nrRows,shChi);

			if(threadIdx.x<ROWS_B){
				//results[row2]=row2;			
				unsigned int row2=blockIdx.x* ROWS_B+threadIdx.x;
				if(row2<nrRows){
					float chi = selfSum[row2]+shMainSelfSum-4*shChi[threadIdx.x];
					result[row2]=vecLabels[row2]*shLabel*expf(-shGamma*chi);
				}
			}

		}//if row<nrRows  
}//end func

/************************************************************************/
/* 
	Evaluators
*/
/************************************************************************/

extern "C" __global__ void rbfSliceEllpackEvaluator(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* svSelfDot,
	const float* svAlpha,
	const float* svY,
	float * results,
	const int nrRows,
	const int align,
	const float vecSelfDot,
	const float gamma,
	const int texSel)
{

	__shared__ float shGamma;
	__shared__ float shVecSelfDot;
	__shared__ int shRows;
	__shared__  float shDot[ThreadPerRow*SliceSize];
	shDot[threadIdx.x]=0.0;	

	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfDot = vecSelfDot,
		shRows= nrRows;
	}
	__syncthreads();

	

	int txm = threadIdx.x %  ThreadPerRow;
	int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
	const int row   = thIdx>>LOG_THREADS;  // global thread index

	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in different streams
		
		if (texSel==1)
		{
			SpMV_SliceEllpack<1>(vecVals,vecCols,vecLengths,sliceStart,align,row,nrRows,shDot);
		}else{
			SpMV_SliceEllpack<2>(vecVals,vecCols,vecLengths,sliceStart,align,row,nrRows,shDot);
		}

		if(txm == 0 ){
			//results[row]=shDot[threadIdx.x];
			results[row]=svY[row]*svAlpha[row]*expf(-shGamma*(svSelfDot[row]+shVecSelfDot-2*shDot[threadIdx.x]));
		}

	}	

}


extern "C" __global__ void expChi2SliceEllpackEvaluator(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* svSelfSum,
	const float* svAlpha,
	const float* svY,
	float * results,
	const int nrRows,
	const int align,
	const float vecSelfSum,
	const float gamma,
	const int texSel)
{

	__shared__ float shGamma;
	__shared__ float shVecSelfSum;
	__shared__ int shRows;
	__shared__  float shChi[ThreadPerRow*SliceSize];
	shChi[threadIdx.x]=0.0;	

	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfSum = vecSelfSum,
		shRows= nrRows;
	}
	__syncthreads();



	int txm = threadIdx.x %  ThreadPerRow;
	int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
	const int row   = thIdx>>LOG_THREADS;  // global thread index

	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in different streams

		if (texSel==1)
		{
			SpMV_SliceEllpack_nChi2<1>(vecVals,vecCols,vecLengths,sliceStart,align,row,shRows,shChi);
		}else{
			SpMV_SliceEllpack_nChi2<2>(vecVals,vecCols,vecLengths,sliceStart,align,row,shRows,shChi);
		}

		if(txm == 0 ){
			float chi = svSelfSum[row]+shVecSelfSum-4*shChi[threadIdx.x];
			results[row]=svY[row]*svAlpha[row]*expf(-shGamma*chi);
		}
	}	

}


extern "C" __global__ void nChi2SliceEllpackEvaluator(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* svAlpha,
	const float* svY,
	float * results,
	const int nrRows,
	const int align,
	const int texSel)
{

	__shared__ float shGamma;
	__shared__ float shVecSelfSum;
	__shared__ int shRows;
	__shared__  float shChi[ThreadPerRow*SliceSize];
	shChi[threadIdx.x]=0.0;	

	if(threadIdx.x==0)
	{
		shRows= nrRows;
	}
	__syncthreads();



	int txm = threadIdx.x %  ThreadPerRow;
	int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
	const int row   = thIdx>>LOG_THREADS;  // global thread index

	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in different streams

		if (texSel==1)
		{
			SpMV_SliceEllpack_nChi2<1>(vecVals,vecCols,vecLengths,sliceStart,align,row,shRows,shChi);
		}else{
			SpMV_SliceEllpack_nChi2<2>(vecVals,vecCols,vecLengths,sliceStart,align,row,shRows,shChi);
		}

		if(txm == 0 ){
			results[row]=svY[row]*svAlpha[row]*shChi[threadIdx.x];
		}
	}	

}


extern "C" __global__ void rbfSERTILPEvaluator(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* svSelfDot,
	const float* svAlpha,
	const float* svY,
	float * results,
	const int nrRows,
	const int align,
	const float vecSelfDot,
	const float gamma,
	const int texSel)
{
	__shared__ float shGamma;
	__shared__ float shVecSelfDot;
	__shared__ int shRows;
	__shared__ int shSliceStart;
	__shared__  float shDot[ThreadPerRow*SliceSize];
	shDot[threadIdx.x]=0.0;	

	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfDot = vecSelfDot,
		shRows= nrRows;
		shSliceStart=sliceStart[blockIdx.x];
	}
	__syncthreads();


	
	const int row   = (blockIdx.x*blockDim.x+threadIdx.x)>>LOG_THREADS;  // global thread index

	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in different streams

		if (texSel==1)
		{
			SpMV_SERTILP<1>(vecVals,vecCols,vecLengths,shSliceStart,align,row,nrRows,shDot);
		}else{
			SpMV_SERTILP<2>(vecVals,vecCols,vecLengths,shSliceStart,align,row,nrRows,shDot);
		}
		
		if(threadIdx.x<ROWS_B){
			//results[row2]=row2;			
			unsigned int row2=blockIdx.x* ROWS_B+threadIdx.x;
			if(row2<nrRows){
				results[row2]=svY[row2]*svAlpha[row2]*expf(-shGamma*(svSelfDot[row2]+shVecSelfDot-2*shDot[threadIdx.x]));
			}
		}



	}	


}

extern "C" __global__ void expChi2SERTILPEvaluator(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* svSelfSum,
	const float* svAlpha,
	const float* svY,
	float * results,
	const int nrRows,
	const int align,
	const float vecSelfSum,
	const float gamma,
	const int texSel)
{
	__shared__ float shGamma;
	__shared__ float shVecSelfSum;
	__shared__ int shRows;
	__shared__ int shSliceStart;
	__shared__  float shChi[ThreadPerRow*SliceSize];
	shChi[threadIdx.x]=0.0;	

	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfSum = vecSelfSum,
		shRows= nrRows;
		shSliceStart=sliceStart[blockIdx.x];
	}
	__syncthreads();

	const int row   = (blockIdx.x*blockDim.x+threadIdx.x)>>LOG_THREADS;  // global thread index
	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in different streams

		if (texSel==1)
		{
			SpMV_SERTILP<1>(vecVals,vecCols,vecLengths,shSliceStart,align,row,shRows,shChi);
		}else{
			SpMV_SERTILP<2>(vecVals,vecCols,vecLengths,shSliceStart,align,row,shRows,shChi);
		}

		if(threadIdx.x<ROWS_B){
			unsigned int row2=blockIdx.x* ROWS_B+threadIdx.x;
			if(row2<nrRows){
				float chi = svSelfSum[row2]+shVecSelfSum-4*shChi[threadIdx.x];
				results[row2]=svY[row2]*svAlpha[row2]*expf(-shGamma*chi);
			}
		}
	}	
}


extern "C" __global__ void nChi2SERTILPEvaluator(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const float* svAlpha,
	const float* svY,
	float * results,
	const int nrRows,
	const int align,
	const int texSel)
{
	__shared__ int shRows;
	__shared__ int shSliceStart;
	__shared__  float shChi[ThreadPerRow*SliceSize];
	shChi[threadIdx.x]=0.0;	

	if(threadIdx.x==0)
	{
		shRows= nrRows;
		shSliceStart=sliceStart[blockIdx.x];
	}
	__syncthreads();

	const int row   = (blockIdx.x*blockDim.x+threadIdx.x)>>LOG_THREADS;  // global thread index
	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in different streams

		if (texSel==1)
		{
			SpMV_SERTILP<1>(vecVals,vecCols,vecLengths,shSliceStart,align,row,shRows,shChi);
		}else{
			SpMV_SERTILP<2>(vecVals,vecCols,vecLengths,shSliceStart,align,row,shRows,shChi);
		}

		if(threadIdx.x<ROWS_B){
			unsigned int row2=blockIdx.x* ROWS_B+threadIdx.x;
			if(row2<nrRows){
				results[row2]=svY[row2]*svAlpha[row2]*shChi[threadIdx.x];
			}
		}
	}	
}

/************************************************************************/
/* 
	Sliced ellpack 
*/
/************************************************************************/


/*
Computes sparse matrix dense vector multiplication, matrix in Sliced Ellpack format, vector in texture reference
Template parameter 'TexSel' indicates the texture we use in particular CUDA kernel call

shDot - pointer to shared memory, uses for shared parallel reduction and computing final results,
		each dot product is in position which are multiple of ThreadPerRow, eg. if ThreadPerRow=4 then results are on position
		[0,4,8,....]
*/
template<int TexSel> __device__ void SpMV_SliceEllpack(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shDot)
{
	
	int txm = threadIdx.x %  ThreadPerRow;

	float sub = 0.0;
	int maxRow = (int)ceil(vecLengths[row]/(float)ThreadPerRow);
	int col=-1;
	float value =0;
	int ind=0;

	for(int i=0; i < maxRow; i++){
		ind = i*align+sliceStart[blockIdx.x]+threadIdx.x;
		col     = vecCols[ind];
		value = vecVals[ind];
		sub += value * fetchTex<TexSel>(col);// tex1Dfetch(mainVecTexRef, col);
	}

	shDot[threadIdx.x] = sub;
	__syncthreads();


	for(int s=ThreadPerRow/2; s>0; s>>=1) //s/=2
	{
		if(txm < s){
			shDot[threadIdx.x] += shDot[threadIdx.x+s];
		}
	}

}


template<int TexSel> __device__ void SpMV_SliceEllpack_nChi2(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shChi){

		int txm = threadIdx.x %  ThreadPerRow;

		float chi = 0.0;
		int maxRow = (int)ceil(vecLengths[row]/(float)ThreadPerRow);
		int col=-1;
		float val1 =0;
		float val2 =0;
		int ind=0;

		for(int i=0; i < maxRow; i++){
			ind = i*align+sliceStart[blockIdx.x]+threadIdx.x;
			col     = vecCols[ind];
			val1 = vecVals[ind];
			val2 = fetchTex<TexSel>(col);

			chi+=(val1*val2)/(val1+val2+FLT_MIN);
		}

		shChi[threadIdx.x] = chi;
		__syncthreads();


		for(int s=ThreadPerRow/2; s>0; s>>=1) //s/=2
		{
			if(txm < s){
				shChi[threadIdx.x] += shChi[threadIdx.x+s];
			}
		}

}


/*
Computes sparse matrix dense vector multiplication, matrix in SERTILP format, vector in texture reference
Template parameter 'TexSel' indicates the texture we use in particular CUDA kernel call

shDot - pointer to shared memory, uses for shared parallel reduction and computing final results,
		each dot product is in position which are multiple of ThreadPerRow, eg. if ThreadPerRow=4 then results are on position
		[0,4,8,....]
*/
template<int TexSel> __device__ void SpMV_SERTILP(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int shSliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shDot)
{
	
	/*define ROWS_B SliceSize*/
	int idxT = threadIdx.x % ThreadPerRow; //thread number in Thread group
	int idxR = threadIdx.x/ThreadPerRow; //row index mapped into block region

	int maxRow = vecLengths[row];
	float val[PREFETCH_SIZE];
	int col[PREFETCH_SIZE];
	float dot[PREFETCH_SIZE]={0};

	unsigned int j=0;
	unsigned int arIdx=0;
	for(int i=0; i < maxRow; i++){

		#pragma unroll
		for( j=0; j<PREFETCH_SIZE;j++)	{
			//arIdx = (i*PREFETCH_SIZE+j )*align+sliceStart[blockIdx.x]+threadIdx.x;
			arIdx = (i*PREFETCH_SIZE+j )*align+shSliceStart+threadIdx.x;
			col[j] = vecCols[arIdx];
			val[j] = vecVals[arIdx];
		}

		#pragma unroll
		for( j=0; j<PREFETCH_SIZE;j++){
			dot[j] +=val[j]*fetchTex<TexSel>(col[j]); 
		}
	}

	#pragma unroll
	for( j=1; j<PREFETCH_SIZE;j++){
		dot[0]+=dot[j];	
	}


	shDot[idxT*ROWS_B+idxR]=dot[0];
	__syncthreads();		

	//reduction to some level
	for( j=blockDim.x/2; j>=ROWS_B; j>>=1) //s/=2
	{
		if(threadIdx.x<j){
			shDot[threadIdx.x]+=shDot[threadIdx.x+j];
		}
		__syncthreads();
	}
}


template<int TexSel> __device__ void SpMV_SERTILP_nChi2(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int shSliceStart, 
	const int align,
	const int row,
	const int nrRows,
	volatile float* shChi)
{

	/*define ROWS_B SliceSize*/
	int idxT = threadIdx.x % ThreadPerRow; //thread number in Thread group
	int idxR = threadIdx.x/ThreadPerRow; //row index mapped into block region

	int maxRow = vecLengths[row];
	
	float val1[PREFETCH_SIZE];
	float val2=0;

	int col[PREFETCH_SIZE];
	float chi2[PREFETCH_SIZE]={0};

	unsigned int j=0;
	unsigned int arIdx=0;
	for(int i=0; i < maxRow; i++){

		#pragma unroll
		for( j=0; j<PREFETCH_SIZE;j++)	{
			//arIdx = (i*PREFETCH_SIZE+j )*align+sliceStart[blockIdx.x]+threadIdx.x;
			arIdx = (i*PREFETCH_SIZE+j )*align+shSliceStart+threadIdx.x;
			col[j] = vecCols[arIdx];
			val1[j] = vecVals[arIdx];
		}

		#pragma unroll
		for( j=0; j<PREFETCH_SIZE;j++){
			val2=fetchTex<TexSel>(col[j]);
			chi2[j]+=(val1[j]*val2)/(val1[j]+val2+FLT_MIN);
		}
	}

	#pragma unroll
	for( j=1; j<PREFETCH_SIZE;j++){
		chi2[0]+=chi2[j];	
	}


	shChi[idxT*ROWS_B+idxR]=chi2[0];
	__syncthreads();		

	//reduction to some level
	for( j=blockDim.x/2; j>=ROWS_B; j>>=1) //s/=2
	{
		if(threadIdx.x<j){
			shChi[threadIdx.x]+=shChi[threadIdx.x+j];
		}
		__syncthreads();
	}

}


/************************* HELPER FUNCTIONS ****************************/

extern "C" __global__ void makeDenseVectorSlicedEllRT(const float *vecVals,
	const int *vecCols,
	const int *vecLengths, 
	const int * sliceStart, 
	float *mainVector,
	const int mainVecIdx,
	const int nrRows,
	const int vecDim,
	const int align){


		__shared__ int shMaxNNZ;
		__shared__ int shSliceNr;
		__shared__ int shRowInSlice;

		if(threadIdx.x==0)
		{
			shMaxNNZ =	vecLengths[mainVecIdx];
			//in which slice main vector is?
			shSliceNr = mainVecIdx/SliceSize;
			shRowInSlice = mainVecIdx% SliceSize;
		}

		int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
		//int tmx = threadIdx.x % ThreadPerRow;	

		if(thIdx < vecDim)
		{
			//set all vector values to zero
			mainVector[thIdx]=0.0;

			if(thIdx <shMaxNNZ){
				int threadNr = thIdx%ThreadPerRow;
				int rowSlice= thIdx/ThreadPerRow;

				//int	ind = sliceStart[shSliceNr]+shStartRow+tmx;

				int idx = sliceStart[shSliceNr] + align * rowSlice + shRowInSlice * ThreadPerRow + threadNr;

				int col     = vecCols[idx];
				float value = vecVals[idx];
				mainVector[col]=value;
			}


		}//end if

}//end func
