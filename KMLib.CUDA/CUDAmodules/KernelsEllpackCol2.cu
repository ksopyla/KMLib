/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#include <float.h>

#define PREFETCH_SIZE 2

//__device__ const unsigned int PREFETCH_SIZE=2;

texture<float,1,cudaReadModeElementType>  VecI_TexRef;
texture<float,1,cudaReadModeElementType>  VecJ_TexRef;


//cuda kernel funtion for computing SVM RBF kernel, uses 
// Ellpack-R fromat for storing sparse matrix, labels are in texture cache,  uses ILP - prefetch vector elements in registers
// arrays vals and colIdx should be aligned to PREFETCH_SIZE
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//selfDot - array of precomputed self linear product 
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
//gamma - gamma parameter for RBF 
extern "C" __global__ void rbfEllpackILPcol2(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   const float* y,
									   float * results,
									   const int num_rows,
									   const int indexI,
									   const int indexJ,
									   const float gamma)
{
	

	__shared__ float shGamma;
	__shared__ int shIdxI;
	__shared__ int shIdxJ;
	__shared__ float shISelfDot;
	__shared__ float shJSelfDot;
	__shared__ float shYI;
	__shared__ float shYJ;
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shIdxI=indexI;
		shIdxJ = indexJ;
		shGamma = gamma;
		shISelfDot = selfDot[shIdxI];
		shJSelfDot = selfDot[shIdxJ];
		shYI = y[shIdxI];
		shYJ = y[shIdxJ];
	}
	__syncthreads();
		
	const unsigned int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		float preVals[PREFETCH_SIZE];
		int preColls[PREFETCH_SIZE];
		
		
		//float dot[2][PREFETCH_SIZE]={0};

		float dotI[PREFETCH_SIZE]={0,0};
		
		float dotJ[PREFETCH_SIZE]={0,0};

		int maxEl = rowLength[row];
	
		unsigned int j=0;
		//unsigned int arIdx=0;
		for(unsigned int i=0; i<maxEl;i++)
		{
			#pragma unroll 2
			for( j=0; j<PREFETCH_SIZE;j++)			
			{
				preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*shRows+row];
				preVals[j]=vals[ (i*PREFETCH_SIZE+j)*shRows+row];
				
				//arIdx = (i*PREFETCH_SIZE+j)*shRows+row;
				//preColls[j]=colIdx[arIdx];
				//preVals[j]=vals[arIdx];
			}
			
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++){
				//dot[0][j]+=preVals[j]*tex1Dfetch(VecI_TexRef,preColls[j]);
				//dot[1][j]+=preVals[j]*tex1Dfetch(VecJ_TexRef,preColls[j]);
				
				dotI[j]+=preVals[j]*tex1Dfetch(VecI_TexRef,preColls[j]);
				dotJ[j]+=preVals[j]*tex1Dfetch(VecJ_TexRef,preColls[j]);

			}
			
		}
				
		#pragma unroll
		for( j=1; j<PREFETCH_SIZE;j++){
			//dot[0][0]+=dot[0][j];
			//dot[1][0]+=dot[1][j];
			dotI[0]+=dotI[j];
			dotJ[0]+=dotJ[j];
		}
		
		results[row]=y[row]*shYI*expf(-shGamma*(selfDot[row]+shISelfDot-2*dotI[0]));
		results[row+shRows]=y[row]*shYJ*expf(-shGamma*(selfDot[row]+shJSelfDot-2*dotJ[0]));
		//float yRow = y[row];
		//float selfDotRow = selfDot[row];
		//results[row]= yRow*shYI*expf(-shGamma*(selfDotRow+shISelfDot-2*dotI[0]));
		//results[row+shRows]=yRow*shYJ*expf(-shGamma*(selfDotRow+shJSelfDot-2*dotJ[0]));
		
	}	

}


extern "C" __global__ void rbfEllpackILPcol2_Prefetch2(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* selfDot,
									   const float* y,
									   float * results,
									   const int num_rows,
									   const int indexI,
									   const int indexJ,
									   const float gamma)
{
	

	__shared__ float shGamma;
	__shared__ int shIdxI;
	__shared__ int shIdxJ;
	__shared__ float shISelfDot;
	__shared__ float shJSelfDot;
	__shared__ float shYI;
	__shared__ float shYJ;
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shIdxI=indexI;
		shIdxJ = indexJ;
		shGamma = gamma;
		shISelfDot = selfDot[shIdxI];
		shJSelfDot = selfDot[shIdxJ];
		shYI = y[shIdxI];
		shYJ = y[shIdxJ];
	}
	__syncthreads();
		
	const unsigned int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		float preVals[2];
		int preColls[2];
		float dotI[2]={0,0};	
		float dotJ[2]={0,0};

		int maxEl = rowLength[row];
		unsigned int arIdx = row;	
		for(unsigned int i=0; i<maxEl;i++)
		{
				//preColls[0]=colIdx[ i*2*shRows+row];
				//preVals[0]=vals[i*2*shRows+row];
				//preColls[1]=colIdx[ (i*2+1)*shRows+row];
				//preVals[1]=vals[(i*2+1)*shRows+row];

				//arIdx = i*2*shRows+row;
				preColls[0]=colIdx[arIdx];
				preVals[0]=vals[arIdx];
				arIdx+=shRows;
				preColls[1]=colIdx[ arIdx];
				preVals[1]=vals[arIdx];
				arIdx+=shRows;

				dotI[0]+=preVals[0]*tex1Dfetch(VecI_TexRef,preColls[0]);
				dotI[1]+=preVals[1]*tex1Dfetch(VecI_TexRef,preColls[1]);
				dotJ[0]+=preVals[0]*tex1Dfetch(VecJ_TexRef,preColls[0]);
				dotJ[1]+=preVals[1]*tex1Dfetch(VecJ_TexRef,preColls[1]);

		}
					
		dotI[0]+=dotI[1];
		dotJ[0]+=dotJ[1];
		
		//results[row]=y[row]*shYI*expf(-shGamma*(selfDot[row]+shISelfDot-2*dotI[0]));
		//results[row+shRows]=y[row]*shYJ*expf(-shGamma*(selfDot[row]+shJSelfDot-2*dotJ[0]));
		float yRow = y[row];
		float selfDotRow = selfDot[row];
		results[row]= yRow*shYI*expf(-shGamma*(selfDotRow+shISelfDot-2*dotI[0]));
		results[row+shRows]=yRow*shYJ*expf(-shGamma*(selfDotRow+shJSelfDot-2*dotJ[0]));
		
	}	

}


