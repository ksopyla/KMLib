﻿/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#include <float.h>

#include <Config.h>


template<int TexSel> __device__ float SpMV_Ellpack_ILP(const float * vals,
									   const int * colIdx, 
									   const int * rowLength,
									   const int row,
									   const int num_rows);

template<int TexSel> __device__ float SpMV_Ellpack(const float * vals,
									   const int * colIdx, 
									   const int * rowLength,
									   const int row,
									   const int num_rows);

template<int TexSel> __device__ void SpMV_ERTILP(const float * vals,
									   const int * colIdx, 
									   const int * rowLength,
									   const int row,
									   const int num_rows,
									   volatile float* shResults);





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
extern "C" __global__ void rbfEllpackFormatKernel_ILP_func(const float * vals,
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
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
		shRows= num_rows;
	}
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		float dot = SpMV_Ellpack_ILP<1>(vals,colIdx,rowLength,row,num_rows);
		results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
	}	

}


//cuda kernel funtion for computing SVM RBF kernel, uses 
// Ellpack-R fromat for storing sparse matrix, labels are in texture cache,  uses ILP - prefetch vector elements in registers
// and uses T - threads to process one row
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
extern "C" __global__ void rbfERTILP(const float * vals,
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
	__shared__ int shRows;

	__shared__ float shDot[BLOCK_SIZE];
	shDot[threadIdx.x]=0.0;	
		
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}
	__syncthreads();
		

	//const int idx  = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	int row  = (blockDim.x * blockIdx.x + threadIdx.x)/THREADS_ROW;

	//const int rowsB= blockDim.x/THREADS_ROW ;//BLOCK_SIZE/THREADS_ROW;  //rows in block
	#define rowsB BLOCK_SIZE/THREADS_ROW

	

	if(row<shRows)
	{
					
		SpMV_ERTILP<1>(vals,colIdx,rowLength,row,shRows,shDot);
		//if(row2<shRows){
		if(threadIdx.x<rowsB){
			//results[row2]=row2;			
			unsigned int row2=blockIdx.x* rowsB+threadIdx.x;
			//results[row2]=shDot[tid];
		    results[row2]=tex1Dfetch(labelsTexRef,row2)*shLabel*expf(-shGamma*(selfDot[row2]+shMainSelfDot-2*shDot[threadIdx.x]));
		}
	}//if row<nrRows	

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
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
		shRows=numRows;
	}
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	//const int num_rows =numRows;
	if(row<shRows)
	{
		/*int maxEl = rowLength[row];
		
		float dot=0;
		
		int col=-1;
		float val=0;
		int i=0;
		for(i=0; i<maxEl;i++)
		{
			col=colIdx[num_rows*i+row];
			val= vals[num_rows*i+row];
			dot+=val*tex1Dfetch(mainVecTexRef,col);
		}*/

		int labelProd = tex1Dfetch(labelsTexRef,row)*shLabel;
		float dot = SpMV_Ellpack<1>(vals,colIdx,rowLength,row,numRows);
		results[row]=labelProd*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));;
	}	

}






/*********** Chi2 kernels **********************/

//cuda kernel funtion for computing SVM Chi-Square kernel in its normalized version,
// K(x,y)= Sum( (xi*yi)/(xi+yi))
// Ellpack-R fromat for storing sparse matrix, labels are in texture cache,  uses ILP - prefetch vector elements in registers
// and uses T - threads to process one row
// arrays vals and colIdx should be aligned to PREFETCH_SIZE
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//results - array of results Linear Kernel
//num_rows -number of vectors
//mainVecIndex - main vector index, needed for retriving its label
extern "C" __global__ void nChi2EllRTILP(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   float * results,
									   const int num_rows,
									   const int mainVecIndex)
{
	

	
	__shared__ int shMainVecIdx;
	__shared__ float shLabel;
	__shared__ int shRows;

	__shared__ float shChi2[BLOCK_SIZE];
	shChi2[threadIdx.x]=0.0;	
		
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shMainVecIdx=mainVecIndex;
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}
	__syncthreads();
		

	//const int idx  = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	int row  = (blockDim.x * blockIdx.x + threadIdx.x)/THREADS_ROW;

	//const int rowsB= blockDim.x/THREADS_ROW ;//BLOCK_SIZE/THREADS_ROW;  //rows in block
	#define rowsB BLOCK_SIZE/THREADS_ROW

	const int tid = threadIdx.x; // index in block
	const int idxR = tid/THREADS_ROW; //row index mapped into block region
	const int idxT = tid%THREADS_ROW; // thread number in Thread Group

	if(row<shRows)
	{
		float vals[PREFETCH_SIZE];
		float val2=0;
		int cols[PREFETCH_SIZE];
		
		float dot[PREFETCH_SIZE]={0};

		int maxEl = rowLength[row]; //original row length divided by T*PREFETCH

		unsigned int j=0;
		unsigned int arIdx=0;
		
		for(int i=0; i<maxEl;i++)
		{
			
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++)			
			{
				arIdx = (i*PREFETCH_SIZE+j)*shRows*THREADS_ROW+row*THREADS_ROW+idxT;
				cols[j]=colIdx[arIdx];
				vals[j]=vals[arIdx];
			}
			
			#pragma unroll
			for( j=0; j<PREFETCH_SIZE;j++){
				val2=tex1Dfetch(mainVecTexRef,cols[j]);
				dot[j]+=(vals[j]*val2)/(vals[j]+val2+FLT_MIN);
			}
			
		}
		

		#pragma unroll
		for( j=1; j<PREFETCH_SIZE;j++){
				dot[0]+=dot[j];
				
		}

		//__syncthreads();	

		// special indexing, values for example for T=4 BlockSize=256
		//for row=0 values are stored on position 0,64,128,192 
		//for row=1 values are stored on position 1,65,129,193 ...
		shChi2[idxT*rowsB+idxR]=dot[0];
		
		__syncthreads();		

	
		//reduction to some level
		for( j=blockDim.x/2; j>=rowsB; j>>=1) //s/=2
		{
			if(tid<j){
				shChi2[tid]+=shChi2[tid+j];
			}
			__syncthreads();
		}			
			
		//if(row2<shRows){
		if(tid<rowsB){
			//results[row2]=row2;			
			unsigned int row2=blockIdx.x* rowsB+tid;
			//results[row2]=shDot[tid];
		    results[row2]=shChi2[tid];
		}
	}	

}





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
extern "C" __global__ void nChi2EllpackKernel(const float * vals,
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
			val2 = tex1Dfetch(mainVecTexRef,col1);
			
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
			val2 = tex1Dfetch(mainVecTexRef,col1);
			
			chi+= (val1*val2)/(val1+val2+FLT_MIN);
			
		}
		chi=rowSum[row]+shMainSelfSum-4*chi;
		results[row]=labelProd*expf(-shGamma*chi);
	}	

}



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
			val2[sw] = tex1Dfetch(mainVecTexRef,col[sw]);
			
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
			val2 = tex1Dfetch(mainVecTexRef,col1);
			
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
		shMainVals[s]=tex1Dfetch(mainVecTexRef,shMainCols[s]);
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
			val2 = tex1Dfetch(mainVecTexRef,col1);
			
			chi+= (val1-val2)*(val1-val2)/(val1+val2+FLT_MIN);
			
			//vector in Ellpack format might miss some previous columns which are non zero in dense vector
			//we want to "catch up with main dense vector"
			
			while(k<shMainNNZ && prevCol<col1) //prevCol=cols[numRows*k+shMainVecIdx];
			{
				//it is sufficient to add only value of dense vector,
				//because sparse vector values in this position is zero
				chi+=tex1Dfetch(mainVecTexRef,prevCol);
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
			//chi+=tex1Dfetch(mainVecTexRef,prevCol);
		}
		results[row]=labelProd*(1-0.5*chi);
	}	

}


/********************** Evaluators **********************************/


extern "C" __global__ void rbfEllpackILPEvaluator(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* svSelfDot,
										const float* svAlpha,
										const float* svY,
									   float * results,
									   const int num_rows,
									   const float vecSelfDot,
									   const float gamma,
										const int texSel)
{
	
	__shared__ float shGamma;
	__shared__ float shVecSelfDot;
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfDot = vecSelfDot,
		shRows= num_rows;
	}
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in defferent streams
		float dot = texSel==1 ? SpMV_Ellpack_ILP<1>(vals,colIdx,rowLength,row,num_rows): SpMV_Ellpack_ILP<2>(vals,colIdx,rowLength,row,num_rows) ;
		results[row]=svY[row]*svAlpha[row]*expf(-shGamma*(svSelfDot[row]+shVecSelfDot-2*dot));
		//results[row]=dot;
	}	

}

extern "C" __global__ void rbfEllpackEvaluator(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* svSelfDot,
										const float* svAlpha,
										const float* svY,
									   float * results,
									   const int num_rows,
									   const float vecSelfDot,
									   const float gamma,
										const int texSel)
{
	
	__shared__ float shGamma;
	__shared__ float shVecSelfDot;
	__shared__ int shRows;
	
	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfDot = vecSelfDot,
		shRows= num_rows;
	}
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in defferent streams
		float dot = texSel==1 ? SpMV_Ellpack<1>(vals,colIdx,rowLength,row,num_rows): SpMV_Ellpack<2>(vals,colIdx,rowLength,row,num_rows) ;
		results[row]=svY[row]*svAlpha[row]*expf(-shGamma*(svSelfDot[row]+shVecSelfDot-2*dot));
		//results[row]=dot;
	}	

}




extern "C" __global__ void rbfERTILPEvaluator(const float * vals,
									   const int * colIdx, 
									   const int * rowLength, 
									   const float* svSelfDot,
										const float* svAlpha,
										const float* svY,
									   float * results,
									   const int num_rows,
									   const float vecSelfDot,
									   const float gamma,
										const int texSel)
{
	
	__shared__ float shGamma;
	__shared__ float shVecSelfDot;
	__shared__ int shRows;
	__shared__ float shDot[BLOCK_SIZE];
	shDot[threadIdx.x]=0.0;	
	
	if(threadIdx.x==0)
	{
		shGamma = gamma;
		shVecSelfDot = vecSelfDot,
		shRows= num_rows;
	}
	__syncthreads();
	
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		//hack for choosing different texture reference when launch in defferent streams
		texSel==1 ? SpMV_ERTILP<1>(vals,colIdx,rowLength,row,shRows,shDot): SpMV_ERTILP<2>(vals,colIdx,rowLength,row,shRows,shDot) ;
		if(threadIdx.x<rowsB){
			//results[row2]=row2;			
			unsigned int row2=blockIdx.x* rowsB+threadIdx.x;
			//results[row2]=shDot[tid];
		    results[row2]=svY[row2]*svAlpha[row2]*expf(-shGamma*(svSelfDot[row2]+shVecSelfDot-2*shDot[threadIdx.x]));
		}
	}	

}



/*********************** Sparse matrix dense vector multiplication helpers **********/



template<int TexSel> __device__ float fetchTex(int idx);

template<> __device__ float fetchTex<1>(int idx) { return tex1Dfetch(mainVecTexRef,idx); }
template<> __device__ float fetchTex<2>(int idx) { return tex1Dfetch(mainVec2TexRef,idx); }

//cuda kernel funtion for computing SpMV
// Ellpack-R fromat for storing sparse matrix,  uses ILP - prefetch vector elements in registers
// arrays vals and colIdx should be aligned to PREFETCH_SIZE
//Params:
//vals - array of vectors values
//colIdx  - array of column indexes in ellpack-r fromat
//rowLength -array, contains number of nonzero elements in each row
//num_rows -number of vectors
template<int TexSel> __device__ float SpMV_Ellpack_ILP(const float * vals,
									   const int * colIdx, 
									   const int * rowLength,
									   const int row,
									   const int num_rows)
{
		
	__shared__ int shRows;
	if(threadIdx.x==0)
	{
		shRows = num_rows;
	}
	__syncthreads();

	float preVals[PREFETCH_SIZE];
	int preColls[PREFETCH_SIZE];
		
	float dot[PREFETCH_SIZE]={0};

	int maxEl = rowLength[row];
	

	for(int i=0; i<maxEl;i++)
	{
		#pragma unroll
		for(int j=0; j<PREFETCH_SIZE;j++)			
		{
			preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*shRows+row];
			preVals[j]=vals[ (i*PREFETCH_SIZE+j)*shRows+row];
		}
		
		#pragma unroll
		for(int j=0; j<PREFETCH_SIZE;j++){
			//dot[j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
			dot[j]+=preVals[j]* fetchTex<TexSel>(preColls[j]);
		}
		
	}
				
	#pragma unroll
	for(int j=1; j<PREFETCH_SIZE;j++){
		dot[0]+=dot[j];
	}
		
	return dot[0];		
}



template<int TexSel> __device__ float SpMV_Ellpack(const float * vals,
									   const int * colIdx, 
									   const int * rowLength,
									   const int row,
									   const int numRows)
{
	const int num_rows =numRows;
	int maxEl = rowLength[row];
	float dot=0;
		
	int col=-1;
	float val=0;
	int i=0;
	for(i=0; i<maxEl;i++)
	{
		col=colIdx[num_rows*i+row];
		val= vals[num_rows*i+row];
		//dot+=val*tex1Dfetch(mainVecTexRef,col);
		dot+=val*fetchTex<TexSel>(col);
	}

	return dot;
}


template<int TexSel> __device__ void SpMV_ERTILP(const float * vals,
									   const int * colIdx, 
									   const int * rowLength,
									   const int row,
									   const int shRows,
									   volatile float* shDot)
{

	const int tid = threadIdx.x; // index in block
	const int idxR = tid/THREADS_ROW; //row index mapped into block region
	const int idxT = tid%THREADS_ROW; // thread number in Thread Group

	

	float preVals[PREFETCH_SIZE];
	int preColls[PREFETCH_SIZE];
		
	float dot[PREFETCH_SIZE]={0};

	int maxEl = rowLength[row]; //original row length divided by T*PREFETCH

	unsigned int j=0;
	unsigned int arIdx=0;
		
	for(int i=0; i<maxEl;i++)
	{
		
		#pragma unroll
		for( j=0; j<PREFETCH_SIZE;j++)			
		{
			arIdx = (i*PREFETCH_SIZE+j)*shRows*THREADS_ROW+row*THREADS_ROW+idxT;
			preColls[j]=colIdx[arIdx];
			preVals[j]=vals[arIdx];
		}
		
		#pragma unroll
		for( j=0; j<PREFETCH_SIZE;j++){
			dot[j]+=preVals[j]*fetchTex<TexSel>(preColls[j]);
		}
	}
	
	#pragma unroll
	for( j=1; j<PREFETCH_SIZE;j++){
		dot[0]+=dot[j];
	}

	//__syncthreads();	

	// special indexing, values for example for T=4 BlockSize=256
	//for row=0 values are stored on position 0,64,128,192 
	//for row=1 values are stored on position 1,65,129,193 ...
	shDot[idxT*rowsB+idxR]=dot[0];
		
	__syncthreads();		
	
	//reduction to some level
	for( j=blockDim.x/2; j>=rowsB; j>>=1) //s/=2
	{
		if(tid<j){
			shDot[tid]+=shDot[tid+j];
		}
		__syncthreads();
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
