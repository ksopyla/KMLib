/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#include <float.h>

#include <Config.h>

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
	__shared__ int shRows;

	//__shared__ float shDot[PREFETCH_SIZE*BLOCK_SIZE];
	//for(int j=0; j<PREFETCH_SIZE;j++){
	//	//shDot[threadIdx.x*PREFETCH_SIZE+j]=0.0;
	//	shDot[threadIdx.x+PREFETCH_SIZE*j]=0.0;
	//}
			
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}
		
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		float preVals[PREFETCH_SIZE];
		int preColls[PREFETCH_SIZE];
		//float preVecVals[PREFETCH_SIZE];
		//float dot=0;
		float dot[PREFETCH_SIZE]={0};

		int maxEl = rowLength[row];
		//how many elements are the rest after division
		//int rest = maxEl%PREFETCH_SIZE;
		//int mainIter = ceilf( (maxEl+0.0)/PREFETCH_SIZE);
		for(int i=0; i<maxEl;i++)
		{
			//int subIter= min(maxEl-i*PREFETCH_SIZE,PREFETCH_SIZE);
			
			#pragma unroll
			for(int j=0; j<PREFETCH_SIZE;j++)			
			{
				preColls[j]=colIdx[ (i*PREFETCH_SIZE+j)*shRows+row];
				preVals[j]=vals[ (i*PREFETCH_SIZE+j)*shRows+row];
			}

			#pragma unroll
			for(int j=0; j<PREFETCH_SIZE;j++){
				//dot+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
				dot[j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
				
				//bank confilict
				//shDot[threadIdx.x*PREFETCH_SIZE+j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
				//bank confilict free
				//shDot[threadIdx.x+PREFETCH_SIZE*j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
			}
		}
		//
		//float dot = 0;
		//#pragma unroll
		//for(int j=0; j<PREFETCH_SIZE;j++){
		//		//dot+=shDot[threadIdx.x*PREFETCH_SIZE+j];
		//		dot+=shDot[threadIdx.x+PREFETCH_SIZE*j];

		//}
		//results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));

		#pragma unroll
		for(int j=1; j<PREFETCH_SIZE;j++){
				dot[0]+=dot[j];
		}
		results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot[0]));
		
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
		shMainVec[k]=tex1Dfetch(mainVecTexRef,k);
	
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
				//dot+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
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
		shMainVec[k]=tex1Dfetch(mainVecTexRef,k);
	
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
			dot+=val*tex1Dfetch(mainVecTexRef,col);
		}

	

		//results[row]=shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
		//results[row]=labelProd*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		//results[row]=labelProd*exp(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));

		//val = expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		//val = floorf(val*10000+0.5)/10000;
		results[row]=labelProd*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));;
		
		//results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
	}	

}


extern "C" __global__ void rbfEllpackFormatKernel_ILP_sum(const float * vals,
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
		
		int col[2]={-1,-1};
		float val[2]={0,0};
		float sum[2] = {0, 0};
		int i=0;
		int sw=0;
		for(i=0; i<maxEl;i++)
		{
			sw = i%2;
			//sw = i&1; //equals i%2
			col[sw]=colIdx[num_rows*i+row];
			val[sw]= vals[num_rows*i+row];
			sum[sw]+=val[sw]*tex1Dfetch(mainVecTexRef,col[sw]);
		}
		dot=sum[0]+sum[1];


		/*int col[PREFETCH_SIZE];
		float val[PREFETCH_SIZE];
		float sum[PREFETCH_SIZE];
		int i=0;
		int sw=0;
		for(i=0; i<maxEl;i++)
		{
			sw = i%PREFETCH_SIZE;
			col[sw]=colIdx[num_rows*i+row];
			val[sw]= vals[num_rows*i+row];
			sum[sw]+=val[sw]*tex1Dfetch(mainVecTexRef,col[sw]);
		}
		for(int k=0; k<PREFETCH_SIZE;k++)
			dot+=sum[k];*/

		results[row]=labelProd*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		
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
