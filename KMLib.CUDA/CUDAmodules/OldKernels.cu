/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/


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
	//	shDot[threadIdx.x*PREFETCH_SIZE+j]=0.0;
	//}
			
	//myTex1Dfetch<1>(5);
	
	if(threadIdx.x==0)
	{
		shRows = num_rows;
		shMainVecIdx=mainVecIndex;
		shGamma = gamma;
		shMainSelfDot = selfDot[shMainVecIdx];
		shLabel = tex1Dfetch(labelsTexRef,shMainVecIdx);
	}
	__syncthreads();
		
	const int row   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index

	if(row<shRows)
	{
		float preVals[PREFETCH_SIZE];
		int preColls[PREFETCH_SIZE];
		//float preVecVals[PREFETCH_SIZE];
		//float dot=0;
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
				//dot+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
				dot[j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
				//shDot[threadIdx.x*PREFETCH_SIZE+j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
			}
			
		}
		
		
		//volatile float *shMem = shDot;
		//float dot = 0;
		//#pragma unroll
		//for(int j=1; j<PREFETCH_SIZE;j++){
		//		//dot+=shDot[threadIdx.x*PREFETCH_SIZE+j];
		//		shDot[threadIdx.x*PREFETCH_SIZE+0]+=shDot[threadIdx.x*PREFETCH_SIZE+j];
		//}

		//__syncthreads();
		//results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));
		//results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*shDot[threadIdx.x*PREFETCH_SIZE+0]));

		#pragma unroll
		for(int j=1; j<PREFETCH_SIZE;j++){
				dot[0]+=dot[j];
		}
		results[row]=tex1Dfetch(labelsTexRef,row)*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot[0]));
		
	}	

}

extern "C" __global__ void rbfEllRTILP_old(const float * vals,
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

	const int tid = threadIdx.x; // index in block
	const int idxR = tid/THREADS_ROW; //row index mapped into block region
	const int idxT = tid%THREADS_ROW; // thread number in Thread Group

	if(row<shRows)
	{
		float preVals[PREFETCH_SIZE];
		int preColls[PREFETCH_SIZE];
		
		float dot[PREFETCH_SIZE]={0};

		//todo: move to shared mem
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
				dot[j]+=preVals[j]*tex1Dfetch(mainVecTexRef,preColls[j]);
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
			
		//if(row2<shRows){
		if(tid<rowsB){
			//results[row2]=row2;			
			unsigned int row2=blockIdx.x* rowsB+tid;
			//results[row2]=shDot[tid];
		    results[row2]=tex1Dfetch(labelsTexRef,row2)*shLabel*expf(-shGamma*(selfDot[row2]+shMainSelfDot-2*shDot[tid]));
		}
	}//if row<nrRows	

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


extern "C" __global__ void rbfEllpackFormatKernel_old(const float * vals,
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
	const int num_rows =numRows;
	if(row<shRows)
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

		
		//float dot = SpMV_Ellpack<1>(vals,colIdx,rowLength,row,numRows);
		results[row]=labelProd*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*dot));;
	}	

}