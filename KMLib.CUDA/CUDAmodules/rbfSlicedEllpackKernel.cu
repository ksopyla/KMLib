/*

author: Krzysztof Sopyła
mail: krzysztofsopyla@gmail.com

Licence: contact with author
web page: http://wmii.uwm.edu.pl/wydzial/kadra/get/143
*/

texture<float,1,cudaReadModeElementType> mainVecTexRef;

//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;

__device__ const int ThreadPerRow=4;
__device__ const int SliceSize=64;


#define VECDIM 597

//gamma parameter in RBF
//__constant__ float GammaDev=0.5;

extern __shared__  float sh_data[];

//Use sliced Ellpack format for computing rbf kernel
//vecVals - vectors values in Sliced Ellpack,
//vecCols - array containning column indexes for non zero elements
//vecLengths  - number of non zero elements in row
//sliceStart   - determine where particular slice starts and ends
//selfDot    - precomputed self dot product
//result  - for final result
//mainVecIdx - index of main vector
//nrows   - number of rows
//ali	  - align
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
	__shared__  float sh_cache[ThreadPerRow*SliceSize];
	
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

  int tx = threadIdx.x;
  int txm = tx % 4; //tx% ThreadPerRow
  int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
  
  //map group of thread to row, in this case 4 threads are mapped to one row
  int row =  thIdx>> 2; // 
  
  if (row < nrRows){
	  float sub = 0.0;
	   int maxRow = (int)ceil(vecLengths[row]/(float)ThreadPerRow);
	  int col=-1;
	  float value =0;
	  int ind=0;

	  for(int i=0; i < maxRow; i++){
		  ind = i*align+sliceStart[blockIdx.x]+tx;
		 
		  col     = vecCols[ind];
		  value = vecVals[ind];
		  sub += value * tex1Dfetch(mainVecTexRef, col);
	  }
  
   sh_cache[tx] = sub;
   __syncthreads();

	volatile float *shMem = sh_cache;
   //for 4 thread per row
  
	if(txm < 2){
	  shMem[tx]+=shMem[tx+2];
	  shMem[tx] += shMem[tx+1];

	  if(txm == 0 ){
		  result[row]=vecLabels[row]*shLabel*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*sh_cache[tx]));
		  //result[row]=vecLabels[row]*shLabel*expf(-GammaDev*(selfDot[row]+shMainSelfDot-2*sh_cache[tx]));
	  }
   }


}//if row<nrRows 

/*
if(txm < 2){
	  sh_cache[tx]+=sh_cache[tx+2];
	  sh_cache[tx] += sh_cache[tx+1];
}

if(thIdx<nrRows ){

  result[thIdx]=vecLabels[thIdx]*shLabel*expf(-shGamma*(selfDot[thIdx]+shMainSelfDot-2*sh_cache[4*tx]));
}*/

  
}//end func



//Use sliced Ellpack format for computing rbf kernel
//vecVals - vectors values in Sliced Ellpack,
//vecCols - array containning column indexes for non zero elements
//vecLengths  - number of non zero elements in row
//sliceStart   - determine where particular slice starts and ends
//selfDot    - precomputed self dot product
//result  - for final result
//mainVecIdx - index of main vector
//nrows   - number of rows
//ali	  - align
extern "C" __global__ void rbfSlicedEllpackKernel_shared(const float *vecVals,
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
	__shared__  float sh_cache[ThreadPerRow*SliceSize];
	
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

	__shared__ float shMainVecAR[VECDIM];
	volatile float *shMainVec =shMainVecAR;
	
	for(int k=threadIdx.x;k<VECDIM;k+=blockDim.x)
		shMainVec[k]=tex1Dfetch(mainVecTexRef,k);
	
	__syncthreads();

  int tx = threadIdx.x;
  int txm = tx % 4; //tx% ThreadPerRow
  int thIdx = (blockIdx.x*blockDim.x+threadIdx.x);
  
  //map group of thread to row, in this case 4 threads are mapped to one row
  int row =  thIdx>> 2; // 
  
  if (row < nrRows){
	  float sub = 0.0;
	   int maxRow = (int)ceil(vecLengths[row]/(float)ThreadPerRow);
	  int labelProd = vecLabels[row]*shLabel;
	  int ind = -1;
	  int col =-1;
	  float value=0;

	  for(int i=0; i < maxRow; i++){
		  ind = i*align+sliceStart[blockIdx.x]+tx;
		  
		  col     = vecCols[ind];
		  value = vecVals[ind];

		  sub += value * shMainVec[col];
	  }
  
   sh_cache[tx] = sub;
   __syncthreads();

	volatile float *shMem = sh_cache;
   //for 4 thread per row
  
	if(txm < 2){
	  shMem[tx]+=shMem[tx+2];
	  shMem[tx] += shMem[tx+1];

	  if(txm == 0 ){
		  //result[row]=vecLabels[row]*shLabel*expf(-0.5*(selfDot[row]+shMainSelfDot-2*sh_cache[tx]));
		  value = labelProd*expf(-shGamma*(selfDot[row]+shMainSelfDot-2*sh_cache[tx]));
		  //value = floorf(value*1000+0.5)/1000;
		  result[row]=value;
		  //result[row]=vecLabels[row]*shLabel*expf(-GammaDev*(selfDot[row]+shMainSelfDot-2*sh_cache[tx]));
	  }
   }


}//if row<nrRows 


  
}//end func


