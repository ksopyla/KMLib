/*
	CUDA kernel for Linear SVM solver based on LIBLINEAR package
	 http://www.csie.ntu.edu.tw/~cjlin/liblinear/
	 Paper: "A Dual Coordinate Descent Method for Large-scale Linear SVM" Hsieh et al., ICML 2008

	

*/



//texture for vector, which is used for matrix vector multiplication
//in SVM, when we have to compute many dot products (one vector with others)
texture<float,1,cudaReadModeElementType> mainVectorTexRef;

//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;



#define BLOCK_SIZE 128

#define WARP_SIZE 32



/*
Based on cuda kernels from 
"Efcient Sparse Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garlandy
December 11, 2008
*/
//
//cuda kernel funtion for computing part of Gradient in method of solving linear SVM,
//grad = w'*xi*yi-1+alpha[i]*C
//this cuda kernel computes only first part w'*xi*yi where w-vector is in tex cache, yi is in tex cache
// xi - is i-th row in matrix containning all elements, matrix is in CSR fromat
//Remarks: based on spmv_csr_vector_kernel from publication above
//Params:
//vals - array of vectors values
//idx  - array of vectros indexes in CSR fromat
//vecPointers -array of pointers(indexes) to idx and vals array to specific vectors
//results - array of results Linear Kernel
//num_rows - number of vectors, stored in CSR matrix format, each vector is stored in one row of matrix

extern "C" __global__ void ComputeDotProd(const float * vals,
									   const int * idx, 
									   const int * vecPointers, 
									   float * results,
									   const int num_rows)
{
	__shared__ float sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
	__shared__ int ptrs[BLOCK_SIZE/WARP_SIZE][2];
		

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
			
			results[row] = tex1Dfetch(labelsTexRef,row)*sdata[threadIdx.x];
		}

			
	}
}



/*
	Implements solve_l2r_l1l2_svc method.
	Cuda kernel computes outer loop in main algorithm.
	Elements matrix is in CSR format
	[vector values]
	[vector indexes]
	[pointers to starting index to vector "i"]
Params:
G - contains precomputed dot product between vector "W" and all obcjects(vectors) multipicated by label, 
	"in out" parameter
QD  - array, diagonal cache QD=Qii+diag
diag - 3 dim array, diag[] = { 0.5/Cneg, 0, 0.5/Cpos} , specific diag is taken by diag[yi+1]
	   if yi=-1 we take diag[0], if yi=1 we take diag[2], diag[1] - is not used
alpha - array with alpha coeficients
paramsC 

*/
extern "C" __global__ void lin_l2r_l1l2_svc_csr_solver_with_gradient(
	float* G,
	const float* QD,
	const float* diag,
	const float* upper_bound,
	const float* alpha,
	const float paramC
	)
{

	int i =  blockDim.x * blockIdx.x + threadIdx.x;
	//grad = W'* element[i]*Y[i]
	float grad = G[i];
	float yi = tex1Dfetch(labelsTexRef,i);
	float alpha_i = alpha[i];

	//in LibLinear we have to compute
	//G=W*element*yi-1+alpah[i]*Dii
	grad = grad-1;
	
	grad+=alpha_i*diag[(int)yi+1];

	//paramC could be in constant cache
	float C = paramC;
	float PG=0;
	
	/*
	below we compute projected gradient, but we don't want use 'if' statemets
		if alpha[i]==0
			PG=min(0,grad[i])
		else if alpha[i]==C
			PG=max(0,grad[i]
		else
			PG=grad[i]
	
	we use formula:
	1. map alpha[i] to 
		-1 - alpha[i]==0
		 0   0<alpha[i]<C
		 1   alpah[i]==C

		 mapAlpha = -1+ floor(alpha[i]/C)+ceil(alpha[i]/C);
		 
		 what if C==infinity? 

	2. compute PG base on maped alpha
	pg= pg+floor(0.5*sign(pg*mapAlpha))*pg
	
	float alpha_C = alpha[i]/C;
	int mapAlpha= -1+floorf(alpha_C)+ceil(alpha_C);
	PG= PG+floor(0.5*sign(PG*mapAlpha))*PG;

	G[i] = PG;
	*/
	/*
	for L2-SVM we can simplify expresion, we don't have to check if alpha[i]=C because C is infinity
	*/
	PG=G[i];
	int signG = signbit(PG);
	int isPosAlpha = isPositive(alpha_i);

	float ifTest=(signG+isPosAlpha+0.0f)/(signG+isPosAlpha+1.0f);
	ifTest= ceilf(ifTest);
	PG=ifTest*PG;

	//if PG< 1e-12, to znaczy że już jesteśmy w optimum,
	//lecz to powinno zachodzić dla wszystkich

	//normaly in paper is Min(Max(alpha-G/QD[i],0.0),U) but in our case U is infinty 
	//so min part was ommitted
	float newAlpha = fmaxf(alpha_i-grad/QD[i],0.0f);
	
}

/*
	function checks if x is positive without 'if' statement

	if x> 0 return 1
	else return 0
*/
__device__ int isPositive(float x)
{ 
	//signbit returns 1 if x is negative and 0 otherwise
	// could be a problem if x=-0.0 ?
  int pos = signbit(x);	//  0-if x>0	1 if x<0	0 if x=0
  int neg = signbit(-x);//  1-if x>0	0 if x<0	0 if x=0
  
  return neg*(1-pos);

  //other solution
  /*
  float test = x>0.0f;
  return 1.0f &&test
  */
}