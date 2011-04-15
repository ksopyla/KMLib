/*
	CUDA kernel for Linear SVM solver based on LIBLINEAR package
	 http://www.csie.ntu.edu.tw/~cjlin/liblinear/
	 Paper: "A Dual Coordinate Descent Method for Large-scale Linear SVM" Hsieh et al., ICML 2008

	

*/



//texture for vector, which is used for matrix vector multiplication
//in SVM, when we have to compute many dot products (one vector with others)
texture<float,1,cudaReadModeElementType> mainVectorTexRef;


texture<float,1,cudaReadModeElementType> deltasTexRef;

//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;


//constant array for diagonal shift in L2-SVM diag_shift[]={ 0.5/Cn, 0 , 0.5/Cp}
//where Cn,Cp penalty parameters for negative elements and positive
__device__  __constant__ float diag_shift[3];

//main vector dimension
__device__ __constant__ int Dim;


// 1/square(Dim)
__device__ __constant__ float stepScaling=0.0f;

#define BLOCK_SIZE 128

#define WARP_SIZE 32



/*
	function checks if x is positive without 'if' statement

	if x> 0 return 1
	else return 0
*/
__device__ int isPositive(float x)
{ 
	//signbit returns 1 if x is negative and 0 otherwise
	// could be a problem if x=-0.0 ?
/* 
 int pos = signbit(x);	//  0-if x>0	1 if x<0	0 if x=0
  int neg = signbit(-x);//  1-if x>0	0 if x<0	0 if x=0
  
  return neg*(1-pos);
*/
  //other solution
 
  float test = x>0.0f;
  return 1.0f &&test;
  
}



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
			sum += vals[jj] * tex1Dfetch(mainVectorTexRef,idx[jj]-1); //all indexes starts from 1, but mainVector starts from 0

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

			
	}//end for
}



/*
	Implements solve_l2r_l1l2_svc method.
	Cuda kernel computes outer loop in main algorithm.
	Elements matrix is in CSR format
	[vector values]
	[vector indexes]
	[pointers to starting index to vector "i"]

	N - number of objects for classification
	L - number of ovject features 
Params:

QD  - array of size N, diagonal cache QD=Qii+diag
diag - 3 dim array, diag[] = { 0.5/Cneg, 0, 0.5/Cpos} , specific diag is taken by diag[yi+1]
	   if yi=-1 we take diag[0], if yi=1 we take diag[2], diag[1] - is not used
alpha - array with alpha coeficients of size N
paramsC - 
G - array of size N,contains precomputed dot product between vector "W" and all obcjects(vectors) multipicated by label, 
	after computation G stores al projected gradient for chcecking stop creterion, this is  "in out" parameter
deltas - array of size N, contains step in each dimension, out parameter

*/
extern "C" __global__ void lin_l2r_l2_svc_solver_with_gradient(
	const float* QD,
	float* alpha,
	float* G,
	float* deltas,
	const int elements
	)
{

	int i =  blockDim.x * blockIdx.x + threadIdx.x;
	//grad = W'* element[i]*Y[i]

	if(i<elements){
	float grad = G[i];
	
	float yi = tex1Dfetch(labelsTexRef,i);
	float alpha_i = alpha[i];

	//in LibLinear we have to compute
	//G=W*element*yi-1+alpah[i]*Dii
	grad = grad-1;
	
	grad+=alpha_i*diag_shift[(int)yi+1];

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
	PG=grad;
	

	/*
	this computes PG without using 'if' statements, line 472 in lin solver

	

	if alpha_i=0
		PG= min(0,grad)
	else
		PG=grad

	we could change this to
	if alpha_i=0 and grad <0
		PG = grad
	if alpha_i=0 and grad >=0
		PG = 0
	if alpha_i>0 and grad <0
		PG = grad
	if alpha_i=> and grad >=0
		PG = grad

	we can set PG using formula:
	signG=1 if grad<0
	signG=0 if grad>=0

	isPosAlpha=1 if alpha_i>0
	isPosAlpha=0 if alpha_i<=0

	float ifTest=(signG+isPosAlpha+0.0f)/(signG+isPosAlpha+1.0f);
	ifTest= ceilf(ifTest);
	PG=ifTest*PG;
	*/
	int signG = signbit(PG);
	int isPosAlpha = isPositive(alpha_i);
	PG=PG*ceilf((signG+isPosAlpha+0.0f)/(signG+isPosAlpha+1.0f));

	//if PG< 1e-12, to znaczy że już jesteśmy w optimum,
	//lecz to powinno zachodzić dla wszystkich
	//we store 
	
	G[i]=PG;
	//G[i]=diag_shift[(int)yi+1]-5;

	//we should compute delta only if PG>0
	//but we want to omit branching so we computed delta but
	//grad = PG  if PG=0 then delta ==0
	grad=PG;
	
//normaly in paper is Min(Max(alpha-G/QD[i],0.0),U) but in our case U is infinty 
	//so min part was ommitted
	float deltaAlpha = fmaxf(alpha_i-grad/(QD[i]+diag_shift[(int)yi+1] ),0.0f)-alpha_i;
	
	//stepScaling - scaling parameter
	//deltas[i]=deltaAlpha*yi*stepScaling;
	//set new alpha
	//alpha[i]=alpha_i+deltaAlpha*stepScaling;

	deltas[i]=deltaAlpha*yi;
	alpha[i]=alpha_i+deltaAlpha;
}//end if(i<elements)
}


//cuda kernel funtion for updating  W-vector in method of solving linear SVM,
//the idea is almost the same as in CudaDotProd function, 
//each warp computes multiplication between step vector (D) and each column
//
//
//					   | x11 x12 .. x1n|
//					   | x21 x22 .. x2n|
//	[D1, D2, ..., Dl]* | .    .  ..  . |
//					   | .    .  ..  . |
//					   | xl1 xl2 .. xln|
// l- number of elements
// n - vector dim
// we have to compute sums  sum_k = Sum_i (D_i*x_ik)
// sum_1 = D1*x11+ D2*x21 +...+Dl*xl1
// sum_2 =
// ...
// sum_l
// when we have sums we can compute change for vector W
// W[k]+= sum_k
//
//matrix is in CSC fromat
//Params:
//vals - array of vectors values, column order
//idx  - array of vectros indexes in CSC fromat (compact sparse column)
//vecPointers -array of pointers(indexes) to idx and vals array, indicates start and end of specific column
//W - computed W vector - array of size dim, 
//num_cols - number of vectors, stored in CSC matrix format, 
extern "C" __global__ void update_W(const float * vals,
									   const int * idx, 
									   const int * vecPointers, 
									   float * W,
									   const int num_rows)
{

//todo: change all  "*rows" into columns
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
			sum += vals[jj] * tex1Dfetch(deltasTexRef,idx[jj]);

		// reduce local sums to row sum (ASSUME: warpsize 32)
		sdata[threadIdx.x] = sum;
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16]; __syncthreads(); 
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2]; __syncthreads();
		sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1]; __syncthreads();
	   

		

		// first thread writes warp result
		if (thread_lane == 0){
			
			//results[row] = tex1Dfetch(labelsTexRef,row)*sdata[threadIdx.x];
			W[row] =sdata[threadIdx.x];
		}

			
	}
}



