



texture<float,1,cudaReadModeElementType> mainVecTexRef;




extern "C" __global__ void rbfSlicedEllpack(float * a,int N)
{

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (idx<N) a[idx] = a[idx]+tex1Dfetch(simpleTexRef,idx);
}