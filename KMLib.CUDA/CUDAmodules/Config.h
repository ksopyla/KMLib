/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

#ifndef KERNELS_CONFIG
#define KERNELS_CONFIG

texture<float,1,cudaReadModeElementType> mainVectorTexRef;

//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;


#define BLOCK_SIZE 256

#define WARP_SIZE 32

#define PREFETCH_SIZE 2

#define maxNNZ 100

#endif /* KERNELS_CONFIG */