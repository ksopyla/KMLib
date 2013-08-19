/*

author: Krzysztof Sopyła
mail: krzysztofsopyla@gmail.com

License: contact with author
web page: http://wmii.uwm.edu.pl/wydzial/kadra/get/143
*/


#ifndef KERNELS_CONFIG
#define KERNELS_CONFIG

texture<float,1,cudaReadModeElementType> mainVecTexRef;

//texture fo labels assiociated with vectors
texture<float,1,cudaReadModeElementType> labelsTexRef;


#define BLOCK_SIZE 256

#define WARP_SIZE 32

#define PREFETCH_SIZE 2

#endif /* KERNELS_CONFIG */