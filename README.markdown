# KMLib - Kernel Machine Library 

##Description
SVM (Support Vectors Machine) library for .net, main goal is extensibility. You can easily implement your custom kernel or use already implemented (Linear,RBF). Lib includes some SVM kernels(linear,RBF) which use NVIDIA CUDA technology for computing products. 

All vectors are in sparese format due to you can train and test bigger data set (many elements and many object features)

### Author: Krzysztof Sopyła !krzysztofsopyla[at] gmail.com

## Requirements
- .net 4.0 
- CUDA 3.0 for CUDA enabled SVM kernels
- dnAnalitycs (dll  included in project)
- CUDA.net (dll included in project)

## How to use
Look into KMLibUsageApp project.

	IKernel<SparseVector> kernel2 = new CudaRBFKernel(0.5f);



## How to extend
//todo