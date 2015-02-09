
#KMLib - Kernel Machine Library with GPU SVM solver in .Net

##Description 
Support Vectors Machine library in .net (SVM .Net), main goal is extensibility. You can easily implement your custom kernel or use already implemented (Linear,RBF,Chi-Square,Exp Chi-Square). 
Library includes _GPU SVM solver_ for kernels linear,RBF,Chi-Square and Exp Chi-Square which use NVIDIA CUDA technology. It allows for classification big sparse datasets through utilization of matrix sparse format CSR[1], Ellpack-R[2], Sliced EllR-T[3]

More on http://wmii.uwm.edu.pl/~ksopyla/svm-net-with-cuda-kmlib/

__Author: Krzysztof Sopyła <krzysztofsopyla@gmail.com>__

__Author: Sławomir Figiel <fivitti@gmail.com>__

License: MIT

If you use this software in academic publication please cite:

@inproceedings{Sopyla2012,
author = {Sopy\la, Krzysztof and Drozda, Pawe\l and G\’{o}recki, Przemys\law},
title = {SVM with CUDA accelerated kernels for big sparse problems},
booktitle = {Proceedings of the 11th international conference on Artificial Intelligence and Soft Computing – Volume Part I},
series = {ICAISC’12},
year = {2012},
isbn = {978-3-642-29346-7},
pages = {439–447},
numpages = {9},
url = {http://dx.doi.org/10.1007/978-3-642-29347-4_51},
doi = {10.1007/978-3-642-29347-4_51},
acmid = {2342010},
publisher = {Springer-Verlag}}


## Requirements 
- .net 4.0 
- CUDA 3.0 driver for CUDA enabled SVM kernels
- CUDA.net (dll included in project)

## How to use
Look into KMLibUsageApp project for detail.

### Simple classification procedure
1. Read dataset into Problem class
2. Create the kernel
3. Use validation class which does
	1. Create CSVM object for classification 
	2. Train model
	3. Predict

Code should look like this:

	
	//1. Read dataset into problem class, a1a dataset is available
    // at http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a
    
	Problem<SparseVector> train = IOHelper.ReadDNAVectorsFromFile("a1a", 123);
    Problem<SparseVector> test = IOHelper.ReadDNAVectorsFromFile("a1a.t", 123);

	//2. Choose and then create the kernel, RBF kernel with gamma=0.5
    IKernel<SparseVector> kernel = new RbfKernel(0.5f);
	
	//3. Use validation class, last parameter is penalty C in svm solver
	double tempAcc = Validation.TestValidation(train, test, kernel, 8f);

## Flexibility
There are many ways, you can change the classification procedure. 
First you can choose witch svm solver CSVM class use, Plat SMO solver, LIBSVM solver, there are also few experimental solver, or you can implement and easily plug in your solver.
Second you can choose different SVM kernels: Linear, RBF, CudaLinear, CudaRbf or you can implement 
	IKernel<TProblemElements>
interface and use your custom kernel.


## Build procedure
For Fermi card in post build events in KmLib.GPU use
del *.cubin
nvcc cudaSVMKernels.cu linSVMSolver.cu  --cubin   -ccbin "%VS90COMNTOOLS%../../VC/bin" -m32 -arch=sm_21  -use_fast_math
xcopy "$(TargetDir)*.cubin" "$(SolutionDir)Libs" /Y

For other Cuda device
del *.cubin
nvcc cudaSVMKernels.cu linSVMSolver.cu  --cubin   -ccbin "%VS90COMNTOOLS%../../VC/bin" -m32   -use_fast_math
xcopy "$(TargetDir)*.cubin" "$(SolutionDir)Libs" /Y

If you work on 64bit system you can also change -m32 to -m64

References
[1] „SVM with CUDA Accelerated Kernels for Big Sparse Problems”, K. Sopyła, P. Drozda, P. Górecki (2012), Lecture Notes in Artificial Intelligence, Springer Berlin / Heidelberg
[2] "Support vector machines on gpu with sparse matrix format" Tsung-Kai Lin and Shao-Yi Chien. Machine Learning and Applications, Fourth International Conference on,
0:313–318, 2010
[3]

