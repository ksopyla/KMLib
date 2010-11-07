#KMLib - Kernel Machine Library



##Description ##
SVM (Support Vectors Machine) library for .net, main goal is extensibility. You can easily implement your custom kernel or use already implemented (Linear,RBF). Lib includes some SVM kernels(linear,RBF) which use NVIDIA CUDA technology for computing products. 

All vectors are in sparese format due to you can train and test bigger data set (many elements and many object features)

__Author: Krzysztof Sopyła <krzysztofsopyla@gmail.com>__

## Requirements ##
- .net 4.0 
- CUDA 3.0 for CUDA enabled SVM kernels
- dnAnalitycs (dll  included in project)
- CUDA.net (dll included in project)

## How to use ##
Look into KMLibUsageApp project for detail.

### Simple classification procedure
1. Read dataset into Problem class
2. Create the kernel
3. Use validation class witch does
	1. Create CSVM object for classification 
	2. Train model
	3. Predict

Code should look like this:

	Problem<SparseVector> train = IOHelper.ReadDNAVectorsFromFile("a1a", 123);
    Problem<SparseVector> test = IOHelper.ReadDNAVectorsFromFile("a1a.t", 123);
	
	// Read dataset into problem class, a1a dataset is available
    // at http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a
    
	Problem<SparseVector> train = IOHelper.ReadDNAVectorsFromFile("a1a", 123);
    Problem<SparseVector> test = IOHelper.ReadDNAVectorsFromFile("a1a.t", 123);

	// Choose and then create the kernel, RBF kernel with gamma=0.5
    IKernel<SparseVector> kernel = new RbfKernel(0.5f);
	
	\/\/3. Use validation class, last parameter is penalty C in svm solver
	double tempAcc = Validation.TestValidation(train, test, kernel, 8f);

### Flexible
There are many ways, you can change the classification procedure. 
First you can choose witch svm solver CSVM class use, Plat SMO solver, LIBSVM solver, there are also few experimental solver, or you can implement and easily plug in your solver.
Second you can choose different SVM kernels: Linear, RBF, CudaLinear, CudaRbf or you can implement 
	IKernel<TProblemElements>
interface and use your custom kernel.



## How to extend ##
//todo