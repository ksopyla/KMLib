/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Runtime.InteropServices;
using KMLib.Kernels;
using KMLib.Helpers;

namespace KMLib.GPU
{

    /// <summary>
    /// Represents Chi^2 Kernel for computing product between two histograms
    /// 
    /// K(x,y)= Sum( (xi*yi)/(xi+yi))
    /// 
    /// vectors should contains positive numbers(like histograms does) and should be normalized
    /// sum(xi)=1
    /// Data are stored in SERTILP format 
    /// 
    /// </summary>
    public class CuNChi2SERTILPKernel : CuVectorKernel, IDisposable
    {

        
        private int sliceSize;
        private int threadsPerRow;
        
        //private int blockSize;
        
        private int align;
        private CUdeviceptr sliceStartPtr;
        private int blockSize;
        
        
        /// <summary>
        /// How many nonzero are prefech by each thread
        /// </summary>
        private int preFechSize;




        public CuNChi2SERTILPKernel()
        {

            cudaProductKernelName = "nChi2SERTILP";
            //cudaProductKernelName = "rbfSlicedEllpackKernel_shared";

            cudaModuleName = "KernelsSlicedEllpack.cubin";

            threadsPerRow =  4;
            sliceSize =  64;
            preFechSize = 2;

            //threadsPerRow = 2;
            //sliceSize = 4;
        }


        public override void SetMemoryForDenseVector(int mainIndex)
        {
            base.SetMemoryForDenseVector(mainIndex);
        }

        public override void Init()
        {
            base.Init();

            blockSize = threadsPerRow * sliceSize;
            int N = problemElements.Length;
            blocksPerGrid = (int)Math.Ceiling(1.0 * N * threadsPerRow / blockSize);

            align = (int)Math.Ceiling(1.0 * sliceSize * threadsPerRow / 64) * 64;
            

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;
            int[] sliceStart;

            CudaHelpers.TransformToSERTILP(out vecVals, out vecColIdx, out sliceStart, out vecLenght, problemElements, threadsPerRow, sliceSize,preFechSize);


            #region cuda initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);
            sliceStartPtr = cuda.CopyHostToDevice(sliceStart);
            
            labelsPtr = cuda.CopyHostToDevice(Y);
            
            uint memSize = (uint)(problemElements.Length * sizeof(float));
            
            outputIntPtr = cuda.HostAllocate(memSize,CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr = cuda.GetHostDevicePointer(outputIntPtr, 0);

            //normal memory allocation
            //outputPtr = cuda.Allocate((uint)(sizeof(float) * problemElements.Length));


            #endregion

            SetCudaFunctionParameters();

            //allocate memory for main vector, size of this vector is the same as dimension, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[problemElements[0].Dim + 1];
            CudaHelpers.FillDenseVector(problemElements[0], mainVector);

            CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector, ref mainVecPtr);

           // CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuLabelsTexRef, cudaLabelsTexRefName, Y, ref labelsPtr);


        }



        protected override void SetCudaFunctionParameters()
        {

            #region cuda set function parameters
            cuda.SetFunctionBlockShape(cuFunc,blockSize, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, vecLengthPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, sliceStartPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, labelsPtr.Pointer);
            offset += IntPtr.Size;
            kernelResultParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, outputPtr.Pointer);
            offset += IntPtr.Size;

            mainVecIdxParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)mainVectorIdx);
            offset += sizeof(int);

            cuda.SetParameter(cuFunc, offset, (uint)problemElements.Length);
            offset += sizeof(int);

            cuda.SetParameter(cuFunc, offset, align);
            offset += sizeof(int);


            cuda.SetParameterSize(cuFunc, (uint)offset);


            #endregion
        }

        #region IDisposable Members

        public void Dispose()
        {
            if (cuda != null)
            {
               
                cuda.Free(sliceStartPtr);
                
                DisposeResourses();
                
                cuda.UnloadModule(cuModule);

                base.Dispose();
                cuda.Dispose();
                cuda = null;
            }
        }

        #endregion
        public override string ToString()
        {
            return "Cu nChi2_SERTILP";
        }

        public override float Product(SparseVec element1, SparseVec element2)
        {
            //return chiSquared.Product(element1, element2);
            return ChiSquaredNormKernel.ChiSquareNormDist(element1, element2);
        }

        public override float Product(int element1, int element2)
        {
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");

            return ChiSquaredNormKernel.ChiSquareNormDist(problemElements[element1], problemElements[element2]);
        }

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
        }
    }
}
