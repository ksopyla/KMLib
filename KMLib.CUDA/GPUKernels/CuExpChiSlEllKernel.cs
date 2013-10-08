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
    /// Class for computing Exp(-gamma*chi2(x,y)) kernel using cuda.
    /// Data are stored in sliced Ellpack-R format.
    /// 
    /// </summary>
    public class CuExpChiSlEllKernel : CuVectorKernel, IDisposable
    {

        /// <summary>
        /// Array for self dot product 
        /// </summary>
        float[] selfSum;


        private float Gamma;



        /// <summary>
        /// cuda device pointer for stroing self linear dot product
        /// </summary>
        private CUdeviceptr selfSumPtr;
        private int sliceSize;
        private int threadsPerRow;
        
        //private int blockSize;
        
        private int align;
        private CUdeviceptr sliceStartPtr;
        private int blockSize;




        public CuExpChiSlEllKernel(float gamma)
        {
            
            Gamma = gamma;

            cudaProductKernelName = "expChi2SlEllKernel";

            
            cudaModuleName = "KernelsSlicedEllpack.cubin";

            
            threadsPerRow =  4;
            sliceSize =  64;
        }


        public override void SetMemoryForDenseVector(int mainIndex)
        {
            base.SetMemoryForDenseVector(mainIndex);



        }


        public override float Product(SparseVec element1, SparseVec element2)
        {

            float chi = ChiSquaredKernel.ChiSquareDist(element1, element2);

            return (float)Math.Exp(-Gamma * chi);

        }

        public override float Product(int element1, int element2)
        {
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");


            float prod = 0f;

            if (element1 == element2)
            {
                //all parts are the same
                //so we can prod set to 1 beceause exp(0)==1
                prod = 1f;

            }
            else
            {
                //when element1 and element2 are different we have to compute all parts
                float chi = ChiSquaredKernel.ChiSquareDist(problemElements[element1], problemElements[element2]);
                prod = (float)Math.Exp(-Gamma * chi);
            }
            return prod;
        }

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
            //return new RbfParameterSelection();
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

            CudaHelpers.TransformToSlicedEllpack(out vecVals, out vecColIdx, out sliceStart, out vecLenght, problemElements, threadsPerRow, sliceSize);

            selfSum = problemElements.AsParallel().Select(x => x.Values.Sum()).ToArray();

            #region cuda initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);
            sliceStartPtr = cuda.CopyHostToDevice(sliceStart);
            
            labelsPtr = cuda.CopyHostToDevice(Y);
            //!!!!!
            selfSumPtr = cuda.CopyHostToDevice(selfSum);

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


            cuda.SetParameter(cuFunc, offset, selfSumPtr.Pointer);
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

            cuda.SetParameter(cuFunc, offset, Gamma);
            offset += sizeof(float);

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

                cuda.Free(selfSumPtr);
                selfSumPtr.Pointer = IntPtr.Zero;



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
            return "Cu ExpChi2 Sll-Ell";
        }
    }
}
