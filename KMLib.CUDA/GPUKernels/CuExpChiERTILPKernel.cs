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
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Runtime.InteropServices;
using KMLib.Kernels;
using KMLib.Helpers;
using KMLib.GPU.GPUKernels;

namespace KMLib.GPU
{

      /// <summary>
    /// Class for computing Exp(-gamma*chi2(x,y)) kernel using cuda.
    /// Data are stored in ERTILP format with utilization of ILP technique.
    /// T threads are operate on single row and prefetch PrefetchSize values
    /// Values and columns are algined to T*PrefetchSize
    /// 
    /// </summary>
    public class CuExpChiERTILPKernel : CuVectorKernel, IDisposable
    {

        /// <summary>
        /// Array for self dot product 
        /// </summary>
        float[] selfSum;
 

        private float Gamma;



        /// <summary>
        /// cuda device pointer for storing self sum of each vector
        /// </summary>
        private CUdeviceptr selfSumPtr;
        private int ThreadsPerRow;
        private int Prefetch;



        public CuExpChiERTILPKernel(float gamma)
        {
           
            Gamma = gamma;
            cudaProductKernelName = "expChi2ERTILP";
            cudaModuleName = "KernelsEllpack.cubin";
            MakeDenseVectorOnGPU = false;

            ThreadsPerRow = 4;
            Prefetch =2;
            
        }




        public override void SetMemoryForDenseVector(int mainIndex)
        {
            if (MakeDenseVectorOnGPU)
            {
                vecBuilder.BuildDenseVector(mainIndex);
            }else
                base.SetMemoryForDenseVector(mainIndex);
        }


        public override void Init()
        {
           
            base.Init();

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;

            
            //change the blocksPerGrid, because we launch many threads per row
            blocksPerGrid =(int) Math.Ceiling((ThreadsPerRow * problemElements.Length+0.0) / threadsPerBlock);


             int align = ThreadsPerRow*Prefetch;
            CudaHelpers.TransformToERTILPFormat(out vecVals, out vecColIdx, out vecLenght, problemElements,align,ThreadsPerRow);


            selfSum = problemElements.AsParallel().Select(x => x.Values.Sum()).ToArray();


            #region cuda initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);


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

            CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuLabelsTexRef, cudaLabelsTexRefName, Y, ref labelsPtr);

            if (MakeDenseVectorOnGPU)
            {
                vecBuilder = new EllpackDenseVectorBuilder(cuda, mainVecPtr, valsPtr, idxPtr, vecLengthPtr, problemElements.Length, problemElements[0].Dim);
                vecBuilder.Init();
            }

        }



        protected override void SetCudaFunctionParameters()
        {

            #region cuda set function parameters
            cuda.SetFunctionBlockShape(cuFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, vecLengthPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, selfSumPtr.Pointer);
            offset += IntPtr.Size;

            kernelResultParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, outputPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)problemElements.Length);
            offset += sizeof(int);

            mainVecIdxParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)mainVectorIdx);
            offset += sizeof(int);

            cuda.SetParameter(cuFunc, offset, Gamma);
            offset += sizeof(float);

            cuda.SetParameterSize(cuFunc, (uint)offset);


            #endregion
        }



        #region IDisposable Members

        public void Dispose()
        {
            if (cuda != null)
            {
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
            return "Cu ExpChi2 ERTILP";
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
    }
}
