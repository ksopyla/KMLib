﻿/*
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
    /// Represents Chi^2 Kernel for computing product between two histograms
    /// 
    /// K(x,y)= Sum( (xi*yi)/(xi+yi))
    /// 
    /// vectors should contains positive numbers(like histograms does) and should be normalized
    /// sum(xi)=1
    /// Data are stored in Ellpack-R format.
    /// 
    /// </summary>
    public class CuNChi2EllKernel : CuVectorKernel, IDisposable
    {


        public CuNChi2EllKernel()
        {
            //linKernel = new LinearKernel();
            //chiSquared = new ChiSquaredNormKernel();

            cudaProductKernelName = "nChi2EllpackKernel";
            //cudaProductKernelName = "nChi2EllpackKernel_old";
            
            cudaModuleName = "KernelsEllpack.cubin";
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


        public override void Init()
        {
            
            //chiSquared.ProblemElements = problemElements;
            //chiSquared.Y = Y;
            //chiSquared.Init();

            base.Init();

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;

            CudaHelpers.TransformToEllpackRFormat(out vecVals, out vecColIdx, out vecLenght, problemElements);

            #region cuda initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);

            uint memSize = (uint)(problemElements.Length * sizeof(float));
            //allocate mapped memory for our results
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


        public override void SetMemoryForDenseVector(int mainIndex)
        {
            if (MakeDenseVectorOnGPU)
            {
                vecBuilder.BuildDenseVector(mainIndex);
            }
            else
                base.SetMemoryForDenseVector(mainIndex);
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
            
            kernelResultParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, outputPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)problemElements.Length);
            offset += sizeof(int);

            mainVecIdxParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)mainVectorIdx);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFunc, (uint)offset);


            #endregion
        }



        #region IDisposable Members

        public void Dispose()
        {
            if (cuda != null)
            {

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
            return "Cuda nChi^2 Ellpack";
        }
    }

}
