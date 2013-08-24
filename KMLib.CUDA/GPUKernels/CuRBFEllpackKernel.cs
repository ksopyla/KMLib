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
using KMLib.GPU.GPUKernels;

namespace KMLib.GPU
{

    /// <summary>
    /// Class for computing RBF kernel using cuda.
    /// Data are stored in Ellpack-R format.
    /// 
    /// </summary>
    public class CuRBFEllpackKernel : CuVectorKernel, IDisposable
    {

        /// <summary>
        /// Array for self dot product 
        /// </summary>
        float[] selfLinDot;
 

        private float Gamma;



        /// <summary>
        /// cuda device pointer for stroing self linear dot product
        /// </summary>
        private CUdeviceptr selfLinDotPtr;



        public CuRBFEllpackKernel(float gamma)
        {
            linKernel = new LinearKernel();
            Gamma = gamma;
            cudaProductKernelName = "rbfEllpackFormatKernel";
            
            //cudaProductKernelName = "rbfEllpackFormatKernel_shared";
            
            cudaModuleName = "KernelsEllpack.cubin";

            MakeDenseVectorOnGPU = false;
            
        }


        public override float Product(SparseVec element1, SparseVec element2)
        {

            float x1Squere = linKernel.Product(element1, element1);
            float x2Squere = linKernel.Product(element2, element2);

            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));

            return prod;

        }

        public override float Product(int element1, int element2)
        {
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");


            float x1Squere = 0f, x2Squere = 0f, dot = 0f, prod = 0f;

            if (element1 == element2)
            {
                if (DiagonalDotCacheBuilded)
                    return DiagonalDotCache[element1];
                else
                {
                    //all parts are the same
                    // x1Squere = x2Squere = dot = linKernel.Product(element1, element1);
                    //prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));
                    // (x1Squere + x2Squere - 2 * dot)==0 this expresion is equal zero
                    //so we can prod set to 1 beceause exp(0)==1
                    prod = 1f;
                }
            }
            else
            {
                //when element1 and element2 are different we have to compute all parts
                x1Squere = linKernel.Product(element1, element1);
                x2Squere = linKernel.Product(element2, element2);
                dot = linKernel.Product(element1, element2);
                prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));
            }
            return prod;
        }

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
            //return new RbfParameterSelection();
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
            linKernel.ProblemElements = problemElements;
            linKernel.Y = Y;
            linKernel.Init();

            base.Init();

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;

            CudaHelpers.TransformToEllpackRFormat(out vecVals, out vecColIdx, out vecLenght, problemElements);
            
            selfLinDot = linKernel.DiagonalDotCache;

            #region cuda initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);

            
            selfLinDotPtr = cuda.CopyHostToDevice(selfLinDot);

            uint memSize = (uint)(problemElements.Length * sizeof(float));
            //allocate mapped memory for our results
            //CUDARuntime.cudaSetDeviceFlags(CUDARuntime.cudaDeviceMapHost);



            // var e= CUDADriver.cuMemHostAlloc(ref outputIntPtr, memSize, 8);
            //CUDARuntime.cudaHostAlloc(ref outputIntPtr, memSize, CUDARuntime.cudaHostAllocMapped);
            //var errMsg=CUDARuntime.cudaGetErrorString(e);
            //cuda.HostRegister(outputIntPtr,memSize, Cuda)
            outputIntPtr = cuda.HostAllocate(memSize,CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr = cuda.GetHostDevicePointer(outputIntPtr, 0);

            //normal memory allocation
            //outputPtr = cuda.Allocate((uint)(sizeof(float) * problemElements.Length));


            #endregion

            SetCudaFunctionParameters();

            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
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

            cuda.SetParameter(cuFunc, offset, selfLinDotPtr.Pointer);
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
                //free all resources
                cuda.Free(valsPtr);
                valsPtr.Pointer = IntPtr.Zero;
                cuda.Free(idxPtr);
                idxPtr.Pointer = IntPtr.Zero;
                cuda.Free(vecLengthPtr);
                vecLengthPtr.Pointer = IntPtr.Zero;

                cuda.Free(selfLinDotPtr);
                selfLinDotPtr.Pointer = IntPtr.Zero;


                cuda.FreeHost(outputIntPtr);
                //cuda.Free(outputPtr);
                outputPtr.Pointer = IntPtr.Zero;
                cuda.Free(labelsPtr);
                labelsPtr.Pointer = IntPtr.Zero;
                cuda.DestroyTexture(cuLabelsTexRef);

                cuda.Free(mainVecPtr);
                mainVecPtr.Pointer = IntPtr.Zero;

                cuda.DestroyTexture(cuMainVecTexRef);

                cuda.UnloadModule(cuModule);
                cuda.Dispose();
                cuda = null;
            }
        }

        #endregion
    }
}
