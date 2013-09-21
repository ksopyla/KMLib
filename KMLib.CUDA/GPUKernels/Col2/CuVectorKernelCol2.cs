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
using KMLib.Kernels;
//using dnAnalytics.LinearAlgebra;//
using GASS.CUDA.Types;
using GASS.CUDA;
using System.Runtime.InteropServices;
using System.IO;
using KMLib.Helpers;
using System.Diagnostics;
using KMLib.GPU.GPUKernels;

namespace KMLib.GPU.GPUKernels.Col2
{

    /// <summary>
    /// Based clas for all cuda enable svm kernel with computing 2 kernel matrix columns at once
    /// This class encapsulates all necessary variables for cuda.net library
    /// </summary>
    public abstract class CuVectorKernelCol2 : VectorKernel<SparseVec>, IDisposable
    {

        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected string cudaModuleName = "";
        

        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected string cuVecITexRefName = "VecI_TexRef";


        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected string cuVecJTexRefName = "VecJ_TexRef";


        /// <summary>
        /// cuda function name for computing product
        /// </summary>
        protected string cudaProductKernelName;
        #endregion




        /// <summary>
        /// linear kernel for normal product
        /// </summary>
        protected LinearKernel linKernel;


        /// <summary>
        /// vector for computing kernel product with other vectors
        /// </summary>
        /// <remarks>all the time this vector will be modified and copied to cuda array</remarks>
        protected float[] VectorI;

        protected float[] VectorJ;

        /// <summary>
        /// native pointer to output memory region
        /// </summary>
        protected IntPtr outputIntPtr;


        /// <summary>
        /// parameter offset in cuda function kernel for changing mainVectorIdx
        /// </summary>
        protected int IdxIParamOffset=-1;

        protected int IdxJParamOffset = -1;

        /// <summary>
        /// parameter offset in cuda kernel for result 
        /// </summary>
        protected int kernelResultParamOffset=-1;
        /// <summary>
        /// index of current main problem element (vector)
        /// </summary>
        protected uint IVectorIdx = 0;


        protected uint JVectorIdx = 0;


        /// <summary>
        /// average vector lenght, its only a heuristic
        /// </summary>
        protected int avgVectorLenght = 50;

        protected int threadsPerBlock = CUDAConfig.XBlockSize;

        /// <summary>
        /// indicates how many blocks pre grid we create for cuda kernel launch
        /// </summary>
        protected int blocksPerGrid = -1;

        #region cuda types

        /// <summary>
        /// Cuda .net class for cuda opeation
        /// </summary>
        internal CUDA cuda;


        /// <summary>
        /// cuda loaded module
        /// </summary>
        protected CUmodule cuModule;

        /// <summary>
        /// cuda kernel function
        /// </summary>
        protected CUfunction cuFunc;


        /// <summary>
        /// Cuda device pointer to vectors values
        /// </summary>
        protected CUdeviceptr valsPtr;
        /// <summary>
        /// cuda devie pointer to vectors indexes
        /// </summary>
        protected CUdeviceptr idxPtr;
        /// <summary>
        /// cuda device pointer to vectors lenght
        /// </summary>
        protected CUdeviceptr vecLengthPtr;

        /// <summary>
        /// cuda device pointer for output
        /// </summary>
        public CUdeviceptr outputPtr;

        /// <summary>
        /// cuda reference to texture for main "I" vector problem 
        /// </summary>
        protected CUtexref cuVecI_TexRef;

        /// <summary>
        /// cuda reference to texture for main "J" vector problem 
        /// </summary>
        protected CUtexref cuVecJ_TexRef;


        /// <summary>
        /// cuda device pointer to "I" vector problem 
        /// </summary>
        protected CUdeviceptr VecIPtr;

        /// <summary>
        /// cuda device pointer to "J" vector problem 
        /// </summary>
        protected CUdeviceptr VecJPtr;

       

        /// <summary>
        /// cuda pointer to labels, neded for coping to texture
        /// </summary>
        protected CUdeviceptr labelsPtr;
        
        /// <summary>
        /// cuda context for svm kernel
        /// </summary>
        protected CUcontext cuCtx;


        #endregion

        protected bool MakeDenseVectorOnGPU = false;

        

        public override SparseVec[] ProblemElements
        {
            set
            {
                if (value == null) throw new ArgumentNullException("value");
                //linKernel.ProblemElements = value;

                base.ProblemElements = value;

                blocksPerGrid = (value.Length + threadsPerBlock - 1) / threadsPerBlock;
                //blocksPerGrid = (2*value.Length + threadsPerBlock - 1) / threadsPerBlock;
            }
        }


        public override void AllProducts(int element1, float[] results)
        {

            throw new NotSupportedException("Computing one kernel column not supported");
        }

        

        public override void AllProducts(int i,int j, float[][] results)
        {

            

            SetMemoryForDenseVector(i,j);

            
            //set the last parameter for kernel
            IVectorIdx =(uint) i;
            JVectorIdx = (uint) j;
            
            cuda.SetParameter(cuFunc, IdxIParamOffset, IVectorIdx);
            cuda.SetParameter(cuFunc, IdxJParamOffset, JVectorIdx);
            
            
            cuda.Launch(cuFunc, blocksPerGrid, 1);

            
            cuda.SynchronizeContext();

            //copy resulsts form device to host
            //cuda.CopyDeviceToHost(outputPtr, results);
            //copy results from native mapped memory pointer to array,
            //faster then copyDtH function

            //float[] test = new float[2 * problemElements.Length];
            //Marshal.Copy(outputIntPtr, test, 0, test.Length);
            
            
            Marshal.Copy(outputIntPtr, results[0], 0, results[0].Length);
            Marshal.Copy(outputIntPtr + sizeof(float)*results[0].Length, results[1], 0, results[1].Length);

          
         }

        public void AllProductsGPU(int i, int j, CUdeviceptr QiPtr)
        {


            SetMemoryForDenseVector(i, j);


            //set the last parameter for kernel
            IVectorIdx = (uint)i;
            JVectorIdx = (uint)j;

            cuda.SetParameter(cuFunc, IdxIParamOffset, IVectorIdx);
            cuda.SetParameter(cuFunc, IdxJParamOffset, JVectorIdx);
            //this cuda kernel computes results as one long block of memory
            //first bytes stores the i-kernel kolumn elements, last stores j-kernel column elements
            //so as a pointer to result is passed QiPtr and QjPtr is computed accordingly
            //QjPtr = QiPtr + sizeof(float) * problemElements.Length;
            cuda.SetParameter(cuFunc, kernelResultParamOffset, QiPtr);
            cuda.Launch(cuFunc, blocksPerGrid, 1);
            
            
            //cuda.SynchronizeContext();

            
            
        }
        

        public virtual void SetMemoryForDenseVector(int i,int j)
        {
            
            if (IVectorIdx != i )
            {
                SparseVec vecI = problemElements[i];
                CudaHelpers.FillDenseVector(vecI, VectorI);
                cuda.CopyHostToDevice(VecIPtr, VectorI);
            }


            if (JVectorIdx != j)
            {
                SparseVec vecJ = problemElements[j];
                CudaHelpers.FillDenseVector(vecJ, VectorJ);
                cuda.CopyHostToDevice(VecJPtr, VectorJ);
            }
        }


       

        protected void InitCudaModule()
        {
            int deviceNr = 0; 
            cuda = new CUDA(deviceNr, true);
            cuCtx = cuda.CreateContext(deviceNr, CUCtxFlags.MapHost); 
            

            string modluePath = Path.Combine(Environment.CurrentDirectory, cudaModuleName);
            if (!File.Exists(modluePath))
                throw new ArgumentException("Failed access to cuda module" + modluePath);

            cuModule = cuda.LoadModule(modluePath);
            cuFunc = cuda.GetModuleFunction(cudaProductKernelName);            
        }










        /// <summary>
        /// method for setting special parameters to different kernels
        /// </summary>
        protected abstract void SetCudaFunctionParameters();



        public override void SwapIndex(int i, int j)
        {
            throw new NotSupportedException("shrinking is not supported in CUDA kernels");
            //base.SwapIndex(i, j);
        }


        public void Dispose()
        {
            if (cuda != null)
            {

                var c = cuda.PopCurrentContext();
                cuda.DestroyContext();
                var c1 = cuda.PopCurrentContext();
                cuda.DestroyContext();
            }


        }

        protected void DisposeResourses()
        {
            //free all resources
            cuda.Free(valsPtr);
            valsPtr.Pointer = IntPtr.Zero;
            cuda.Free(idxPtr);
            idxPtr.Pointer = IntPtr.Zero;
            cuda.Free(vecLengthPtr);
            vecLengthPtr.Pointer = IntPtr.Zero;


            cuda.FreeHost(outputIntPtr);
            //if (outputPtr.Pointer != IntPtr.Zero)
            //{
            //    cuda.Free(outputPtr);
            //    outputPtr.Pointer = IntPtr.Zero;
            //}

            cuda.Free(labelsPtr);
            labelsPtr.Pointer = IntPtr.Zero;
            

            if (VecIPtr.Pointer != IntPtr.Zero)
            {
                cuda.Free(VecIPtr);
                VecIPtr.Pointer = IntPtr.Zero;

            }

            if (VecJPtr.Pointer != IntPtr.Zero)
            {
                cuda.Free(VecJPtr);
                VecJPtr.Pointer = IntPtr.Zero;

            }



            if (cuVecI_TexRef.Pointer != IntPtr.Zero)
                cuda.DestroyTexture(cuVecI_TexRef);

            if (cuVecJ_TexRef.Pointer != IntPtr.Zero)
                cuda.DestroyTexture(cuVecJ_TexRef);


        }

        
    }
}
