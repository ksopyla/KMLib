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

namespace KMLib.GPU
{

    /// <summary>
    /// Based clas for all cuda enable svm kernel
    /// This class encapsulates all necessary variables for cuda.net library
    /// </summary>
    public abstract class CUDAVectorKernel : VectorKernel<SparseVec>
    {

        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected string cudaModuleName = "cudaSVMKernels.cubin";


        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected string cudaMainVecTexRefName = "mainVectorTexRef";
        /// <summary>
        /// cuda texture name for labels
        /// </summary>
        protected string cudaLabelsTexRefName = "labelsTexRef";

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
        protected float[] mainVector;

        /// <summary>
        /// native pointer to output memory region
        /// </summary>
        protected IntPtr outputIntPtr;


        /// <summary>
        /// parameter offset in cuda function kernel for changing mainVectorIdx
        /// </summary>
        protected int mainVecIdxParamOffset=-1;

        /// <summary>
        /// parameter offset in cuda kernel for result 
        /// </summary>
        protected int kernelResultParamOffset=-1;
        /// <summary>
        /// index of current main problem element (vector)
        /// </summary>
        protected uint mainVectorIdx = 0;


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
        protected CUDA cuda;


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
        protected CUdeviceptr vecLenghtPtr;

        /// <summary>
        /// cuda device pointer for output
        /// </summary>
        public CUdeviceptr outputPtr;

        /// <summary>
        /// cuda reference to texture for main problem element(vector), 
        /// </summary>
        protected CUtexref cuMainVecTexRef;

        protected CUdeviceptr mainVecPtr;

        /// <summary>
        /// cuda refeerenc to texture for labels
        /// </summary>
        protected CUtexref cuLabelsTexRef;

        /// <summary>
        /// cuda pointer to labels, neded for coping to texture
        /// </summary>
        protected CUdeviceptr labelsPtr;


        #endregion


        public override SparseVec[] ProblemElements
        {
            set
            {
                if (value == null) throw new ArgumentNullException("value");
                linKernel.ProblemElements = value;

                base.ProblemElements = value;

                blocksPerGrid = (value.Length + threadsPerBlock - 1) / threadsPerBlock;
                //blocksPerGrid = (2*value.Length + threadsPerBlock - 1) / threadsPerBlock;
            }
        }

        public override void AllProducts(int element1, float[] results)
        {

            //cuda calculation
            //todo: possible small improvements
            //if mainVectorIdx==element1 then we don't have to copy to device
            SparseVec mainVec = problemElements[element1];

            if (mainVectorIdx != element1)
            {
                CudaHelpers.FillDenseVector(mainVec, mainVector);

                cuda.CopyHostToDevice(mainVecPtr, mainVector);

            }
            //uint align = cuda.SetTextureAddress(cuMainVecTexRef, mainVecPtr, (uint)(sizeof(float) * mainVector.Length));

            //copy to texture
            // cuda.CopyHostToArray(cuMainVecArray, mainVector, 0);


            //set the last parameter for kernel
            mainVectorIdx = (uint)element1;
            cuda.SetParameter(cuFunc, mainVecIdxParamOffset, mainVectorIdx);
            
            /*
            CUevent start = cuda.CreateEvent();
            CUevent end = cuda.CreateEvent();
            cuda.RecordEvent(start);
            var st = Stopwatch.StartNew();
            */
            cuda.Launch(cuFunc, blocksPerGrid, 1);

            //cuda.RecordEvent(end);
            cuda.SynchronizeContext();

            //st.Stop();
            //var elapsed2 = st.ElapsedMilliseconds;
            //var elapsed = cuda.ElapsedTime(start, end);


            //copy resulsts form device to host
            //cuda.CopyDeviceToHost(outputPtr, results);
            //copy results from native mapped memory pointer to array,
            //faster then copyDtH function
            Marshal.Copy(outputIntPtr, results, 0, results.Length);


        }

        public void AllProductsGPU(int element1,CUdeviceptr devResultPtr)
        {
            SparseVec mainVec = problemElements[element1];

            if (mainVectorIdx != element1)
            {
                CudaHelpers.FillDenseVector(mainVec, mainVector);

                cuda.CopyHostToDevice(mainVecPtr, mainVector);

            }
            //set the last parameter for kernel
            mainVectorIdx = (uint)element1;
            cuda.SetParameter(cuFunc, mainVecIdxParamOffset, mainVectorIdx);

            cuda.Launch(cuFunc, blocksPerGrid, 1);
            
            //cuda.SynchronizeContext();
        }

        protected void InitCudaModule()
        {
            cuda = new CUDA(0, true);
            var cuCtx = cuda.CreateContext(0, CUCtxFlags.MapHost);
            cuda.SetCurrentContext(cuCtx);

            string modluePath = Path.Combine(Environment.CurrentDirectory, cudaModuleName);
            if (!File.Exists(modluePath))
                throw new ArgumentException("Failed access to cuda module" + modluePath);

            cuModule = cuda.LoadModule(modluePath);
            cuFunc = cuda.GetModuleFunction(cudaProductKernelName);
            //var cuModule2 = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "rbfSlicedEllpackKernel.cubin"));
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

    }
}
