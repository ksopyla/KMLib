using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;
using dnAnalytics.LinearAlgebra;
using GASS.CUDA.Types;
using GASS.CUDA;
using System.Runtime.InteropServices;
using System.IO;

namespace KMLib.GPU
{

    /// <summary>
    /// Based clas for all cuda enable svm kernel
    /// </summary>
    public abstract class CUDAVectorKernel: VectorKernel<SparseVector>
    {

        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected const string cudaModuleName = "cudaSVMKernels.cubin";
        

        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected const string cudaMainVecTexRefName = "mainVectorTexRef";
        /// <summary>
        /// cuda texture name for labels
        /// </summary>
        protected const string cudaLabelsTexRefName = "labelsTexRef";

        /// <summary>
        /// cuda function name for computing product
        /// </summary>
        protected   string cudaProductKernelName;
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
        /// last parameter offset in cuda function kernel for changing mainVectorIdx
        /// </summary>
        protected int lastParameterOffset;
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
        protected CUdeviceptr outputPtr;

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


        public override SparseVector[] ProblemElements
        {
            set
            {
                if (value == null) throw new ArgumentNullException("value");
                linKernel.ProblemElements = value;

                base.ProblemElements = value;

                blocksPerGrid = (value.Length + threadsPerBlock - 1) / threadsPerBlock;
            }
        }

        public override void AllProducts(int element1, float[] results)
        {

            //cuda calculation
            //todo: possible small improvements
            //if mainVectorIdx==element1 then we don't have to copy to device
            SparseVector mainVec = problemElements[element1];

            if (mainVectorIdx != element1)
            {
                CudaHelpers.FillDenseVector(mainVec,mainVector);

                cuda.CopyHostToDevice(mainVecPtr, mainVector);

            }
            //uint align = cuda.SetTextureAddress(cuMainVecTexRef, mainVecPtr, (uint)(sizeof(float) * mainVector.Length));

            //copy to texture
            // cuda.CopyHostToArray(cuMainVecArray, mainVector, 0);


            //set the last parameter for kernel
            mainVectorIdx = (uint)element1;
            cuda.SetParameter(cuFunc, lastParameterOffset, mainVectorIdx);

            cuda.Launch(cuFunc, blocksPerGrid, 1);

            cuda.SynchronizeContext();
            //copy resulsts form device to host
            //cuda.CopyDeviceToHost(outputPtr, results);
            //copy results from native mapped memory pointer to array,
            //faster then copyDtH function
            Marshal.Copy(outputIntPtr, results, 0, results.Length);


        }

        protected void InitCudaModule()
        {
            cuda = new CUDA(0, true);
            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, cudaModuleName));
            cuFunc = cuda.GetModuleFunction(cudaProductKernelName);
        }



        


        protected void SetTextureMemory(ref CUtexref texture, string texName, float[] data, ref CUdeviceptr memPtr)
        {
            texture = cuda.GetModuleTexture(cuModule, texName);
            memPtr = cuda.CopyHostToDevice(data);
            cuda.SetTextureAddress(texture, memPtr, (uint)(sizeof(float) * data.Length));

        }
       


        /// <summary>
        /// method for setting special parameters to different kernels
        /// </summary>
        protected abstract void SetCudaFunctionParameters();
        
    }
}
