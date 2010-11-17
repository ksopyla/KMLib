using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;
using GASS.CUDA.Types;
using GASS.CUDA;
using System.IO;
using System.Threading.Tasks;

namespace KMLib.GPU
{

    /// <summary>
    /// base class for all cuda enabled evaluators
    /// </summary>
    /// <remarks>It sores nesessary data for cuda initialization</remarks>
    public abstract class CudaVectorEvaluator : EvaluatorBase<SparseVector>
    {

        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected const string cudaModuleName = "cudaSVMKernels.cubin";

        /// <summary>
        /// cuda function name for computing prediction
        /// </summary>
        protected string cudaSignKernelName = "setSignForPrediction";

        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected const string cudaMainVecTexRefName = "mainVectorTexRef";

        /// <summary>
        /// cuda function name for computing prediction
        /// </summary>
        protected string cudaEvaluatorKernelName = "linearCSREvaluatorDenseVector";
        #endregion


        /// <summary>
        /// choosed support vector 
        /// </summary>
        /// <remarks>its used for computing partial sum for prediction, 
        /// all the time this vector will be modified and copied to cuda array</remarks>
        protected float[] svVector;

        /// <summary>
        /// support elements values in CSR
        /// </summary>
        protected float[] svVals;
        
        /// <summary>
        /// support elements indexes
        /// </summary>
        protected int[] svIdx;
        
        /// <summary>
        ///support elements lenght 
        /// </summary>
        protected int[] svLenght;

        /// <summary>
        /// native pointer to output memory region
        /// </summary>
        /// <remarks>It store the result</remarks>
        protected IntPtr outputIntPtr;


        /// <summary>
        /// last parameter offset in cuda function kernel for changing <see cref="svIndex"/> 
        /// </summary>
        protected int lastParameterOffset=-1;
        /// <summary>
        /// index of current support element (vector)
        /// </summary>
        protected uint svIndex = 0;

        /// <summary>
        /// size of cuda block in X-axis
        /// </summary>
        protected int blockSizeX=CUDAConfig.XBlockSize;

        protected int blockSizeY=1;

        /// <summary>
        /// array of 2 buffers for concurent data transfer
        /// </summary>
        protected IntPtr[] svVecIntPtrs = new IntPtr[2];

        /// <summary>
        /// dense support vector float buffer size
        /// </summary>
        protected uint memSvSize=0;

        /// <summary>
        /// average vector lenght, its only a heuristic
        /// </summary>
       // protected int avgVectorLenght = 50;
     
        /// <summary>
        /// how many will be blocks for grid
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
        /// cuda kernel function for computing evaluation values
        /// </summary>
        protected CUfunction cuFunc;

        /// <summary>
        /// cuda kernel function for computing final prediction
        /// </summary>
        protected CUfunction cuFuncSign;


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
        protected CUtexref cuSVTexRef;


        /// <summary>
        /// cuda mainVector pointer to device memory (it stores dense support vector)
        /// </summary>
        protected CUdeviceptr mainVecPtr;

        /// <summary>
        /// cuda refeerenc to texture for labels
        /// </summary>
        protected CUtexref cuLabelsTexRef;

        /// <summary>
        /// cuda pointer to labels, neded for coping to texture
        /// </summary>
        protected CUdeviceptr labelsPtr;

        /// <summary>
        /// cuda pointer to support vector non zero alpha coeficients
        /// </summary>
        protected CUdeviceptr alphasPtr;

        /// <summary>
        /// stream for async data transfer and computation
        /// </summary>
        protected CUstream stream;
        
        

        #endregion

        public override void Init()
        {
            

            cuda = new CUDA(0, true);
            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, cudaModuleName));
            cuFunc = cuda.GetModuleFunction(cudaEvaluatorKernelName);

            cuFuncSign = cuda.GetModuleFunction(cudaSignKernelName);

            svVector = new float[TrainningProblem.Elements[0].Count];

             stream = cuda.CreateStream();
            memSvSize = (uint)(TrainningProblem.Elements[0].Count * sizeof(float));
            

            //allocates memory for buffers
            svVecIntPtrs[0] = cuda.AllocateHost(memSvSize);
            svVecIntPtrs[1] = cuda.AllocateHost(memSvSize);
            mainVecPtr = cuda.CopyHostToDeviceAsync(svVecIntPtrs[0], memSvSize, stream);

            cuSVTexRef = cuda.GetModuleTexture(cuModule, "svTexRef");
            cuda.SetTextureFlags(cuSVTexRef, 0);
            cuda.SetTextureAddress(cuSVTexRef, mainVecPtr, memSvSize);

            //todo: copy labels and alphas

            float[] svLabels = new float[TrainedModel.SupportElements.Length];
            float[] svAlphas = new float[TrainedModel.SupportElements.Length];


            Parallel.For(0,TrainedModel.SupportElementsIndexes.Length, 
            i=>{
                int idx = TrainedModel.SupportElementsIndexes[i];
                svLabels[i] = TrainningProblem.Labels[idx];
                svAlphas[i] = TrainedModel.Alpha[idx];
                    
            });
            
            //for (int i = 0; i < TrainedModel.SupportElementsIndexes.Length; i++)
            //{
            //    int idx = TrainedModel.SupportElementsIndexes[i];
            //    svLabels[i]= TrainningProblem.Labels[idx];
            //    svAlphas[i] = TrainedModel.Alpha[idx];

            //}

            labelsPtr = cuda.CopyHostToDevice(svLabels);
            alphasPtr = cuda.CopyHostToDevice(svAlphas);

            IsInitialized = true;

        }
        
        
    }
}
