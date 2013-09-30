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
using KMLib.Evaluate;
using GASS.CUDA.Types;
using GASS.CUDA;
using System.IO;
using System.Threading.Tasks;
using KMLib.Helpers;

namespace KMLib.GPU
{

    /// <summary>
    /// base class for all cuda enabled evaluators
    /// </summary>
    /// <remarks>It sores necessary data for cuda initialization</remarks>
    public abstract class CuEvaluator : Evaluator<SparseVec>
    {

        protected const int NUM_STREAMS = 2;

        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected  string cudaModuleName = "";

        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected string[] cudaVecTexRefName = { "mainVecTexRef", "mainVec2TexRef" };

        /// <summary>
        /// cuda function name for computing prediction
        /// </summary>
        protected string cudaEvaluatorKernelName;


        protected string cudaReduceKernelName = "reduce";
        #endregion


        /// <summary>
        /// rho or "b" parameter
        /// </summary>
        protected float rho;

        /// <summary>
        /// maximum reduction blocks, def=64
        /// </summary>
        protected int maxReductionBlocks = 64;
        /// <summary>
        /// number of blocks used for reduction, grid size
        /// </summary>
        protected int reductionBlocks;
        /// <summary>
        /// number of threads used for reduction
        /// </summary>
        protected int reductionThreads;

        /// <summary>
        /// threads per block, def=128,
        /// </summary>
        protected int maxReductionThreads = 128;


        /// <summary>
        /// threads per block for evaluation function
        /// </summary>
        protected int evalThreads = CUDAConfig.XBlockSize;
        /// <summary>
        /// blocks per grid for evaluation function
        /// </summary>
        protected int evalBlocks=-1;


        /// <summary>
        /// array of 2 buffers for concurrent data transfer
        /// </summary>
        protected IntPtr[] mainVecIntPtrs = new IntPtr[NUM_STREAMS];

        /// <summary>
        /// dense support vector float buffer size
        /// </summary>
        protected uint vectorsDimMemSize=0;

        protected int vectorSelfDotParamOffset;
        protected int texSelParamOffset;
        protected int kernelResultParamOffset;  
        

        #region cuda types

        /// <summary>
        /// Cuda .net class for cuda operation
        /// </summary>
        protected CUDA cuda;


        /// <summary>
        /// cuda loaded module
        /// </summary>
        protected CUmodule cuModule;

        /// <summary>
        /// cuda kernel function for computing evaluation values
        /// </summary>
        protected CUfunction cuFuncEval;

        /// <summary>
        /// cuda kernel function for computing evaluation values
        /// </summary>
        protected CUfunction cuFuncReduce;
       

        /// <summary>
        /// cuda reference to texture for main problem element(vector), 
        /// </summary>
        protected CUtexref[] cuVecTexRef= new CUtexref[NUM_STREAMS];
        


        /// <summary>
        /// cuda mainVector pointer to device memory (it stores dense vector for prediction)
        /// </summary>
        protected CUdeviceptr[] mainVecCuPtr= new CUdeviceptr[NUM_STREAMS];


        protected CUdeviceptr[] evalOutputCuPtr = new CUdeviceptr[NUM_STREAMS];
        

        /// <summary>
        /// cuda pointer to labels, needed for coping to texture
        /// </summary>
        protected CUdeviceptr labelsPtr;

        /// <summary>
        /// cuda pointer to support vector non zero alpha coefficients
        /// </summary>
        protected CUdeviceptr alphasPtr;

        /// <summary>
        /// array of streams for async data transfer and computation
        /// </summary>
        protected CUstream[] stream = new CUstream[NUM_STREAMS];
        
        private CUdeviceptr[] reduceCuPtr = new CUdeviceptr[NUM_STREAMS];
        
        
        private IntPtr[] reduceIntPtrs=new IntPtr[NUM_STREAMS];
        
        
        /// <summary>
        /// Offset in cuda 'setparameter' function for pointer to memory to reduce
        /// </summary>
        private int offsetMemToReduce;

        /// <summary>
        /// Parameter offset for pointer to output reduce array
        /// </summary>
        private int offsetOutMemReduce;
        
        
        /// <summary>
        /// number of support vectors
        /// </summary>
        protected int sizeSV;
        
        

        #endregion

        public override float[] Predict(SparseVec[] elements)
        {

            float[] prediction = new float[elements.Length];

            uint reduceSize = (uint)reductionBlocks * sizeof(float);

            int loop = (elements.Length +NUM_STREAMS-1)/ NUM_STREAMS;
            for (int i = 0; i < loop; i++)
            {
                
                for (int s = 0; s < NUM_STREAMS; s++)
                {
                    int idx = i*NUM_STREAMS+s;
                    if (idx < elements.Length)
                    {
                        var vec = elements[idx];

                        //remove
                        //float[] svDots = TrainedModel.SupportElements.Select(sv => sv.DotProduct(vec)).ToArray();

                        //set nonzero values to dense vector accessible through vecIntPtr
                        CudaHelpers.InitBuffer(vec, mainVecIntPtrs[s]);

#region sync version
                        cuda.CopyHostToDevice(mainVecCuPtr[s], mainVecIntPtrs[s], vectorsDimMemSize);

                        cuda.SetParameter(cuFuncEval, kernelResultParamOffset, evalOutputCuPtr[s]);
                        cuda.SetParameter(cuFuncEval, vectorSelfDotParamOffset, vec.DotProduct());
                        cuda.SetParameter(cuFuncEval, texSelParamOffset, s + 1);
                        cuda.Launch(cuFuncEval, evalBlocks, 1);

                        float[] t = new float[sizeSV];
                        cuda.CopyDeviceToHost(evalOutputCuPtr[s], t);

                        cuda.SetParameter(cuFuncReduce, offsetMemToReduce, evalOutputCuPtr[s]);
                        cuda.SetParameter(cuFuncReduce, offsetOutMemReduce, reduceCuPtr[s]);
                        cuda.Launch(cuFuncReduce, reductionBlocks, 1);

                        cuda.CopyDeviceToHost(reduceCuPtr[s], reduceIntPtrs[s], reduceSize);
                        float[] r = new float[reductionBlocks];
                        cuda.CopyDeviceToHost(reduceCuPtr[s], r);
#endregion


                        //cuda.CopyHostToDeviceAsync(mainVecCuPtr[s], mainVecIntPtrs[s], vectorsDimMemSize, stream[s]);
                        ////cuFunc user different textures
                        //cuda.SetParameter(cuFuncEval, kernelResultParamOffset, evalOutputCuPtr[s]);
                        //cuda.SetParameter(cuFuncEval, vectorSelfDotParamOffset, vec.DotProduct());
                        //cuda.SetParameter(cuFuncEval, texSelParamOffset, s + 1);
                        //cuda.LaunchAsync(cuFuncEval, evalBlocks, 1, stream[s]);

                        //cuda.SetParameter(cuFuncReduce, offsetMemToReduce, evalOutputCuPtr[s]);
                        //cuda.SetParameter(cuFuncReduce, offsetOutMemReduce, reduceCuPtr[s]);
                        //cuda.LaunchAsync(cuFuncReduce, reductionBlocks, 1, stream[s]);

                        //cuda.CopyDeviceToHostAsync(reduceCuPtr[s], reduceIntPtrs[s], reduceSize, stream[s]);
                        
                    }
                }

                //wait for all streams
                cuda.SynchronizeContext();

                for (int s = 0; s < NUM_STREAMS; s++)
                {
                    int idx = i * NUM_STREAMS + s;
                    if (idx < elements.Length)
                    {
                        var vec = elements[idx];
                        //clear the buffer
                        //set nonzero values to dense vector accessible thought vecIntPtr
                        CudaHelpers.SetBufferIdx(vec, mainVecIntPtrs[s], 0.0f);
                        float evalValue = ReduceOnHost(reduceIntPtrs[s], reductionBlocks);

                        prediction[idx] = evalValue;
                    }
                }

                

            }



            return prediction;
        }

      

        private float ReduceOnHost(IntPtr reduceIntPtr, int reduceSize)
        {

            double sum = 0;
            unsafe
            {

                float* vecPtr = (float*)reduceIntPtr.ToPointer();
                for (int j = 0; j < reduceSize; j++)
                {
                    
                    sum+=vecPtr[j];
                }

            }


            //float[] t = new float[maxReductionBlocks];
            //System.Runtime.InteropServices.Marshal.Copy(reduceIntPtr, t, 0, t.Length);
            //float s = t.Sum();

            sum -= TrainedModel.Bias;

            float ret = sum < 0 ? -1 : 1;
            return ret;


        }


        public override void Init()
        {

            sizeSV = TrainedModel.SupportElements.Length;

            InitCuda();

            SetCudaData();


            SetCudaRedFunctionParams();

            

            IsInitialized = true;

        }

        abstract protected void SetCudaEvalFunctionParams();
        

        protected void SetCudaRedFunctionParams()
        {
            
            CudaHelpers.GetNumThreadsAndBlocks(sizeSV,maxReductionBlocks, maxReductionThreads, ref reductionThreads, ref reductionBlocks);

            

            cuda.SetFunctionBlockShape(cuFuncReduce, reductionThreads, 1, 1);

            int offset = 0;
            offsetMemToReduce = offset;
            cuda.SetParameter(cuFuncReduce, offset, evalOutputCuPtr[0].Pointer);
            offset += IntPtr.Size;

            offsetOutMemReduce = offset;
            cuda.SetParameter(cuFuncReduce, offset, reduceCuPtr[0].Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncReduce, offset, (uint)sizeSV);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncReduce, (uint)offset);
        }


        protected void SetCudaData()
        {

            float[] svLabels = new float[sizeSV];
            float[] svAlphas = new float[sizeSV];
            Parallel.For(0, sizeSV,
            i =>
            {
                int idx = TrainedModel.SupportElementsIndexes[i];

                svLabels[i] = TrainedModel.Y[i];
                //svLabels[i] = TrainningProblem.Labels[idx];
                svAlphas[i] = TrainedModel.Alpha[idx];

            });

            labelsPtr = cuda.CopyHostToDevice(svLabels);
            alphasPtr = cuda.CopyHostToDevice(svAlphas);


            vectorsDimMemSize = (uint)((TrainedModel.SupportElements[0].Dim+1) * sizeof(float));
            for (int i = 0; i < NUM_STREAMS; i++)
            {
                stream[i] = cuda.CreateStream();

                //allocates memory for one vector, size = vector dim
                mainVecIntPtrs[i] = cuda.AllocateHost(vectorsDimMemSize);
                mainVecCuPtr[i] = cuda.CopyHostToDevice(mainVecIntPtrs[i], vectorsDimMemSize);

                //allocate memory for output, size == #SV
                evalOutputCuPtr[i] = cuda.Allocate(svAlphas);
               
                cuVecTexRef[i] = cuda.GetModuleTexture(cuModule, cudaVecTexRefName[i]);
                //cuda.SetTextureFlags(cuVecTexRef[i], 0);
                cuda.SetTextureAddress(cuVecTexRef[i], mainVecCuPtr[i], vectorsDimMemSize);

                uint reduceMemSize = (uint)maxReductionBlocks * sizeof(float);
                reduceIntPtrs[i] = cuda.AllocateHost(reduceMemSize);
                reduceCuPtr[i] = cuda.CopyHostToDevice(reduceIntPtrs[i], reduceMemSize);
                //reduceCuPtr[i] = cuda.Allocate((uint)maxReductionBlocks * sizeof(float));

            }
            
            
        }

        private void InitCuda()
        {
            cuda = new CUDA(0, true);

            var cuCtx = cuda.CreateContext(0, CUCtxFlags.MapHost);
            cuda.SetCurrentContext(cuCtx);




            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, cudaModuleName));
            
            cuFuncEval = cuda.GetModuleFunction(cudaEvaluatorKernelName );
            

            cuFuncReduce = cuda.GetModuleFunction(cudaReduceKernelName);
        }
        
        
    }
}
