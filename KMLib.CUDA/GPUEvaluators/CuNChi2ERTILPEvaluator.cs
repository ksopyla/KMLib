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
using KMLib.Helpers;

namespace KMLib.GPU.GPUEvaluators
{
    public class CuNChi2ERTILPEvaluator : CuEvaluator
    {
        private GASS.CUDA.Types.CUdeviceptr valsPtr;
        private GASS.CUDA.Types.CUdeviceptr idxPtr;
        private GASS.CUDA.Types.CUdeviceptr vecLengthPtr;

        /// <summary>
        /// how many threads are assigned for row
        /// </summary>
        private int ThreadsPerRow=4;

        /// <summary>
        /// how many non zero elements are loaded in cuda kernel
        /// </summary>
        private int Prefetch=2;
        
        public CuNChi2ERTILPEvaluator(float gamma)
        {
            cudaEvaluatorKernelName = "nChi2ERTILPEvaluator";
            cudaModuleName = "KernelsEllpack.cubin";

        }

        protected override void SetCudaEvalFunctionParams()
        {
            cuda.SetFunctionBlockShape(cuFuncEval, evalThreads, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncEval, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncEval, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncEval, offset, vecLengthPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncEval, offset, alphasPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncEval, offset, labelsPtr.Pointer);
            offset += IntPtr.Size;

            kernelResultParamOffset = offset;
            cuda.SetParameter(cuFuncEval, offset, evalOutputCuPtr[0].Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncEval, offset, (uint)sizeSV);
            offset += sizeof(int);

            texSelParamOffset = offset;
            cuda.SetParameter(cuFuncEval, offset, 1);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncEval, (uint)offset);
        }

       

        public override void Init()
        {
            base.Init();

             SetCudaDataForERTILP();

             SetCudaEvalFunctionParams();
        }

        private void SetCudaDataForERTILP()
        {

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;

            

            evalBlocks = (int)Math.Ceiling((ThreadsPerRow * sizeSV + 0.0) / evalThreads);
            int align = ThreadsPerRow * Prefetch;
            CudaHelpers.TransformToERTILPFormat(out vecVals, out vecColIdx, out vecLenght, TrainedModel.SupportElements, align, ThreadsPerRow);
           
            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);

        }

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

    }
}
