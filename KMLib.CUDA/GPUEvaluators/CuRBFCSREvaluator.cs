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

    /// <summary>
    /// Prediction with RBF kernel in CSR format
    /// Prediction is done one by one, with use of Cuda streams and asynchronous computation of two prediction vectors
    /// </summary>
    public class CuRBFCSREvaluator: CuEvaluator
    {
        private GASS.CUDA.Types.CUdeviceptr valsPtr;
        private GASS.CUDA.Types.CUdeviceptr idxPtr;
        private GASS.CUDA.Types.CUdeviceptr vecPointerPtr;
        private GASS.CUDA.Types.CUdeviceptr selfDotPtr;

        
        private float gamma;
        private int vectorSelfDotParamOffset;



        public CuRBFCSREvaluator(float gamma)
        {
            this.gamma = gamma;
            cudaEvaluatorKernelName = "rbfCsrEvaluator";
            cudaModuleName = "KernelsEllpack.cubin";

        }

        protected override void SetCudaEvalFuncParamsForVector(SparseVec vec)
        {
            cuda.SetParameter(cuFuncEval, vectorSelfDotParamOffset, vec.DotProduct());
        }

        protected override void SetCudaEvalFunctionParams()
        {


            cuda.SetFunctionBlockShape(cuFuncEval, evalThreads, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncEval, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncEval, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncEval, offset, vecPointerPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncEval, offset, selfDotPtr.Pointer);
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

            vectorSelfDotParamOffset = offset;
            cuda.SetParameter(cuFuncEval, offset, 0);
            offset += sizeof(int);

            cuda.SetParameter(cuFuncEval, offset, gamma);
            offset += sizeof(float);

            texSelParamOffset = offset;
            cuda.SetParameter(cuFuncEval, offset, 1);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncEval, (uint)offset);
        }
         

       

        public override void Init()
        {
            base.Init();

             SetCudaData();

             SetCudaEvalFunctionParams();
        }

        private void SetCudaData()
        {

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;

            CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght, TrainedModel.SupportElements);

            

            float[] selfLinDot = TrainedModel.SupportElements.Select(c => c.DotProduct()).ToArray();

            evalBlocks = (sizeSV+evalThreads-1) / evalThreads;
            
            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecIdx);
            vecPointerPtr = cuda.CopyHostToDevice(vecLenght);
            selfDotPtr = cuda.CopyHostToDevice(selfLinDot);

        }

    }
}
