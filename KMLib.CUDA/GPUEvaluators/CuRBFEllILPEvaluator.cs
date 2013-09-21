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

namespace KMLib.GPU.GPUEvaluators
{
    public class CuRBFEllILPEvaluator: CuEvaluator
    {
        private GASS.CUDA.Types.CUdeviceptr valsPtr;
        private GASS.CUDA.Types.CUdeviceptr idxPtr;
        private GASS.CUDA.Types.CUdeviceptr vecLengthPtr;
        private GASS.CUDA.Types.CUdeviceptr selfDotPtr;

        private int kernelResultParamOffset;
        private float gamma;
        private int vectorSelfDotParamOffset;
        private int texSelParamOffset;



        public CuRBFEllILPEvaluator(float gamma)
        {
            this.gamma = gamma;
            cudaEvaluatorKernelName = "rbfEllpackILPEvaluator";
            cudaModuleName = "KernelsEllpack.cubin";

        }
        

        protected override void SetCudaEvalFunctionParams()
        {
           
            for (int i = 0; i < NUM_STREAMS; i++)
            {

                cuda.SetFunctionBlockShape(cuFuncEval[i], evalThreads, 1, 1);

                int offset = 0;
                cuda.SetParameter(cuFuncEval[i], offset, valsPtr.Pointer);
                offset += IntPtr.Size;
                cuda.SetParameter(cuFuncEval[i], offset, idxPtr.Pointer);
                offset += IntPtr.Size;

                cuda.SetParameter(cuFuncEval[i], offset, vecLengthPtr.Pointer);
                offset += IntPtr.Size;

                cuda.SetParameter(cuFuncEval[i], offset, selfDotPtr.Pointer);
                offset += IntPtr.Size;


                cuda.SetParameter(cuFuncEval[i], offset, alphasPtr.Pointer);
                offset += IntPtr.Size;

                cuda.SetParameter(cuFuncEval[i], offset, labelsPtr.Pointer);
                offset += IntPtr.Size;

                kernelResultParamOffset = offset;
                cuda.SetParameter(cuFuncEval[i], offset, outputCuPtr[i].Pointer);
                offset += IntPtr.Size;

                cuda.SetParameter(cuFuncEval[i], offset, (uint)sizeSV);
                offset += sizeof(int);

                vectorSelfDotParamOffset = offset;
                cuda.SetParameter(cuFuncEval[i], offset, 0);
                offset += sizeof(int);

                cuda.SetParameter(cuFuncEval[i], offset, gamma);
                offset += sizeof(float);

                texSelParamOffset = offset;
                cuda.SetParameter(cuFuncEval[i], offset, i+1);
                offset += sizeof(int);

                cuda.SetParameterSize(cuFuncEval[i], (uint)offset);
            }


        }


        public override void Init()
        {
            base.Init();

             SetCudaDataForEllpack();


        }

        private void SetCudaDataForEllpack()
        {

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;

            int align = 2;
            CudaHelpers.TransformToEllpackRFormat(out vecVals, out vecColIdx, out vecLenght, TrainedModel.SupportElements, align);

            float[] selfLinDot = TrainedModel.SupportElements.Select(c => c.DotProduct()).ToArray();

            evalBlocks = (sizeSV+evalThreads) / evalThreads;
            
            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);
            
            selfDotPtr = cuda.CopyHostToDevice(selfLinDot);

            

        }

    }
}
