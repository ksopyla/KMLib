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

    /// <summary>
    /// evaluator for n chi2 kernel with Ellpack format kernel
    /// </summary>
    public class CuNChi2EllpackEvaluator: CuEvaluator
    {

        private GASS.CUDA.Types.CUdeviceptr valsPtr;
        private GASS.CUDA.Types.CUdeviceptr idxPtr;
        private GASS.CUDA.Types.CUdeviceptr vecLengthPtr;

        public CuNChi2EllpackEvaluator(float gamma)
        {
            cudaEvaluatorKernelName = "nChi2EllpackEvaluator";
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

             SetCudaDataForEllpack();

             SetCudaEvalFunctionParams();
        }

        private void SetCudaDataForEllpack()
        {

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;


            CudaHelpers.TransformToEllpackRFormat(out vecVals, out vecColIdx, out vecLenght, TrainedModel.SupportElements);

            evalBlocks = (sizeSV+evalThreads-1) / evalThreads;
            
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
