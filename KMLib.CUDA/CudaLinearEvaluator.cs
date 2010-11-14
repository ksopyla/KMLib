using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;
using GASS.CUDA;
using GASS.CUDA.Types;

namespace KMLib.GPU
{

    /// <summary>
    /// Represents evaluation class which use CUDA, 
    /// </summary>
    public class CudaLinearEvaluator : CudaVectorEvaluator
    {




        /// <summary>
        /// Predicts the specified elements.
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <returns></returns>
        public override float[] Predict(SparseVector[] elements)
        {
            throw new NotImplementedException();


            // elements values
            float[] vecVals;
            //elements indexes
            int[] vecIdx;
            //elements lenght
            int[] vecLenght;
            CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght, elements);

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            uint memElementsSize = (uint)(elements.Length * sizeof(float));
            //allocate mapped memory for our results
            outputIntPtr = cuda.HostAllocate(memElementsSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr = cuda.GetHostDevicePointer(outputIntPtr, 0);


            //todo: Set the cuda kernel paramerters


            for (int k = 0; k < TrainedModel.SupportElements.Length; k++)
            {

                //set the buffer values from k-th support vector
                CudaHelpers.InitBuffer(TrainedModel.SupportElements[k], svVecIntPtrs[k % 2]);

                cuda.SynchronizeStream(stream);
                //copy asynchronously from buffer to devece
                cuda.CopyHostToDeviceAsync(mainVecPtr, svVecIntPtrs[k % 2], memSvSize, stream);
                //set the last parameter in kernel (column index)    
                cuda.SetParameter(cuFunc, lastParameterOffset, (uint)k);
                //launch kernl    
                cuda.LaunchAsync(cuFunc, gridDimX, 1, stream);


                if (k > 0)
                {
                    //clear the previous host buffer
                    CudaHelpers.SetBufferIdx(TrainedModel.SupportElements[k - 1], svVecIntPtrs[(k + 1) % 2], 0.0f);
                }

            }

        }


    }
}
