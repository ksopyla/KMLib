using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;
using GASS.CUDA;
using System.Diagnostics;
using GASS.CUDA.Types;
using System.Runtime.InteropServices;
using KMLib.Helpers;

namespace KMLib.GPU
{

    /// <summary>
    /// Represents RBF evaluator, its used for prediction an unseen elements. For prediction use CUDA, data is in CSR fromat.
    /// </summary>
    public class CudaRBFCSREvaluator : CudaVectorEvaluator, IDisposable
    {


        CUdeviceptr elSelf;
        CUdeviceptr svSelf;

        float gamma;
        public CudaRBFCSREvaluator(float gamma)
        {
            this.gamma = gamma;
            cudaEvaluatorKernelName = "rbfCSREvaluatorDenseVector";
        }

        /// <summary>
        /// Predicts the specified elements.
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <returns>array of predicted labels +1 or -1</returns>
        public override float[] Predict(SparseVec[] elements)
        {

            if (!IsInitialized)
                throw new ApplicationException("Evaluator is not initialized. Call init method");


            //tranfsorm elements to matrix in CSR format
            // elements values
            float[] vecVals;
            //elements indexes
            int[] vecIdx;
            //elements lenght
            int[] vecLenght;
            CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght, elements);


            float[] elSelfDot = new float[elements.Length];

            Stopwatch t = Stopwatch.StartNew();
            for (int j = 0; j < elements.Length; j++)
            {
                float res = 0;
                for (int k = vecLenght[j]; k < vecLenght[j + 1]; k++)
                    res += vecVals[k] * vecVals[k];

                elSelfDot[j] = res;
            }
            t.Stop();

            float[] svSelfDot = new float[TrainedModel.SupportElements.Length];
            for (int i = 0; i < TrainedModel.SupportElements.Length; i++)
            {

                svSelfDot[i] = (float)TrainedModel.SupportElements[i].DotProduct();
            }

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            //we don't need this any more
            vecIdx = null;
            vecVals = null;
            vecLenght = null;

            elSelf = cuda.CopyHostToDevice(elSelfDot);
            svSelf = cuda.CopyHostToDevice(svSelfDot);

            elSelfDot = null;
            svSelfDot = null;

            uint memElementsSize = (uint)(elements.Length * sizeof(float));
            //allocate mapped memory for our results
            outputIntPtr = cuda.HostAllocate(memElementsSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr = cuda.GetHostDevicePointer(outputIntPtr, 0);

            // Set the cuda kernel paramerters
            #region set cuda parameters
            uint Rows = (uint)elements.Length;
            uint Cols = (uint)TrainedModel.SupportElements.Length;


            cuda.SetFunctionBlockShape(cuFunc, blockSizeX, blockSizeY, 1);

            int offset = 0;
            //set elements param
            cuda.SetParameter(cuFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, vecLenghtPtr.Pointer);
            offset += IntPtr.Size;

            //set labels param
            cuda.SetParameter(cuFunc, offset, labelsPtr.Pointer);
            offset += IntPtr.Size;
            //set alphas param
            cuda.SetParameter(cuFunc, offset, alphasPtr.Pointer);
            offset += IntPtr.Size;

            //self dot 
            cuda.SetParameter(cuFunc, offset, svSelf.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, elSelf.Pointer);
            offset += IntPtr.Size;
            //set output (reslut) param
            cuda.SetParameter(cuFunc, offset, outputPtr.Pointer);
            offset += IntPtr.Size;
            //set number of elements param
            cuda.SetParameter(cuFunc, offset, (uint)Rows);
            offset += sizeof(int);
            //set number of support vectors param
            cuda.SetParameter(cuFunc, offset, (uint)Cols);
            offset += sizeof(int);

            //set gamma parameter
            cuda.SetParameter(cuFunc, offset, gamma);
            offset += sizeof(float);

            //set support vector index param
            lastParameterOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)0);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFunc, (uint)offset);
            #endregion

            int gridDimX = (int)Math.Ceiling((Rows + 0.0) / (blockSizeX));


            for (int k = 0; k < TrainedModel.SupportElements.Length; k++)
            {
                //set the buffer values from k-th support vector
                CudaHelpers.InitBuffer(TrainedModel.SupportElements[k], svVecIntPtrs[k % 2]);

                cuda.SynchronizeStream(stream);
                //copy asynchronously from buffer to devece
                cuda.CopyHostToDeviceAsync(mainVecPtr, svVecIntPtrs[k % 2], memSvSize, stream);
                //set the last parameter in kernel (column index)   
                // colIndexParamOffset
                cuda.SetParameter(cuFunc, lastParameterOffset, (uint)k);
                //launch kernl    
                cuda.LaunchAsync(cuFunc, gridDimX, 1, stream);

                if (k > 0)
                {
                    //clear the previous host buffer
                    CudaHelpers.SetBufferIdx(TrainedModel.SupportElements[k - 1], svVecIntPtrs[(k + 1) % 2], 0.0f);
                }

            }

            //CUdeviceptr symbolAdr;
            //CUDARuntime.cudaGetSymbolAddress(ref symbolAdr,"RHO");
            rho = TrainedModel.Bias;
            //IntPtr symbolVal = new IntPtr(&rho);
            //CUDARuntime.cudaMemcpyToSymbol("RHO", symbolVal, 1, 1, cudaMemcpyKind.cudaMemcpyHostToDevice);

            //set label sign on cuda device
            cuda.SetFunctionBlockShape(cuFuncSign, blockSizeX, blockSizeY, 1);
            int signFuncOffset = 0;
            //set array param
            cuda.SetParameter(cuFuncSign, signFuncOffset, outputPtr.Pointer);
            signFuncOffset += IntPtr.Size;
            //set size 
            cuda.SetParameter(cuFuncSign, signFuncOffset, Rows);
            signFuncOffset += sizeof(int);

            cuda.SetParameter(cuFuncSign, signFuncOffset, rho);
            signFuncOffset += sizeof(float);

            cuda.SetParameterSize(cuFuncSign, (uint)signFuncOffset);

            //gridDimX is valid for this function
            cuda.LaunchAsync(cuFuncSign, gridDimX, 1, stream);

            //wait for all computation
            cuda.SynchronizeContext();


            float[] result = new float[elements.Length];
            //copy result
            Marshal.Copy(outputIntPtr, result, 0, elements.Length);

            return result;

        }


        public void Dispose()
        {
            if (cuda != null)
            {

                //free all resources
                cuda.Free(valsPtr);
                valsPtr.Pointer =IntPtr.Zero;

                cuda.Free(idxPtr);
                idxPtr.Pointer =IntPtr.Zero;

                cuda.Free(vecLenghtPtr);
                vecLenghtPtr.Pointer =IntPtr.Zero;

                cuda.Free(svSelf);
                svSelf.Pointer =IntPtr.Zero;

                cuda.Free(elSelf);
                elSelf.Pointer =IntPtr.Zero;


                cuda.FreeHost(outputIntPtr);

                //cuda.Free(outputPtr);
                outputPtr.Pointer =IntPtr.Zero;

                cuda.FreeHost(svVecIntPtrs[0]);
                cuda.FreeHost(svVecIntPtrs[1]);

                cuda.Free(mainVecPtr);
                mainVecPtr.Pointer =IntPtr.Zero;
                cuda.DestroyTexture(cuSVTexRef);

                cuda.Free(labelsPtr);
                labelsPtr.Pointer =IntPtr.Zero;
                if (cuLabelsTexRef.Pointer.ToInt32() != 0)
                    cuda.DestroyTexture(cuLabelsTexRef);


                cuda.Free(alphasPtr);

                cuda.DestroyStream(stream);

                //Marshal.FreeHGlobal(svVecIntPtrs[0]);
                //Marshal.FreeHGlobal(svVecIntPtrs[1]);

                cuda.UnloadModule(cuModule);

                cuda.Dispose();
                cuda = null;

                TrainedModel.Y = null;
                TrainedModel.Alpha = null;
                TrainedModel.SupportElements = null;
                TrainedModel.SupportElementsIndexes = null;
                TrainedModel = null;




                IsInitialized = false;
            }
        }
    }
}
