using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Runtime.InteropServices;
using KMLib.Kernels;

namespace KMLib.GPU
{

    /// <summary>
    /// Class for computing RBF kernel using cuda.
    /// 
    /// </summary>
    public class CudaRBFKernel : CUDAVectorKernel, IDisposable
    {
       
        //const string cudaKernelName = "rbfCsrFormatKernel";
       
        const string cudaSelfDotTexRefName = "selfDotTexRef";

        /// <summary>
        /// Array for self dot product 
        /// </summary>
        float[] selfLinDot;


        private float Gamma;
       
        

       /// <summary>
       /// cuda device pointer for stroing self linear dot product
       /// </summary>
        private CUdeviceptr selfLinDotPtr;


       

        public CudaRBFKernel(float gamma)
        {
            linKernel = new LinearKernel();
            Gamma = gamma;
            cudaProductKernelName = "rbfCsrFormatKernel";
        }


        public override float Product(SparseVector element1, SparseVector element2)
        {

            float x1Squere = linKernel.Product(element1, element1);
            float x2Squere = linKernel.Product(element2, element2);

            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));

            return prod;

        }

        public override float Product(int element1, int element2)
        {
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");


            float x1Squere = 0f, x2Squere = 0f, dot = 0f, prod = 0f;

            if (element1 == element2)
            {
                if (DiagonalDotCacheBuilded)
                    return DiagonalDotCache[element1];
                else
                {
                    //all parts are the same
                    // x1Squere = x2Squere = dot = linKernel.Product(element1, element1);
                    //prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));
                    // (x1Squere + x2Squere - 2 * dot)==0 this expresion is equal zero
                    //so we can prod set to 1 beceause exp(0)==1
                    prod = 1f;
                }
            }
            else
            {
                //when element1 and element2 are different we have to compute all parts
                x1Squere = linKernel.Product(element1, element1);
                x2Squere = linKernel.Product(element2, element2);
                dot = linKernel.Product(element1, element2);
                prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));
            }
            return prod;
        }

        public override ParameterSelection<SparseVector> CreateParameterSelection()
        {
            throw new NotImplementedException();
            //return new RbfParameterSelection();
        }

      


        public override void Init()
        {
            linKernel.Init();
           
            base.Init();

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;
           CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght,problemElements);


            selfLinDot = linKernel.DiagonalDotCache;

            #region cuda initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            //!!!!!
            selfLinDotPtr = cuda.CopyHostToDevice(selfLinDot);

            uint memSize = (uint)(problemElements.Length * sizeof(float));
            //allocate mapped memory for our results
            outputIntPtr = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr = cuda.GetHostDevicePointer(outputIntPtr, 0);

            //normal memory allocation
            //outputPtr = cuda.Allocate((uint)(sizeof(float) * problemElements.Length));

            
            #endregion

            SetCudaFunctionParameters();

            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[problemElements[0].Count];
            CudaHelpers.FillDenseVector(problemElements[0],mainVector);

            SetTextureMemory(ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector, ref mainVecPtr);

            SetTextureMemory(ref cuLabelsTexRef, cudaLabelsTexRefName, Labels, ref labelsPtr);
           

        }

        

        protected override void SetCudaFunctionParameters()
        {
            
            #region cuda set function parameters
            cuda.SetFunctionBlockShape(cuFunc, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFunc, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFunc, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, vecLenghtPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, selfLinDotPtr.Pointer);
            offset += IntPtr.Size;


            cuda.SetParameter(cuFunc, offset, outputPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)problemElements.Length);
            offset += sizeof(int);

            lastParameterOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)mainVectorIdx);
            offset += sizeof(int);

            cuda.SetParameter(cuFunc, offset, Gamma);
            offset += sizeof(float);

            cuda.SetParameterSize(cuFunc, (uint)offset);


            #endregion
        }

       

        #region IDisposable Members

        public void Dispose()
        {
            if (cuda != null)
            {
                //free all resources
                cuda.Free(valsPtr);
                cuda.Free(idxPtr);
                cuda.Free(vecLenghtPtr);

                cuda.Free(outputPtr);
                cuda.Free(labelsPtr);
                cuda.DestroyTexture(cuLabelsTexRef);

                cuda.Free(mainVecPtr);

                cuda.DestroyTexture(cuMainVecTexRef);

                cuda.Dispose();
                cuda = null;
            }
        }

        #endregion
    }
}
