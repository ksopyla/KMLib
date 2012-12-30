using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Runtime.InteropServices;
using KMLib.Kernels;
using KMLib.Helpers;

namespace KMLib.GPU
{

    /// <summary>
    /// Represents Chi^2 Kernel for computing product between two histograms
    /// 
    /// K(x,y)= Sum( (xi*yi)/(xi+yi))
    /// 
    /// vectors should contains positive numbers(like histograms does) and should be normalized
    /// sum(xi)=1
    /// Data are stored in Ellpack-R format.
    /// 
    /// </summary>
    public class CuChiSquaredNormEllpackKernel : CuVectorKernel, IDisposable
    {

        ChiSquaredNormKernel chiSquared;


        public CuChiSquaredNormEllpackKernel()
        {
            linKernel = new LinearKernel();
            chiSquared = new ChiSquaredNormKernel();
            
            cudaProductKernelName = "chiSquaredNormEllpackKernel";
            
        }


        public override float Product(SparseVec element1, SparseVec element2)
        {
            return chiSquared.Product(element1, element2);
        }

        public override float Product(int element1, int element2)
        {
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");


            return chiSquared.Product(element1, element2);
        }

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
        }




        public override void Init()
        {
            
            chiSquared.ProblemElements = problemElements;
            chiSquared.Y = Y;
            chiSquared.Init();

            base.Init();

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;

            CudaHelpers.TransformToEllpackRFormat(out vecVals, out vecColIdx, out vecLenght, problemElements);

            #region cuda initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecColIdx);
            vecLengthPtr = cuda.CopyHostToDevice(vecLenght);

            uint memSize = (uint)(problemElements.Length * sizeof(float));
            //allocate mapped memory for our results
            outputIntPtr = cuda.HostAllocate(memSize,CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr = cuda.GetHostDevicePointer(outputIntPtr, 0);

            //normal memory allocation
            //outputPtr = cuda.Allocate((uint)(sizeof(float) * problemElements.Length));


            #endregion

            SetCudaFunctionParameters();

            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[problemElements[0].Dim + 1];
            CudaHelpers.FillDenseVector(problemElements[0], mainVector);

            CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector, ref mainVecPtr);

            CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuLabelsTexRef, cudaLabelsTexRefName, Y, ref labelsPtr);


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

            cuda.SetParameter(cuFunc, offset, vecLengthPtr.Pointer);
            offset += IntPtr.Size;
            
            kernelResultParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, outputPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)problemElements.Length);
            offset += sizeof(int);

            mainVecIdxParamOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)mainVectorIdx);
            offset += sizeof(int);

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
                valsPtr.Pointer = IntPtr.Zero;
                cuda.Free(idxPtr);
                idxPtr.Pointer = IntPtr.Zero;
                cuda.Free(vecLengthPtr);
                vecLengthPtr.Pointer = IntPtr.Zero;

                cuda.FreeHost(outputIntPtr);
                //cuda.Free(outputPtr);
                outputPtr.Pointer = IntPtr.Zero;
                cuda.Free(labelsPtr);
                labelsPtr.Pointer = IntPtr.Zero;
                cuda.DestroyTexture(cuLabelsTexRef);

                cuda.Free(mainVecPtr);
                mainVecPtr.Pointer = IntPtr.Zero;

                cuda.DestroyTexture(cuMainVecTexRef);

                cuda.UnloadModule(cuModule);
                cuda.Dispose();
                cuda = null;
            }
        }

        #endregion

        public override string ToString()
        {
            return "Cuda Chi-Squared Norm Kernel";
        }
    }

}
