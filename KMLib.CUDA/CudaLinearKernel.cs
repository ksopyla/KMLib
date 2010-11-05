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
    /// class for computing linear kernel using cuda
    /// </summary>
    public class CudaLinearKernel : CUDAVectorKernel, IDisposable
    {
        
        //const string cudaKernelName = "linearCsrFormatKernel";
       

        public CudaLinearKernel()
        {
            linKernel = new LinearKernel();
            cudaProductKernelName = "linearCsrFormatKernel";
        }

        public override float Product(SparseVector element1, SparseVector element2)
        {
            return linKernel.Product(element1, element2);
        }

        public override float Product(int element1, int element2)
        {
            return linKernel.Product(element1, element2);
        }

        public override ParameterSelection<SparseVector> CreateParameterSelection()
        {
            return linKernel.CreateParameterSelection();
        }


        public override void Init()
        {
            base.Init();

            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;
            TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght);

            #region cuda initialization

            InitCudaModule();
           
            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            uint memSize = (uint)(problemElements.Length * sizeof(float));
            //allocate mapped memory for our results
            outputIntPtr = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr = cuda.GetHostDevicePointer(outputIntPtr, 0);

            

            #endregion
            SetCudaFunctionParameters();

            
            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[problemElements[0].Count];
            CopyMainVectorVals(problemElements[0]);

            //get reference to cuda texture for main vector
            //cuMainVecTexRef = cuda.GetModuleTexture(cuModule, cudaMainVecTexRefName);
            //mainVecPtr = cuda.CopyHostToDevice(mainVector);
            //cuda.SetTextureAddress(cuMainVecTexRef, mainVecPtr, (uint)(sizeof(float) * mainVector.Length));

            SetTextureMemory(ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector,ref mainVecPtr);

            //cuLabelsTexRef = cuda.GetModuleTexture(cuModule, cudaLabelsTexRefName);
            //labelsPtr = cuda.CopyHostToDevice(Labels);
            //uint align = cuda.SetTextureAddress(cuLabelsTexRef, labelsPtr, (uint)(sizeof(float) * Labels.Length));

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

            cuda.SetParameter(cuFunc, offset, outputPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFunc, offset, (uint)problemElements.Length);
            offset += sizeof(int);

            lastParameterOffset = offset;
            cuda.SetParameter(cuFunc, offset, (uint)mainVectorIdx);



            offset += sizeof(int);
            cuda.SetParameterSize(cuFunc, (uint)offset);

            // cuda.UseRuntimeExceptions = false;
            // cuda.SetFunctionSharedSize(cuFunc, (uint)(sizeof(float) * problemElements[0].Count));

            //var cuerr = CUDARuntime.cudaGetLastError();
            //string errMsg = CUDARuntime.cudaGetErrorString(cuerr);
            //var err= cuda.LastError;
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
                
                //cuda.Free(outputPtr);
                //cuda.FreeHost(outputIntPtr);

                

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
