using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;

using System.IO;
using System.Runtime.InteropServices;
using KMLib.Kernels;
using KMLib.Helpers;

using cufy = Cudafy;
using cufyHost=Cudafy.Host;
using cufyTrans=Cudafy.Translator;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;


namespace KMLib.GPU
{

    /// <summary>
    /// Class for computing RBF kernel using cuda.
    /// Data are stored in Ellpack-R format.
    /// 
    /// </summary>
    public class CudaRBFSlicedEllpackKernel : VectorKernel<SparseVec>, IDisposable
    {

        #region cuda module constant and names

        string moduleName = "rbfSlicedEllpackKernel";
        string functionName = "rbfSlicedEllpack";

        int threadsPerRow = 4;
        int sliceSize = 64;

        #endregion

        #region cudafy fields

        /// <summary>
        /// gpu handle
        /// </summary>
        GPGPU gpu;

        private CudafyModule module;

        #endregion

        /// <summary>
        /// Array for self dot product 
        /// </summary>
        float[] selfLinDot;


        private float Gamma;

        /// <summary>
        /// linear kernel for normal product
        /// </summary>
        protected LinearKernel linKernel;


        /// <summary>
        /// vector for computing kernel product with other vectors
        /// </summary>
        /// <remarks>all the time this vector will be modified and copied to cuda array</remarks>
        protected float[] mainVector;
        

              

        public CudaRBFSlicedEllpackKernel(float gamma)
        {
            linKernel = new LinearKernel();
            Gamma = gamma;
            

        }


        public override float Product(SparseVec element1, SparseVec element2)
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

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
            //return new RbfParameterSelection();
        }

      


        public override void Init()
        {
            linKernel.ProblemElements = problemElements;
            linKernel.Y = Y;
            linKernel.Init();
           
            base.Init();

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;
            int[] sliceStart;

            CudaHelpers.TransformToSlicedEllpack(out vecVals, out vecColIdx, out sliceStart, out vecLenght, problemElements, 4, 64);
            
            selfLinDot = linKernel.DiagonalDotCache;

            #region cudafy initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            float[] valsPtr = gpu.CopyToDevice(vecVals);
            int[] idxPtr = gpu.CopyToDevice(vecColIdx);
            int[] vecLenghtPtr = gpu.CopyToDevice(vecLenght);
            int[] sliceStartPtr = gpu.CopyToDevice(sliceStart);
            //!!!!!
            float[] selfLinDotPtr = gpu.CopyToDevice(selfLinDot);

            int memSize = (problemElements.Length * sizeof(float));
            
            gpu.EnableSmartCopy();
            //allocate mapped memory for our results
            var outputIntPtr = gpu.HostAllocate<float>(memSize); // .HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            var outputPtr = gpu.GetDeviceMemoryFromIntPtr(outputIntPtr);// cuda.GetHostDevicePointer(outputIntPtr, 0);

            //normal memory allocation
            //outputPtr = cuda.Allocate((uint)(sizeof(float) * problemElements.Length));

            
            #endregion

          

            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[problemElements[0].Dim+1];
            CudaHelpers.FillDenseVector(problemElements[0],mainVector);

           // SetTextureMemory(ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector, ref mainVecPtr);
            //  SetTextureMemory(ref cuLabelsTexRef, cudaLabelsTexRefName, Y, ref labelsPtr);
           

        }

        private void InitCudaModule()
        {
           cufy.CudafyModes.Target = cufy.eGPUType.Cuda;

            gpu = CudafyHost.GetDevice(CudafyModes.Target);

            module = CudafyModule.TryDeserialize(moduleName);
            if (module == null || !module.TryVerifyChecksums())
            {
                module = CudafyTranslator.Cudafy(typeof(CudaRBFSlicedEllpackKernel));
                module.Serialize();
            }
            gpu.LoadModule(module);
        }

        


       

        #region IDisposable Members

        public void Dispose()
        {
            gpu.FreeAll();

            gpu.UnloadModules();
        }

        #endregion
    }
}
