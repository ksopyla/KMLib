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
using cufyHost = Cudafy.Host;
using cufyTrans = Cudafy.Translator;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using GASS.CUDA;
using GASS.CUDA.Types;


namespace KMLib.GPU
{

    /// <summary>
    /// Class for computing RBF kernel using cuda.
    /// Data are stored in Ellpack-R format.
    /// </summary>
    /// <remarks>
    /// this implementation use cudafy
    /// </remarks>
    public class CuRBFSlicedEllpackKernel :  VectorKernel<SparseVec>, IDisposable
    {

        #region cuda module constant and names

        string moduleName = "rbfSlicedEllpackKernel";
                             

        #endregion

        #region cudafy fields

        /// <summary>
        /// gpu handle
        /// </summary>
        GPGPU gpu;

        private CudafyModule module;


        /// <summary>
        /// Cuda.net handle
        /// </summary>
        CUDA cuGPU;

        CUtexref cuMainVecTexRef;

        CUdeviceptr mainVectorPtr;

        #endregion

        /// <summary>
        /// Array for self dot product 
        /// </summary>
        float[] selfLinDot;



        /// <summary>
        /// linear kernel for normal product
        /// </summary>
        protected LinearKernel linKernel;


        /// <summary>
        /// vector for computing kernel product with other vectors
        /// </summary>
        /// <remarks>all the time this vector will be modified and copied to cuda array</remarks>
        protected float[] mainVector;

        
        int mainVecIdx;

        #region Cudafy variables
        
        /// <summary>
        /// pointer to vector values array
        /// </summary>
        float[] valsPtr;
        /// <summary>
        /// pointer to vector column indexes
        /// </summary>
        int[] idxPtr;
        /// <summary>
        /// pointer to vector lenght array, vector length is divided by number of threads
        /// </summary>
        int[] vecLenghtPtr;
        /// <summary>
        /// pointer to array with slice start
        /// </summary>
        int[] sliceStartPtr;

        /// <summary>
        /// pointer to self dot product array
        /// </summary>
        private float[] selfLinDotPtr;

        private float[] labelsPtr;

        private IntPtr outputIntPtr;

        private object outputPtr;
        
        #endregion

        int blockPerGrid;
        int blockSize;
        int threadsPerRow;
        int sliceSize;
        int align;

        private string cudaMainVecTexRefName = "mainVecTexRef";
        string cudaFunctionName = "rbfSlicedEllpackKernel";
        private float Gamma;
        
        

        public CuRBFSlicedEllpackKernel(float gamma)
        {
            linKernel = new LinearKernel();
            Gamma = gamma;

            threadsPerRow = 4;
            sliceSize = 64;
        }


        public override float Product(SparseVec element1, SparseVec element2)
        {

            float x1Squere = element1.DotProduct();
            float x2Squere = element2.DotProduct();//linKernel.Product(element2, element2);
            float dot = element1.DotProduct(element2); //linKernel.Product(element1, element2);

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


        public override void AllProducts(int element1, float[] results)
        {
            //base.AllProducts(element1, results);
            var mainVec = problemElements[element1];
            if (mainVecIdx != element1)
            {
                CudaHelpers.FillDenseVector(mainVec,mainVector);

                cuGPU.CopyHostToDevice(mainVectorPtr,mainVector);
            }

            mainVecIdx = element1;

            //float elapsed;
            //gpu.StartTimer();
            gpu.Launch(blockPerGrid,blockSize,cudaFunctionName,valsPtr,idxPtr,vecLenghtPtr,sliceStartPtr,selfLinDotPtr,labelsPtr,outputPtr,mainVecIdx,problemElements.Length,Gamma,align);

            //elapsed = gpu.StopTimer();
            gpu.Synchronize();

            //elapsed = gpu.StopTimer();
            Marshal.Copy(outputIntPtr, results, 0, results.Length);

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

            

            blockSize = threadsPerRow * sliceSize;
            int N=problemElements.Length;
            blockPerGrid = (int)Math.Ceiling(1.0*N*threadsPerRow/blockSize);
            
            align = (int)Math.Ceiling( 1.0*sliceSize * threadsPerRow / 64)*64;

            float[] vecVals;
            int[] vecColIdx;
            int[] vecLenght;
            int[] sliceStart;

            CudaHelpers.TransformToSlicedEllpack(out vecVals, out vecColIdx, out sliceStart, out vecLenght, problemElements, threadsPerRow,sliceSize);

            selfLinDot = linKernel.DiagonalDotCache;

            #region cudafy initialization

            InitCudaModule();

            //copy data to device, set cuda function parameters
            valsPtr = gpu.CopyToDevice(vecVals);
            idxPtr = gpu.CopyToDevice(vecColIdx);
            vecLenghtPtr = gpu.CopyToDevice(vecLenght);
            sliceStartPtr = gpu.CopyToDevice(sliceStart);
            //!!!!!
            selfLinDotPtr = gpu.CopyToDevice(selfLinDot);
            labelsPtr = gpu.CopyToDevice(Y);



          
            //gpu.CopyToConstantMemory(new float[] { Gamma }, GammaDev);

            //float[] GammaDev =new float[] { Gamma };
            //float[] GammaDevPtr = gpu.Allocate<float>(1);
            //gpu.CopyToConstantMemory<float>(GammaDev,GammaDevPtr);

            //float[] Gammas = new float[] { Gamma };
            //float[] GammaDev = gpu.Allocate<float>(1);
            //gpu.CopyToConstantMemory<float>(Gammas, GammaDev);


            int memSize = (problemElements.Length * sizeof(float));

            
            //allocate mapped memory for our results
            //outputIntPtr = gpu.HostAllocate<float>(problemElements.Length); // .HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            //outputPtr = gpu.GetDeviceMemoryFromIntPtr(outputIntPtr);// cuda.GetHostDevicePointer(outputIntPtr, 0);

            outputIntPtr = cuGPU.HostAllocate((uint)memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            outputPtr =  cuGPU.GetHostDevicePointer(outputIntPtr, 0);

            #endregion


            
            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[problemElements[0].Dim + 1];
            CudaHelpers.FillDenseVector(problemElements[0], mainVector);
                        
           

            CudaHelpers.SetTextureMemory(cuGPU, ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector, ref mainVectorPtr);
            //CudaHelpers.SetTextureMemory(cuGPU, ref cuLabelsTexRef, cudaLabelsTexRefName, Y, ref labelsPtr);
        }

        private void InitCudaModule()
        {
            cufy.CudafyModes.Target = cufy.eGPUType.Cuda;

            gpu = CudafyHost.GetDevice(CudafyModes.Target);
            cuGPU = (CUDA)((CudaGPU)gpu).CudaDotNet;
            var ctx = cuGPU.CreateContext(0, CUCtxFlags.MapHost);
            cuGPU.SetCurrentContext(ctx);

           // gpu.EnableSmartCopy();
            
            module = CudafyModule.TryDeserialize(moduleName);
            if (module == null || !module.TryVerifyChecksums())
            {
                module = CudafyTranslator.Cudafy(typeof(CuRBFSlicedEllpackKernel));
                module.Serialize();
            }
            gpu.LoadModule(module);
        }


     

        [CudafyDummy]
        public static void rbfSlicedEllpackKernel(GThread th, float[] vecVals, int[] vecCols, int[] vecLengths, int[] sliceStart, float[] result,int mainVecIdx, int nrRows,float gamma, int align)
        {
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
