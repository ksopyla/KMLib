using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using GASS.CUDA;
using GASS.CUDA.Types;




namespace KMLib.Kernels.GPU
{
    class CuLinearKernel : VectorKernel<SparseVector> , IDisposable
    {

        /// <summary>
        /// linear kernel for normal product
        /// </summary>
        private LinearKernel linKernel;


        /// <summary>
        /// Cuda .net class for cuda opeation
        /// </summary>
        private CUDA cuda;


        /// <summary>
        /// cuda loaded module
        /// </summary>
        CUmodule cuModule;

        /// <summary>
        /// cuda kernel function
        /// </summary>
        CUfunction cuFunc;


        /// <summary>
        /// Cuda device pointer to vectors values
        /// </summary>
        CUdeviceptr valsPtr ;
        /// <summary>
        /// cuda devie pointer to vectors indexes
        /// </summary>
        CUdeviceptr idxPtr ;
        /// <summary>
        /// cuda device pointer to vectors lenght
        /// </summary>
        CUdeviceptr vecLenghtPtr;

        /// <summary>
        /// cuda device pointer for output
        /// </summary>
        CUdeviceptr outputPtr;

        /// <summary>
        /// cuda reference to texture, 
        /// </summary>
        CUtexref cuTexRef;
        /// <summary>
        /// average vector lenght, its only a heuristic
        /// </summary>
        private int avgVectorLenght=50;

        static int threadsPerBlock = 256;
        static int blocksPerGrid = -1;

        public override SparseVector[] ProblemElements
        {
            set
            {
                if (value == null) throw new ArgumentNullException("value");
                linKernel.ProblemElements = value;

                base.ProblemElements = value;

                blocksPerGrid = (value.Length + threadsPerBlock - 1) / threadsPerBlock;
            }
        }


        public CuLinearKernel()
        {
            linKernel = new LinearKernel();

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

        public override float[] AllProducts(int element1)
        {

            //cuda calculation


            return base.AllProducts(element1);
        }


        public override void Init()
        {
            base.Init();
            
          


            //transform elements to specific array format -> CSR http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR_or_CRS.29
            //
            
            //list for all vectors values
            List<float> vecValsL = new List<float>(problemElements.Length*avgVectorLenght);
            
            //list for all vectors indexes
            List<int> vecIdxL = new List<int>(problemElements.Length*avgVectorLenght);
            
            //list of lenght of each vector, list of pointers
            List<int> vecLenghtL = new List<int>(problemElements.Length);

            //arrays for values, indexes and lenght
            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;

            
            int vecStartIdx = 0;
            for (int i = 0; i < problemElements.Length; i++)
            {
                var vec = problemElements[i];
                
                vecValsL.AddRange(vec.mValues);
                
                vecIdxL.AddRange(vec.mIndices);
                
                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vec.mValueCount;

            }
            //for last index
            vecLenghtL.Add(vecStartIdx);

            //convert list to arrays
            vecVals = vecValsL.ToArray();
            vecIdx = vecIdxL.ToArray();
            vecLenght = vecLenghtL.ToArray();

            //set list reference to null to free memeory
            vecIdxL = null;
            vecLenghtL = null;
            vecValsL = null;

            #region cuda initialization

            cuda = new CUDA(0, true);
            //copy data to device, set cuda function parameters
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);

            //alocate memory on device
            float[] output = new float[problemElements.Length];
            outputPtr = cuda.Allocate(output);

            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "structKernel.cubin"));
            cuFunc = cuda.GetModuleFunction("spmv_csr_vector_kernel");
            
            #endregion

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

            //this parameter is not nessesary
            cuda.SetParameter(cuFunc, offset, (uint)vecStartIdx);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFunc, (uint)offset);

            #endregion
            CUtexref cuTexRef = cuda.GetModuleTexture(cuModule, "texRef");
            cuda.SetTextureFlags(cuTexRef, 0);
        }


        #region IDisposable Members

        public void Dispose()
        {
            if (cuda != null)
            {

                //todo free all resources
                cuda.Free(valsPtr);
                cuda.Free(idxPtr);
                cuda.Free(outputPtr);
                cuda.Free(vecLenghtPtr);
                
                cuda.DestroyArray(cuArr);
               
                cuda.DestroyTexture(cuTexRef);

                cuda.Dispose();
                cuda = null;
            }
        }

        #endregion
    }
}
