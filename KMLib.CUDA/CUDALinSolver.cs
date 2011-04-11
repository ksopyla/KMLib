using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using KMLib.SVMSolvers;
using System.Diagnostics;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Runtime.InteropServices;

namespace KMLib.GPU
{
    /// <summary>
    /// Solver for linear SVM based on LIBLINEAR package
    /// http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    /// Paper: "A Dual Coordinate Descent Method for Large-scale Linear SVM" Hsieh et al., ICML 2008
    /// 
    /// modified version for computing on CUDA devices
    /// author: Krzysztof Sopyła (krzysztofsopyla@gmail.com)
    /// </summary>
    public class CUDALinSolver : LinearSolver
    {

        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected const string cudaModuleName = "linSVMSolver.cubin";


        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected const string cudaMainVecTexRefName = "mainVectorTexRef";
        /// <summary>
        /// cuda texture name for labels
        /// </summary>
        protected const string cudaLabelsTexRefName = "labelsTexRef";

        /// <summary>
        /// cuda function name for computing product between W vector and all vector elements
        /// </summary>
        protected string cudaProductKernelName="ComputeDotProd";

        /// <summary>
        /// cuda function name for computing step
        /// </summary>
        protected string cudaSolveL2SVM = "lin_l2r_l2_svc_solver_with_gradient";

        /// <summary>
        /// cuda function name for updating W vector
        /// </summary>
        protected string cudaUpdateW = "update_W";


        #endregion

        #region cuda types

        /// <summary>
        /// Cuda .net class for cuda opeation
        /// </summary>
        protected CUDA cuda;


        /// <summary>
        /// cuda loaded module
        /// </summary>
        protected CUmodule cuModule;

        /// <summary>
        /// cuda kernel function for computing dot product beetween W and all elements
        /// </summary>
        protected CUfunction cuFuncDotProd;

        /// <summary>
        /// cuda kernel function for computing steps
        /// </summary>
        protected CUfunction cuFuncSolver;


        /// <summary>
        /// cuda kernel function for updating vector W
        /// </summary>
        protected CUfunction cuFuncUpdateW;

        /// <summary>
        /// Cuda device pointer to vectors values in CSR matrix  format
        /// </summary>                                  
        protected CUdeviceptr valsCSRPtr;               
        /// <summary>                                   
        /// cuda devie pointer to vectors indexes in CSR matrix format
        /// </summary>                                    
        protected CUdeviceptr idxCSRPtr;                  
        /// <summary>                                     
        /// cuda device pointer to vectors lenght in CSR matrix format
        /// </summary>
        protected CUdeviceptr vecLenghtCSRPtr;


        /// <summary>
        /// Cuda device pointer to vectors values in CSC matrix format
        /// </summary>                                   
        protected CUdeviceptr valsCSCPtr;                
        /// <summary>                                    
        /// cuda devie pointer to vectors indexes in CSC matrix format
        /// </summary>                                  
        protected CUdeviceptr idxCSCPtr;                
        /// <summary>                                   
        /// cuda device pointer to vectors lenght in CSC matrix format
        /// </summary>
        protected CUdeviceptr vecLenghtCSCPtr;


        /// <summary>
        /// cuda device pointer to diagonal cache QD
        /// </summary>
        private CUdeviceptr qdPtr;


        /// <summary>
        /// cuda device pointer for output
        /// </summary>
        protected CUdeviceptr gradPtr;

        /// <summary>
        /// cuda device pointer for output deltas
        /// </summary>
        protected CUdeviceptr deltasPtr;


        /// <summary>
        /// cuda reference to texture for deltas, 
        /// </summary>
        protected CUtexref cuDeltasTexRef;

        /// <summary>
        /// cuda device pointer for alpha array's
        /// </summary>
        private CUdeviceptr alphaPtr;


        /// <summary>
        /// cuda device pointer for diag, needed for coping to constant array on device
        /// </summary>
        private CUdeviceptr diagPtr;

        /// <summary>
        /// cuda reference to texture for main problem element(vector), 
        /// </summary>
        protected CUtexref cuMainVecTexRef;

        protected CUdeviceptr mainVecPtr;

        /// <summary>
        /// cuda refeerenc to texture for labels
        /// </summary>
        protected CUtexref cuLabelsTexRef;

        /// <summary>
        /// cuda pointer to labels, neded for coping to texture
        /// </summary>
        protected CUdeviceptr labelsPtr;


        #endregion


        /// <summary>
        /// vector "W" for computing kernel product with other vectors
        /// </summary>
        /// <remarks>all the time this vector will be modified and copied to cuda array</remarks>
        protected float[] mainVector;

        

        /// <summary>
        /// native pointer to memory region, used for computing  dot product and gradiens in solver
        /// </summary>
        protected IntPtr gradIntPtr;

        /// <summary>
        /// native pointer to memory region, used for computing steps in solver
        /// </summary>
        protected IntPtr deltasIntPtr;
        private float[] alpha;
        private float[] deltas;
        private float[] QD;
      


        protected int threadsPerBlock = CUDAConfig.XBlockSize;

        /// <summary>
        /// indicates how many blocks pre grid we create for cuda kernel launch
        /// </summary>
        protected int blocksPerGrid = -1;
       

        /// <summary>
        /// Construct linear solver
        /// </summary>
        /// <param name="problem">trainning problem</param>
        /// <param name="C">penalty parameter</param>
        public CUDALinSolver(Problem<SparseVec> problem, float C)
            : base(problem, C)
        {
        }

        public CUDALinSolver(Problem<SparseVec> problem, float C, int[] weightedLabels, double[] weights)
            : base(problem, C, weightedLabels, weights)
        {
        }

        public override Model<SparseVec> ComputeModel()
        {

            int j;
            int l = problem.ElementsCount; //prob.l;
            int n = problem.FeaturesCount;// prob.n;
            int w_size = n; // prob.n;
            Model<SparseVec> model = new Model<SparseVec>();
            model.FeaturesCount = n;

            if (bias >= 0)
            {
                //Add to each feature vector last feature ==1;
                model.FeaturesCount = n - 1;
            }
            else
                model.FeaturesCount = n;

            model.Bias = bias;

            int[] perm = new int[l];
            // group training data of the same class
            //GroupClassesReturn rv = groupClasses(prob, perm);
            int nr_class = 0;
            int[] label;// = new int[l];// = rv.label;
            int[] start;// = rv.start;
            int[] count;// = rv.count;

            groupClasses(problem, out nr_class, out label, out start, out count, perm);

            model.NumberOfClasses = nr_class;


            model.Labels = new float[nr_class];
            for (int i = 0; i < nr_class; i++)
                model.Labels[i] = (float)label[i];

            // calculate weighted C
            double[] weighted_C = new double[nr_class];
            for (int i = 0; i < nr_class; i++)
            {
                weighted_C[i] = C;
            }


            SetClassWeights(nr_class, label, weighted_C);

            // constructing the subproblem
            //permutated vectors
            SparseVec[] permVec = new SparseVec[problem.ElementsCount];
            Debug.Assert(l == problem.ElementsCount);
            for (int i = 0; i < l; i++)
            {
                permVec[i] = problem.Elements[perm[i]];
            }


            Problem<SparseVec> sub_prob = new Problem<SparseVec>();
            sub_prob.ElementsCount = l;
            sub_prob.FeaturesCount = n;
            //we set labels below
            sub_prob.Y = new float[sub_prob.ElementsCount];
            sub_prob.Elements = permVec;

            //Initailize CUDA driver and load module
            InitCudaModule();

            if (nr_class == 2)
            {
               
                float[] w = new float[w_size];
                

                int e0 = start[0] + count[0];
                int k = 0;
                for (; k < e0; k++)
                    sub_prob.Y[k] = +1;
                for (; k < sub_prob.ElementsCount; k++)
                    sub_prob.Y[k] = -1;

                //copy all needed data to CUDA device
                SetCudaData(sub_prob);

                //Fill data on CUDA
                FillDataOnCuda(sub_prob,w, weighted_C[0],weighted_C[1]);

                solve_l2r_l2_svc_cuda(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);
                //solve_l2r_l1l2_svc(model.W, epsilon, weighted_C[0], weighted_C[1], solverType);

                model.W = new double[w_size];
                for (int s = 0; s < w.Length; s++)
                {
                    model.W[s] = w[s];
                }
            }
            else
            {
                model.W = new double[w_size * nr_class];
                float[] w = new float[w_size];

                SetCudaData(sub_prob);

                ///one against many
                for (int i = 0; i < nr_class; i++)
                {
                    int si = start[i];
                    int ei = si + count[i];

                    int k = 0;
                    for (; k < si; k++)
                        sub_prob.Y[k] = -1;
                    for (; k < ei; k++)
                        sub_prob.Y[k] = +1;
                    for (; k < sub_prob.ElementsCount; k++)
                        sub_prob.Y[k] = -1;

                    FillDataOnCuda(sub_prob,w, weighted_C[i],C);
                    //train_one(sub_prob, param, w, weighted_C[i], param.C);
                    solve_l2r_l2_svc_cuda(sub_prob, w, epsilon, weighted_C[i], C);

                    for (j = 0; j < n; j++)
                        model.W[j * nr_class + i] = w[j];
                }
            }

            DisposeCuda();

            return model;
        }


        /// <summary>
        /// fill data which can be changed like
        /// Y - labels
        /// diag -
        /// </summary>
        /// <param name="sub_prob"></param>
        /// <param name="w"></param>
        /// <param name="Cn"></param>
        /// <param name="Cp"></param>
        private void FillDataOnCuda(Problem<SparseVec> sub_prob, float[] w, double Cn, double Cp)
        {
            cuda.CopyHostToDevice(labelsPtr, sub_prob.Y);

            float[] diag = new float[] { (float)(0.5/Cn),0 , (float)(0.5/Cp)};
            
            cuda.CopyHostToDevice(diagPtr, diag);

            cuda.CopyHostToDevice(mainVecPtr,w);
        }

        private void SetCudaData(Problem<SparseVec> sub_prob)
        {

            int vecDim = sub_prob.Elements[0].Dim;

            /* 
             * copy vectors to CUDA device
             */ 
            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;
            CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght, sub_prob.Elements);
            valsCSRPtr = cuda.CopyHostToDevice(vecVals);
            idxCSRPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtCSRPtr = cuda.CopyHostToDevice(vecLenght);


            CudaHelpers.TransformToCSCFormat(out vecVals, out vecIdx, out vecLenght, sub_prob.Elements);
            valsCSCPtr = cuda.CopyHostToDevice(vecVals);
            idxCSCPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtCSCPtr = cuda.CopyHostToDevice(vecLenght);
            


            /* 
             * allocate memory for gradient
             */ 
            uint memSize = (uint)(sub_prob.ElementsCount * sizeof(float));
            //allocate mapped memory for our results (dot product beetween vector W and all elements)
            gradIntPtr = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            gradPtr = cuda.GetHostDevicePointer(gradIntPtr, 0);

            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[vecDim + 1];
            //move W wector
            //CudaHelpers.FillDenseVector(problemElements[0], mainVector);
            SetTextureMemory(ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector, ref mainVecPtr);


            //set texture memory for labels
            SetTextureMemory(ref cuLabelsTexRef, cudaLabelsTexRefName, sub_prob.Y, ref labelsPtr);


            /*
             * data for cuda solver
            */

            //normaly for L2 solver QDii= xii*xii+Diag_i
            //where Diag_i = 0.5/Cp if yi=1
            //      Diag_i = 0.5/Cn if yi=-1
            //but we will add this on GPU
            QD = new float[sub_prob.ElementsCount];
            alpha = new float[sub_prob.ElementsCount];
            deltas = new float[sub_prob.ElementsCount];
            float[] diag = new float[3];
            for (int i = 0; i < sub_prob.ElementsCount; i++)
            {
                QD[i] = sub_prob.Elements[i].DotProduct();
                alpha[i] = 0;
                deltas[i] = 0;
            }

            qdPtr = cuda.CopyHostToDevice(QD);

            alphaPtr = cuda.Allocate(alpha);


            //deltasPtr = cuda.Allocate(deltas);
            SetTextureMemory(ref cuDeltasTexRef, "deltasTexRef", deltas, ref deltasPtr);

            diagPtr = cuda.GetModuleGlobal(cuModule, "diag_shift");
            //set this in fill function
            //cuda.CopyHostToDevice(diagPtr, diag);

            CUdeviceptr dimPtr = cuda.GetModuleGlobal(cuModule, "Dim");
            //todo: check if it ok
            cuda.Memset(dimPtr,(uint) vecDim, 1);
            //int[] dimArr = new int[] { vecDim };
            //cuda.CopyHostToDevice(dimPtr,dimArr);
            
            //CUDARuntime.cudaMemcpyToSymbol("Dim", dimPtr, 1, 0, cudaMemcpyKind.cudaMemcpyHostToDevice);
            //CUDARuntime.cudaMemcpyToSymbol("Dim", ,1,0, cudaMemcpyKind.cudaMemcpyHostToDevice);

            CUdeviceptr deltaScalingPtr = cuda.GetModuleGlobal(cuModule, "DimRSqrt");

            //two ways of computing scaling param, should be the same, but it depends on rounding.
            float scaling =(float) ( 1.0/ Math.Sqrt(vecDim));
            float scaling2 = (float)(Math.Sqrt(vecDim)/vecDim);
            //only for debug, 
            Debug.Assert(scaling == scaling2, "scaling param not equal");
            //set scaling constant
            cuda.Memset(deltaScalingPtr,(uint) scaling, 1);

            //cuda.CopyHostToDevice(dimPtr, problem.Elements[0].Dim);

            SetCudaParameters(sub_prob);
        }


        /// <summary>
        /// Sets parameters needeb by cuda kernels
        /// </summary>
        /// <param name="sub_prob"></param>
        private void SetCudaParameters(Problem<SparseVec> sub_prob)
        {
            /* 
             * Set Cuda functions parameters 
             */


            /*
             *  Set cuda function parmeters for computing Dot product
             */
            cuda.SetFunctionBlockShape(cuFuncDotProd, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncDotProd, offset, valsCSRPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncDotProd, offset, idxCSRPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, vecLenghtCSRPtr.Pointer);
            offset += IntPtr.Size;


            cuda.SetParameter(cuFuncDotProd, offset, gradPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncDotProd, (uint)offset);


            /*
             *  Set Cuda function parameters for computing deltas
             */
            //todo: is threads per block for solver corect?
            cuda.SetFunctionBlockShape(cuFuncSolver, threadsPerBlock, 1, 1);
            int offset2 = 0;
            cuda.SetParameter(cuFuncSolver, offset2, qdPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameter(cuFuncSolver, offset2, alphaPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameter(cuFuncSolver, offset2, gradPtr.Pointer);
            offset2 += IntPtr.Size;
            cuda.SetParameter(cuFuncSolver, offset2, deltasPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameterSize(cuFuncSolver, (uint)offset2);


            /*
             * Set cuda function parameters for updating W vector
             */

            //todo: is threads per block for updates W corect?
            cuda.SetFunctionBlockShape(cuFuncUpdateW, threadsPerBlock, 1, 1);

            int offset3 = 0;
            cuda.SetParameter(cuFuncUpdateW, offset3, valsCSCPtr.Pointer);
            offset3 += IntPtr.Size;
            cuda.SetParameter(cuFuncUpdateW, offset3, idxCSRPtr.Pointer);
            offset3 += IntPtr.Size;

            cuda.SetParameter(cuFuncUpdateW, offset3, vecLenghtCSRPtr.Pointer);
            offset3 += IntPtr.Size;


            cuda.SetParameter(cuFuncUpdateW, offset3, mainVecPtr.Pointer);
            offset3 += IntPtr.Size;

            cuda.SetParameter(cuFuncUpdateW, offset3, (uint)sub_prob.Elements[0].Dim);
            offset3 += sizeof(int);

            cuda.SetParameterSize(cuFuncUpdateW, (uint)offset3);
        }


       

       

        /// <summary>
        /// Dispose all object used by CUDA
        /// </summary>
        private void DisposeCuda()
        {
            if (cuda != null)
            {
                //free all resources
                cuda.Free(valsCSRPtr);
                cuda.Free(valsCSCPtr);
                valsCSRPtr.Pointer = 0;
                valsCSCPtr.Pointer = 0;
                
                cuda.Free(idxCSRPtr);
                cuda.Free(idxCSCPtr);
                idxCSRPtr.Pointer = 0;
                idxCSCPtr.Pointer = 0;
                
                cuda.Free(vecLenghtCSRPtr);
                cuda.Free(vecLenghtCSCPtr);
                vecLenghtCSRPtr.Pointer = 0;
                vecLenghtCSCPtr.Pointer = 0;

                cuda.Free(diagPtr);
                diagPtr.Pointer = 0;

                cuda.Free(qdPtr);
                qdPtr.Pointer = 0;
                cuda.Free(diagPtr);
                diagPtr.Pointer = 0;
                cuda.Free(alphaPtr);
                alphaPtr.Pointer = 0;
                cuda.Free(gradPtr);
                gradPtr.Pointer = 0;

                cuda.Free(deltasPtr);
                deltasPtr.Pointer = 0;
                cuda.DestroyTexture(cuDeltasTexRef);

                cuda.Free(labelsPtr);
                labelsPtr.Pointer = 0;
                cuda.DestroyTexture(cuLabelsTexRef);

                cuda.Free(mainVecPtr);
                mainVecPtr.Pointer = 0;

                cuda.DestroyTexture(cuMainVecTexRef);

                cuda.UnloadModule(cuModule);
                cuda.Dispose();
                cuda = null;
            }

        }

        private void solve_l2r_l2_svc_cuda(Problem<SparseVec> sub_prob, float[] w, double epsilon, double Cp, double Cn)
        {
            throw new NotImplementedException();
           
            //blocks per Grid for compuing dot prod
            int bpgDotProd = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;
            //blocks per Grid for solver kernel
            int bpgSolver = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;
            //blocks per Grid for update_W kernel
            int bpgUpdateW = (sub_prob.Elements[0].Dim + threadsPerBlock - 1) / threadsPerBlock;
           
            int maxIter = 20;
            int iter = 0;
            while (iter<maxIter)
            {

                //computes dot product between W and all elements

                cuda.Launch(cuFuncDotProd, bpgDotProd, 1);

                cuda.Launch(cuFuncSolver, bpgSolver, 1);

                cuda.Launch(cuFuncUpdateW, bpgUpdateW, 1);


                //take grad and check stop condition
                //Marshal.Copy(gradIntPtr, , 0, results.Length);

                iter++;
            }

            cuda.SynchronizeContext();
            //copy resulsts form device to host
            cuda.CopyDeviceToHost(mainVecPtr, w);
            //copy results from native mapped memory pointer to array,
            //faster then copyDtH function
           // Marshal.Copy(gradIntPtr, , 0, results.Length);

        }

        protected void InitCudaModule()
        {
            cuda = new CUDA(0, true);
            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, cudaModuleName));
            cuFuncDotProd  = cuda.GetModuleFunction(cudaProductKernelName);
            cuFuncSolver = cuda.GetModuleFunction(cudaSolveL2SVM);
            cuFuncUpdateW = cuda.GetModuleFunction(cudaUpdateW);
        }

        protected void SetTextureMemory(ref CUtexref texture, string texName, float[] data, ref CUdeviceptr memPtr)
        {
            texture = cuda.GetModuleTexture(cuModule, texName);
            memPtr = cuda.CopyHostToDevice(data);
            cuda.SetTextureAddress(texture, memPtr, (uint)(sizeof(float) * data.Length));

        }

    }
}
