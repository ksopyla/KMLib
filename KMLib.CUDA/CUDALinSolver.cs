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
        /// cuda kernel function
        /// </summary>
        protected CUfunction cuFuncDotProd;

        /// <summary>
        /// cuda kernel function
        /// </summary>
        protected CUfunction cuFuncSolver;

        /// <summary>
        /// Cuda device pointer to vectors values
        /// </summary>
        protected CUdeviceptr valsPtr;
        /// <summary>
        /// cuda devie pointer to vectors indexes
        /// </summary>
        protected CUdeviceptr idxPtr;
        /// <summary>
        /// cuda device pointer to vectors lenght
        /// </summary>
        protected CUdeviceptr vecLenghtPtr;

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
                model.W = new double[w_size];

                

                int e0 = start[0] + count[0];
                int k = 0;
                for (; k < e0; k++)
                    sub_prob.Y[k] = +1;
                for (; k < sub_prob.ElementsCount; k++)
                    sub_prob.Y[k] = -1;

                //copy all needed data to CUDA device
                SetCudaData(sub_prob);

                //Fill data on CUDA
                FillDataOnCuda(sub_prob,model.W, weighted_C[0],weighted_C[1]);

                solve_l2r_l2_svc_cuda(sub_prob, model.W, epsilon, weighted_C[0], weighted_C[1]);
                //solve_l2r_l1l2_svc(model.W, epsilon, weighted_C[0], weighted_C[1], solverType);
            }
            else
            {
                model.W = new double[w_size * nr_class];
                double[] w = new double[w_size];

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
        private void FillDataOnCuda(Problem<SparseVec> sub_prob, double[] w, double Cn, double Cp)
        {
            throw new NotImplementedException();
        }

        private void SetCudaData(Problem<SparseVec> sub_prob)
        {

            /* 
             * copy vectors to CUDA device
             */ 
            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;
            CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght, sub_prob.Elements);
            valsPtr = cuda.CopyHostToDevice(vecVals);
            idxPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtPtr = cuda.CopyHostToDevice(vecLenght);


            /* 
             * allocate memory for gradient
             */ 
            uint memSize = (uint)(sub_prob.ElementsCount * sizeof(float));
            //allocate mapped memory for our results (dot product beetween vector W and all elements)
            gradIntPtr = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            gradPtr = cuda.GetHostDevicePointer(gradIntPtr, 0);

            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[sub_prob.Elements[0].Dim + 1];
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
            deltasPtr = cuda.Allocate(deltas);

            diagPtr = cuda.GetModuleGlobal(cuModule, "diag_shift");
            
            //set this in fill function
            //cuda.CopyHostToDevice(diagPtr, diag);

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
            cuda.SetParameter(cuFuncDotProd, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncDotProd, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, vecLenghtPtr.Pointer);
            offset += IntPtr.Size;


            cuda.SetParameter(cuFuncDotProd, offset, gradPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncDotProd, (uint)offset);


            /*
             *  Set Cuda function parameters for computing deltas
             */

            cuda.SetFunctionBlockShape(cuFuncSolver, threadsPerBlock, 1, 1);
            int offset2 = 0;
            cuda.SetParameter(cuFuncDotProd, offset2, qdPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset2, alphaPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset2, gradPtr.Pointer);
            offset2 += IntPtr.Size;
            cuda.SetParameter(cuFuncDotProd, offset2, deltasPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameterSize(cuFuncDotProd, (uint)offset2);
        }


       

        /// <summary>
        /// Initialize CUDA driver, moves data to graphic card memory etc.
        /// </summary>
        /// <param name="sub_prob">Permuted and grupped sub problem</param>
        /// <param name="Cparams">different weight parameters for penalty C</param>
        private void InitCuda(Problem<SparseVec> sub_prob,double[] Cparams)
        {
          
            throw new NotImplementedException();



          //  SetCudaFunctionParameters();

            

            //SetTextureMemory(ref cuLabelsTexRef, cudaLabelsTexRefName, Y, ref labelsPtr);
        }

        /// <summary>
        /// Dispose all object used by CUDA
        /// </summary>
        private void DisposeCuda()
        {
            throw new NotImplementedException();
        }

        private void solve_l2r_l2_svc_cuda(Problem<SparseVec> sub_prob, double[] w, double epsilon, double Cp, double Cn)
        {
            throw new NotImplementedException();

        }

        protected void InitCudaModule()
        {
            cuda = new CUDA(0, true);
            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, cudaModuleName));
            cuFuncDotProd  = cuda.GetModuleFunction(cudaProductKernelName);
            cuFuncSolver = cuda.GetModuleFunction(cudaSolveL2SVM);
        }

        protected void SetTextureMemory(ref CUtexref texture, string texName, float[] data, ref CUdeviceptr memPtr)
        {
            texture = cuda.GetModuleTexture(cuModule, texName);
            memPtr = cuda.CopyHostToDevice(data);
            cuda.SetTextureAddress(texture, memPtr, (uint)(sizeof(float) * data.Length));

        }

    }
}
