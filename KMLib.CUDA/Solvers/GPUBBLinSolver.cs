using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.SVMSolvers;
using KMLib.Helpers;
using System.Diagnostics;
using GASS.CUDA.Types;
using GASS.CUDA;
using System.IO;

namespace KMLib.GPU.Solvers
{



    /// <summary>
    /// Solver for linear SVM, use Barzilai-Borwein formula for computing optimialization step 
    /// version for computing on CUDA devices
    /// author: Krzysztof Sopyła (krzysztofsopyla@gmail.com)
    /// </summary>
    public class GPUBBLinSolver : LinearSolver
    {
        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected const string cudaModuleName = "linSVMSolver.cubin";


        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected const string cudaWVecTexRefName = "mainVectorTexRef";
        /// <summary>
        /// cuda texture name for labels
        /// </summary>
        protected const string cudaLabelsTexRefName = "labelsTexRef";

        /// <summary>
        /// cuda function name for computing product between W vector and all vector elements
        /// </summary>
        protected string cudaProductKernelName = "ComputeDotProd";

        /// <summary>
        /// cuda function name for computing gradient
        /// </summary>
        protected string cudaGradFinalizeName = "ComputeGradient";

        /// <summary>
        /// cuda function name for computing BB step
        /// </summary>
        protected string cudaComputeBBStepName = "ComputeBBstep";



        /// <summary>
        /// cuda function name for updating W vector
        /// </summary>
        protected string cudaUpdateW = "update_W";

        protected string cudaUpdateAlphaName = "UpdateAlpha";

        /// <summary>
        /// cuda function name for computing objective function value, square W
        /// </summary>
        private string cudaObjWName="VectorSquareW";

        /// <summary>
        /// cuda function name for computing objective function value, square alpha
        /// </summary>
        private string cudaObjAlphaName="VectorSquareAlpha";

        /// <summary>
        /// cuda function name for compuing Lin part in nonmonotonus line search
        /// </summary>
        private string cudaLinPartName = "ComputeLinPart";
        
        
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
        /// cuda kernel function for finalizing computing gradient
        /// </summary>
        protected CUfunction cuFuncGradFinalize;

        /// <summary>
        /// cuda kernel function for updating alpha
        /// </summary>
        protected CUfunction cuFuncUpdateAlpha;

        /// <summary>
        /// cuda kernel function for updating vector W
        /// </summary>
        protected CUfunction cuFuncUpdateW;


        /// <summary>
        /// cuda kernel function for Computing Barzilai-Borwein step
        /// </summary>
        protected CUfunction cuFuncComputeBBstep;


        /// <summary>
        /// cuda kernel function for computing function value for line search
        /// </summary>
        protected CUfunction cuFuncObjSquareW;

        protected CUfunction cuFuncObjSquareAlpha;


        private CUfunction cuFuncLinPart;


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
        /// cuda device pointer for gradient
        /// </summary>
        protected CUdeviceptr gradPtr;

        /// <summary>
        /// cuda device pointer for previous gradient
        /// </summary>
        protected CUdeviceptr gradOldPtr;


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
        /// cuda device pointer for previous alpha array's
        /// </summary>
        private CUdeviceptr alphaOldPtr;

        /// <summary>
        /// cuda device pointer for previous alpha array's
        /// </summary>
        private CUdeviceptr alphaTmpPtr;

        //arrays for kernels using reduction, for computing BB step
        private CUdeviceptr reduceBBAlphaPtr;
        private CUdeviceptr reduceBBGradPtr;
        private CUdeviceptr reduceBBAlphaGradPtr;

        //arrays for kernels using reduction, for computing objective function value
        private CUdeviceptr reduceObjAlphaPtr;
        private CUdeviceptr reduceObjWPtr;

        /// <summary>
        /// cuda device pointer for diag, needed for coping to constant array on device
        /// </summary>
        private CUdeviceptr diagPtr;

        /// <summary>
        /// cuda device pointer for step, needed for coping to constant array on device
        /// </summary>
        private CUdeviceptr stepBBPtr;

        /// <summary>
        /// cuda reference to texture for main problem element(vector), 
        /// </summary>
        protected CUtexref cuWVecTexRef;

        /// <summary>
        /// cuda device pointer for "w" vector, 
        /// </summary>
        protected CUdeviceptr wVecPtr;

        /// <summary>
        /// cuda device pointer for temporary "w" vector, 
        /// </summary>
        protected CUdeviceptr wTempVecPtr;

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
        /// native pointer to memory region, used for computing steps in solver
        /// </summary>
        protected IntPtr deltasIntPtr;




        protected int threadsPerBlock = CUDAConfig.XBlockSize;

        /// <summary>
        /// indicates how many blocks pre grid we create for cuda kernel launch
        /// </summary>
        protected int blocksPerGrid = -1;

        /// <summary>
        /// Offset in parameter list for cuda kernel computing dot product
        /// </summary>
        private int gradParamOffsetInCudaDotProd;
        private int gradParamOffsetInGradFinalize;
        private int alphaParamOffsetInGradFinalize;
        private int wVecParamOffsetInUpdateW;
        private int gradParamOffsetInUpdateAlpha;
        private int alphaParamOffsetInUpdateAlpha;

        private Random rnd = new Random();
        private uint alphaMemSize;
        private uint wVecMemSize;
        private int wVecParamOffsetInObjSquareW;
        private int alphaParamOffsetInObjSquareAlpha;
        private float[] reduceObjW;
        private float[] reduceObjAlpha;
        private int alphaParamOffsetInBBStep;
        private int alphaOldParamOffsetInBBStep;
        private int gradParamOffsetInBBStep;
        private int gradOldParamOffsetInBBStep;
        private int alphaParamOffsetInLinPart;
        private int alphaOldParamOffsetInLinPart;
        private int gradParamOffsetInLinPart;
        

        /// <summary>
        /// Construct linear solver
        /// </summary>
        /// <param name="problem">trainning problem</param>
        /// <param name="C">penalty parameter</param>
        public GPUBBLinSolver(Problem<SparseVec> problem, float C)
            : base(problem, C)
        {
        }

        public GPUBBLinSolver(Problem<SparseVec> problem, float C, int[] weightedLabels, double[] weights)
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
                FillDataOnCuda(sub_prob, w, weighted_C[0], weighted_C[1]);

                solve_l2r_l2_bb_svc_cuda(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);
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

                    FillDataOnCuda(sub_prob, w, weighted_C[i], C);
                    //train_one(sub_prob, param, w, weighted_C[i], param.C);
                    solve_l2r_l2_bb_svc_cuda(sub_prob, w, epsilon, weighted_C[i], C);

                    for (j = 0; j < n; j++)
                        model.W[j * nr_class + i] = w[j];
                }
            }

            DisposeCuda();

            return model;
        }

       




        /// <summary>
        /// Barzilai-Borwein solver
        /// </summary>
        /// <param name="sub_prob"></param>
        /// <param name="w"></param>
        /// <param name="epsilon"></param>
        /// <param name="Cp"></param>
        /// <param name="Cn"></param>
        private void solve_l2r_l2_bb_svc_cuda(Problem<SparseVec> sub_prob, float[] w, double epsilon, double Cp, double Cn)
        {
          
            //blocks per Grid for solver kernel
            int bpgSolver = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;
            //blocks per Grid for update_W kernel
            int bpgUpdateW = (sub_prob.Elements[0].Dim + threadsPerBlock - 1) / threadsPerBlock;

            float obj = float.PositiveInfinity;
            float step = 0.01f;

            int M = 10;
            float sig1 = 0.1f;
            float sig2 = 0.9f;
            float gamma = 10e-4f;
            float lambda = 0;
            float l_min = 10e-20f;
            float l_max = 10e20f;
            float[] func_vals = new float[M];
            float maxFuncVal = 0;

            int maxIter = 20000;
            int iter = 0;

            ComputeGradient(sub_prob);

            while (iter <= maxIter)
            {
                
                maxFuncVal = func_vals.Max();


                lambda = step;
                //nonemonotone line search
                for (int i = 0; i < 10; i++)
                {
                    DoBBstep(-lambda,sub_prob);

                    //compute Obj
                    obj = ComputeObjGPU();

                    //compute linpart
                    float linPart = ComputeLinPart(gamma,lambda); //* (alpha_tmp-alpha)'*grad

                    if (obj <= (maxFuncVal + linPart))
                    {
                        int idx = (iter + 1) % M;
                        func_vals[idx] = obj;
                        break;

                    }
                    lambda = (sig1 * lambda + sig2 * lambda) / 2;
                }

                //change alpha's pointers
                //tmpPtr = alphaOldPtr;
                //alphaOldPtr = alphaPtr;
                //alphaPtr = alphaTmpPtr;
                //alphaTmpPtr = tmpPtr;

                //change w - pointers
                //tempPtr= w
                //w = w_tmp
                //w_tmp = w

                //change gradients
                //gradOldPtr = grad
                //compute new grad
                ComputeGradient(sub_prob);

                //compute BB step
                step = ComputeBBStep();

                iter++;
            }
        }


        /// <summary>
        /// Do BB step, updates alpha and "w" vector
        /// </summary>
        /// <param name="step"></param>
        /// <param name="sub_prob"></param>
        private void DoBBstep(float step, Problem<SparseVec> sub_prob)
        {
       
            throw new NotImplementedException();

            int blocks = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;

            /*
             * Update alpha
             * 
             * 1. copy alpha to alphaTmp
             * 2. copy step into device constant
             * 3. set parameters
             * 
             */ 
            cuda.CopyDeviceToDevice(alphaPtr, alphaTmpPtr, alphaMemSize);

            float[] stepData = new float[] { step };
            cuda.CopyHostToDevice(stepBBPtr, stepData);
            cuda.SetParameter(cuFuncUpdateAlpha, gradParamOffsetInUpdateAlpha, gradPtr.Pointer);
            cuda.SetParameter(cuFuncUpdateAlpha, alphaParamOffsetInUpdateAlpha, alphaTmpPtr.Pointer);
            cuda.Launch(cuFuncUpdateAlpha, blocks, 1);

            /*
             * Update w - based on aplha deltas
             * 
             */
            int bpgUpdateW = (sub_prob.Elements[0].Dim + threadsPerBlock - 1) / threadsPerBlock;

            cuda.CopyDeviceToDevice(wVecPtr, wTempVecPtr, wVecMemSize);
            cuda.Launch(cuFuncUpdateW, bpgUpdateW, 1);


        }

        private float ComputeLinPart(double gamma, double lambda)
        {
            throw new NotImplementedException();


            cuda.SetParameter(cuFunc )
        }


        /// <summary>
        /// compute objective value on GPU
        /// </summary>
        /// <returns></returns>
        private float ComputeObjGPU()
        {
            int bpgReduceObj = -1;
            int bpgReduceAlphaObj = -1;
            /*
             * compute w'*w
             */
            cuda.SetParameter(cuFuncObjSquareW, wVecParamOffsetInObjSquareW, wVecPtr.Pointer);
            cuda.Launch(cuFuncObjSquareW, bpgReduceObj, 1);

            cuda.CopyDeviceToHost(reduceObjWPtr, reduceObjW);

            /*
             * compute alpha[i] * (alpha[i] * diag[y_i + 1] - 2);
             */
            cuda.SetParameter(cuFuncObjSquareAlpha, alphaParamOffsetInObjSquareAlpha, alphaPtr.Pointer);
            cuda.Launch(cuFuncObjSquareAlpha, bpgReduceAlphaObj, 1);

            cuda.CopyDeviceToHost(reduceObjAlphaPtr, reduceObjAlpha);

            /*
             * Do reduction on CPU
             */ 
            float val = 0;
            for (int i = 0; i < reduceObjW.Length; i++)
            {
                val += reduceObjW[i];
            }

            for (int k = 0; k < reduceObjAlpha.Length; k++)
            {
                val += reduceObjAlpha[k];
            }

            val = val / 2;

            return val;


        }

        private float ComputeBBStep()
        {
            throw new NotImplementedException();

            int bpgBBstep =0;

            cuda.SetParameter(cuFuncComputeBBstep, alphaParamOffsetInBBStep, alphaPtr.Pointer);
            cuda.SetParameter(cuFuncComputeBBstep, alphaOldParamOffsetInBBStep, alphaOldPtr.Pointer);
            cuda.SetParameter(cuFuncComputeBBstep, gradParamOffsetInBBStep, gradPtr.Pointer);
            cuda.SetParameter(cuFuncComputeBBstep, gradOldParamOffsetInBBStep, gradOldPtr.Pointer);

            //partial reduction on CPU
            cuda.Launch(cuFuncComputeBBstep, bpgBBstep, 1);


            float[] alphaPartReduce = new float[bpgBBstep];
            float[] gradPartReduce = new float[bpgBBstep];
            float[] alphaGradPartReduce = new float[bpgBBstep];
            //copy from device partial reduce sums
            cuda.CopyDeviceToHost(reduceBBAlphaPtr, alphaPartReduce);
            cuda.CopyDeviceToHost(reduceBBGradPtr, gradPartReduce);
            cuda.CopyDeviceToHost(reduceBBAlphaGradPtr, alphaGradPartReduce);

            cuda.SynchronizeContext();

            float alphaPart = 0, gradPart = 0, alphaGradPart = 0;
            
            //final reduction on CPU
            for (int i = 0; i < bpgBBstep; i++)
            {
                alphaPart += alphaPartReduce[i];
                gradPart += gradPartReduce[i];
                alphaGradPart += alphaGradPartReduce[i];
            }

            float step1 = alphaPart / alphaGradPart;
            float step2 = alphaGradPart / gradPart;

            float step = step1;
            
            //random step works better then modulo step (alternating iter%2)

            double rndProb = rnd.NextDouble();
            if (rndProb > 0.5f)
            {
                step = step2;
            }


            return step;
        }

        /// <summary>
        /// computes gradient for linear solver using GPU
        /// </summary>
        /// <param name="sub_prob"></param>
        private void ComputeGradient(Problem<SparseVec> sub_prob)
        {
            throw new NotImplementedException();

            //blocks per Grid for compuing dot prod
            int bpgDotProd = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;
            int bpgGradFin = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;
            
            //remember old gradient
            var gradTmpPtr = gradOldPtr.Pointer;
            gradOldPtr.Pointer =  gradPtr.Pointer;
            gradPtr.Pointer = gradTmpPtr;

            cuda.SetParameter(cuFuncDotProd, gradParamOffsetInCudaDotProd, gradPtr.Pointer);

            //set wVec tex ref
            cuda.SetTextureAddress(cuWVecTexRef, wVecPtr, wVecMemSize);

            cuda.Launch(cuFuncDotProd, bpgDotProd, 1);
            
            cuda.Launch(cuFuncGradFinalize, bpgGradFin, 1);

            cuda.SynchronizeContext();

        }

        private void SetCudaData(Problem<SparseVec> sub_prob)
        {
            int vecDim = sub_prob.Elements[0].Dim;

            /* 
             * copy vectors to CUDA device
             */

            #region copy trainning examples to GPU
            
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

            #endregion
            /* 
            * allocate memory for gradient
            */
            alphaMemSize = (uint)(sub_prob.ElementsCount * sizeof(float));

            gradPtr = cuda.Allocate(alphaMemSize);
            gradOldPtr = cuda.Allocate(alphaMemSize);


            alphaPtr = cuda.Allocate(alphaMemSize);
            alphaOldPtr = cuda.Allocate(alphaMemSize);
            alphaTmpPtr = cuda.Allocate(alphaMemSize);

            uint wReductionBlocks = 0;
            reduceObjW = new float[wReductionBlocks];
            reduceObjWPtr = cuda.Allocate(wReductionBlocks);

            uint alphaReductionBlocks = 0;
            reduceObjAlpha = new float[alphaReductionBlocks];
            reduceObjAlphaPtr = cuda.Allocate(alphaReductionBlocks);

            //float[] wVec = new float[vecDim];
            wVecMemSize = (uint)vecDim * sizeof(float);
            wTempVecPtr = cuda.Allocate(wVecMemSize);
            //move W wector
            SetTextureMemory(ref cuWVecTexRef, cudaWVecTexRefName, ref wVecPtr, wVecMemSize);

            //set texture memory for labels
            SetTextureMemory(ref cuLabelsTexRef, cudaLabelsTexRefName, sub_prob.Y, ref labelsPtr);


            SetTextureMemory(ref cuDeltasTexRef, "deltasTexRef", ref deltasPtr, alphaMemSize);
           
            diagPtr = cuda.GetModuleGlobal(cuModule, "diag_shift");


            stepBBPtr = cuda.GetModuleGlobal(cuModule, "stepBB");
            float[] stepData = new float[] { 0.1f };
            cuda.CopyHostToDevice(stepBBPtr, stepData);

            SetCudaParameters(sub_prob);

        }

        private void SetCudaParameters(Problem<SparseVec> sub_prob)
        {
            /* 
             * Set Cuda functions parameters 
             */


            /*
             *  Set cuda function parmeters for computing Dot product
             */
            #region Set cuda function parmeters for computing Dot product

            cuda.SetFunctionBlockShape(cuFuncDotProd, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncDotProd, offset, valsCSRPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncDotProd, offset, idxCSRPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, vecLenghtCSRPtr.Pointer);
            offset += IntPtr.Size;

            gradParamOffsetInCudaDotProd = offset;
            cuda.SetParameter(cuFuncDotProd, offset, gradPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncDotProd, (uint)offset);
            #endregion


            #region Set Cuda function parameters for finalising gradient

            cuda.SetFunctionBlockShape(cuFuncGradFinalize, threadsPerBlock, 1, 1);
            int offset2 = 0;
            gradParamOffsetInGradFinalize = offset2;
            cuda.SetParameter(cuFuncGradFinalize, offset2, gradPtr.Pointer);
            offset2 += IntPtr.Size;

            alphaParamOffsetInGradFinalize = offset2;
            cuda.SetParameter(cuFuncGradFinalize, offset2, alphaPtr.Pointer);
            offset2 += IntPtr.Size;

            //cuda.SetParameter(cuFuncGradFinalize, offset2, deltasPtr.Pointer);
            //offset2 += IntPtr.Size;
            cuda.SetParameter(cuFuncGradFinalize, offset2, (uint)sub_prob.ElementsCount);
            offset2 += sizeof(int);

            cuda.SetParameterSize(cuFuncGradFinalize, (uint)offset2);

            #endregion

            #region Set cuda function param for updateing alpha

            cuda.SetFunctionBlockShape(cuFuncUpdateAlpha, threadsPerBlock, 1, 1);
            offset = 0;
            gradParamOffsetInUpdateAlpha = offset;
            cuda.SetParameter(cuFuncUpdateAlpha, offset, gradPtr.Pointer);
            offset += IntPtr.Size;

            alphaParamOffsetInUpdateAlpha = offset;
            cuda.SetParameter(cuFuncUpdateAlpha, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncUpdateAlpha, offset, deltasPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncUpdateAlpha, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncUpdateAlpha, (uint)offset);

            #endregion

            /********************/
            #region Set cuda function param for comuting BB step

            cuda.SetFunctionBlockShape(cuFuncComputeBBstep, threadsPerBlock, 1, 1);
            offset = 0;
            alphaParamOffsetInBBStep = offset;
            cuda.SetParameter(cuFuncComputeBBstep, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;

            alphaOldParamOffsetInBBStep = offset;
            cuda.SetParameter(cuFuncComputeBBstep, offset, alphaOldPtr.Pointer);
            offset += IntPtr.Size;

            gradParamOffsetInBBStep = offset;
            cuda.SetParameter(cuFuncComputeBBstep, offset, gradPtr.Pointer);
            offset += IntPtr.Size;

            gradOldParamOffsetInBBStep = offset;
            cuda.SetParameter(cuFuncComputeBBstep, offset, gradOldPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncComputeBBstep, offset, reduceBBAlphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncComputeBBstep, offset, reduceBBGradPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncComputeBBstep, offset, reduceBBAlphaGradPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncComputeBBstep, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncComputeBBstep, (uint)offset);
            #endregion


            /********************/
            #region Set cuda function param for computing objective function value

            /*
             * Set parameters for computing alpha square
             */ 
            cuda.SetFunctionBlockShape(cuFuncObjSquareAlpha, threadsPerBlock, 1, 1);
            offset = 0;
            alphaParamOffsetInObjSquareAlpha = offset;
            cuda.SetParameter(cuFuncObjSquareAlpha, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncObjSquareAlpha, offset, reduceObjAlphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncObjSquareAlpha, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFuncObjSquareAlpha, (uint)offset);

            
            

            cuda.SetFunctionBlockShape(cuFuncObjSquareW, threadsPerBlock, 1, 1);
            offset = 0;
            wVecParamOffsetInObjSquareW = offset;
            cuda.SetParameter(cuFuncObjSquareW, offset, wVecPtr.Pointer);
            offset += IntPtr.Size;
            
            cuda.SetParameter(cuFuncObjSquareW, offset, reduceObjWPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncObjSquareW, offset, (uint)sub_prob.FeaturesCount);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFuncObjSquareW, (uint)offset);

            #endregion

            #region set function parameters for computing lin part in nonemonotonus line search

            cuda.SetFunctionBlockShape(cuFuncLinPart, threadsPerBlock, 1, 1);
            offset = 0;
            alphaParamOffsetInLinPart = offset;
            cuda.SetParameter(cuFuncLinPart, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;
            alphaOldParamOffsetInLinPart = offset;
            cuda.SetParameter(cuFuncLinPart, offset, alphaOldPtr.Pointer);
            offset += IntPtr.Size;
            gradParamOffsetInLinPart = offset;
            cuda.SetParameter(cuFuncLinPart, offset, gradPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncLinPart, offset, reduceObjAlphaPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncLinPart, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFuncLinPart, (uint)offset);

            #endregion
            /*
             * Set cuda function parameters for updating W vector
             */
            #region Set cuda function parameters for updating W vector

            //todo: is threads per block for updates W corect?
            cuda.SetFunctionBlockShape(cuFuncUpdateW, threadsPerBlock, 1, 1);

            int offset3 = 0;
            cuda.SetParameter(cuFuncUpdateW, offset3, valsCSCPtr.Pointer);
            offset3 += IntPtr.Size;
            cuda.SetParameter(cuFuncUpdateW, offset3, idxCSCPtr.Pointer);
            offset3 += IntPtr.Size;

            cuda.SetParameter(cuFuncUpdateW, offset3, vecLenghtCSCPtr.Pointer);
            offset3 += IntPtr.Size;

            wVecParamOffsetInUpdateW = offset3;
            cuda.SetParameter(cuFuncUpdateW, offset3, wVecPtr.Pointer);
            offset3 += IntPtr.Size;

            //cuda.SetParameter(cuFuncUpdateW, offset3, (uint)(sub_prob.ElementsCount+50) );//[0].Dim-40) );
            cuda.SetParameter(cuFuncUpdateW, offset3, (uint)sub_prob.Elements[0].Dim);
            offset3 += sizeof(int);

            cuda.SetParameterSize(cuFuncUpdateW, (uint)offset3);

            #endregion
        }



        private void InitCudaModule()
        {
            cuda = new CUDA(0, true);
            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, cudaModuleName));

            cuFuncDotProd = cuda.GetModuleFunction(cudaProductKernelName);

            cuFuncGradFinalize = cuda.GetModuleFunction(cudaGradFinalizeName);

            cuFuncComputeBBstep = cuda.GetModuleFunction(cudaComputeBBStepName);

            cuFuncObjSquareW = cuda.GetModuleFunction(cudaObjWName);
            cuFuncObjSquareAlpha = cuda.GetModuleFunction(cudaObjAlphaName);

            cuFuncUpdateW = cuda.GetModuleFunction(cudaUpdateW);

            cuFuncUpdateAlpha = cuda.GetModuleFunction(cudaUpdateAlphaName);

            cuFuncLinPart = cuda.GetModuleFunction(cudaLinPartName);
        }



        /// <summary>
        /// set cuda texture memory 
        /// </summary>
        /// <param name="texture"></param>
        /// <param name="texName"></param>
        /// <param name="memPtr"></param>
        /// <param name="memSize"></param>
        private void SetTextureMemory(ref CUtexref texture, string texName, ref CUdeviceptr memPtr, uint memSize)
        {
            texture = cuda.GetModuleTexture(cuModule, texName);
            memPtr = cuda.Allocate(memSize);
            cuda.SetTextureAddress(texture, memPtr, memSize);
        }

        /// <summary>
        /// set cuda texture memory based on array
        /// </summary>
        /// <param name="texture"></param>
        /// <param name="texName"></param>
        /// <param name="data"></param>
        /// <param name="memPtr"></param>
        protected void SetTextureMemory(ref CUtexref texture, string texName, float[] data, ref CUdeviceptr memPtr)
        {
            texture = cuda.GetModuleTexture(cuModule, texName);
            memPtr = cuda.CopyHostToDevice(data);
            cuda.SetTextureAddress(texture, memPtr, (uint)(sizeof(float) * data.Length));

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

            float[] diag = new float[] { (float)(0.5 / Cn), 0, (float)(0.5 / Cp) };

            cuda.CopyHostToDevice(diagPtr, diag);

            //cuda.CopyHostToDevice(mainVecPtr, w);
        }


        private void DisposeCuda()
        {
            throw new NotImplementedException();
        }
    }
}
