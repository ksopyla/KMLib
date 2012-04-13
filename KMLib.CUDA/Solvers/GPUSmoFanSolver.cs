using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using KMLib.Kernels;
using KMLib.SVMSolvers;
using GASS.CUDA;
using System.IO;
using GASS.CUDA.Types;

namespace KMLib.GPU.Solvers
{
    /// <summary>
    /// SVM GPU L2 - solver
    ///  Solves:
    /// Min 0.5(\alpha^T Q \alpha) + p^T \alpha
    /// y^T \alpha = \delta
    /// y_i = +1 or -1
    /// 0 <= alpha_i <= Cp for y_i = 1
    /// 0 <= alpha_i <= Cn for y_i = -1
    /// solution will be put in \alpha, objective value will be put in obj
    /// </summary>
    /// <remarks>
    /// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918, implementation based on 
    ///  LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    ///  
    /// </remarks>
    public class GPUSmoFanSolver : Solver<SparseVec>
    {

        private CUDAVectorKernel gpuKernel;


        /// <summary>
        /// kernel diagonal
        /// </summary>
        private float[] QD;
        /// <summary>
        /// labels
        /// </summary>
        protected float[] y;
        /// <summary>
        /// gradient
        /// </summary>
        protected float[] G;		// gradient of objective function

        private float[] alpha;

        protected float EPS = 0.001f;

        protected const float INF = float.PositiveInfinity;



        private int problemSize;
        private CUDA cuda;
        private GASS.CUDA.Types.CUmodule cuModule;
        private string cudaModuleName = "gpuFanSmoSolver.cubin";
        private GASS.CUDA.Types.CUfunction cuFuncFindMaxI;
        private string funcFindMaxI = "FindMaxIdx";
        private GASS.CUDA.Types.CUfunction cuFuncFindMinJ;
        private string funcFindMinJ = "FindMinIdx";
        private GASS.CUDA.Types.CUfunction cuFuncFindStopping;
        private string funcFindStoping = "FindStoppingGradVal";
        private GASS.CUDA.Types.CUfunction cuFuncUpdateG;
        private string funcUpdateGFunc="UpdateGrad";


        /// <summary>
        /// maximum reduction blocks, def=64
        /// </summary>
        private int maxReductionBlocks = 64;
        /// <summary>
        /// threads per block, def=128
        /// </summary>
        private int threadsPerBlock = 128;

        /// <summary>
        /// number of blocks used for reduction, grid size
        /// </summary>
        private int reductionBlocks;
        /// <summary>
        /// numer of theread used for reduction
        /// </summary>
        private int reductionThreads;

        /// <summary>
        /// stores gradients value after GPU reduction, size is equal as reductionBlocks
        /// </summary>
        private float[] reduceGrad;
        /// <summary>
        /// stores idx after GPU reduction, size is equal reductionBlocks
        /// </summary>
        private int[] reduceIdx;

        private CUdeviceptr alphaPtr;
        private GASS.CUDA.Types.CUdeviceptr gradPtr;
        private GASS.CUDA.Types.CUdeviceptr yPtr;
        private GASS.CUDA.Types.CUdeviceptr kernelDiagPtr;
        private GASS.CUDA.Types.CUdeviceptr kiPtr;
        private GASS.CUDA.Types.CUdeviceptr kjPtr;
        private GASS.CUDA.Types.CUdeviceptr gradRedPtr;
        private GASS.CUDA.Types.CUdeviceptr idxRedPtr;
        private CUdeviceptr constCPtr;
        
        




        public GPUSmoFanSolver(Problem<SparseVec> problem, IKernel<SparseVec> kernel, float C)
            : base(problem, kernel, C)
        {
            this.C = C;
            problemSize = problem.ElementsCount;

            y = problem.Y;
            QD = kernel.DiagonalDotCache;
            alpha = new float[problemSize];
            G = new float[problemSize];
            gpuKernel = (CUDAVectorKernel)kernel;
        }


        /// <summary>
        /// Computes model by solving optimization problem
        /// </summary>
        /// <returns>Model</returns>
        public override Model<SparseVec> ComputeModel()
        {
            int problemSize = problem.ElementsCount;
            float[] alphaResult = new float[problem.ElementsCount];

            SolutionInfo si = new SolutionInfo();
            Solve(problem.Y, alphaResult, si);

            Model<SparseVec> model = new Model<SparseVec>();
            model.NumberOfClasses = 2;
            model.Alpha = alphaResult;
            model.Bias = si.rho;


            //------------------
            List<SparseVec> supportElements = new List<SparseVec>(alpha.Length);
            List<int> suporrtIndexes = new List<int>(alpha.Length);
            List<float> supportLabels = new List<float>(alpha.Length);
            for (int j = 0; j < alphaResult.Length; j++)
            {
                if (Math.Abs(alphaResult[j]) > 0)
                {
                    supportElements.Add(problem.Elements[j]);
                    suporrtIndexes.Add(j);
                    supportLabels.Add(problem.Y[j]);
                }

            }
            model.SupportElements = supportElements.ToArray();
            model.SupportElementsIndexes = suporrtIndexes.ToArray();
            model.Y = supportLabels.ToArray();



            return model;
        }

        /// <summary>
        /// Solves the optimization problem
        /// </summary>
        /// <param name="minusOnes"></param>
        /// <param name="y_"></param>
        /// <param name="alpha_"></param>
        /// <param name="si"></param>
        /// <param name="shrinking"></param>
        private void Solve(float[] y, float[] alpha, SolutionInfo si)
        {

            // initialize gradient
            {
                //G = new float[problemSize];

                int i;
                for (i = 0; i < problemSize; i++)
                {
                    G[i] = -1;
                }

            }


            InitCudaModule();

            SetCudaData();
            // optimization step
            float GMaxI;
            float GMaxJ;
            int iter = 1;


            while (true)
            {

                //Find i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
                Tuple<int, float> maxPair = FindMaxPair();
                int i = maxPair.Item1;
                GMaxI = maxPair.Item2;

                if (i % 255 == 0)
                {
                    GMaxJ = FindStoppingGradVal();
                }

                //Compute i-th kernel collumn, set the specific memory region on GPU, 
                ComputeKernel(i,kiPtr);


                // j: mimimizes the decrease of obj value
                //    (if quadratic coefficeint <= 0, replace it with tau)
                //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
                int j = FindMinPair(i, GMaxI);




                //Compute j-th kernel collumn
                ComputeKernel(j,kjPtr);


                float old_alpha_i = alpha[i];
                float old_alpha_j = alpha[j];

                //update alpha - serial code, one iteration

                // update gradient G
                float delta_alpha_i = alpha[i] - old_alpha_i;
                float delta_alpha_j = alpha[j] - old_alpha_j;

                //for (int k = 0; k < active_size; k++)
                //{
                //    G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
                //}

                //czy to potrzebne???
                // update alpha_status and G_bar,

            }//end while

            // calculate rho

            si.rho = (GMaxI + GMaxJ) / 2;

            //because we start from 1 not from 0;
            si.iter = iter - 1;
            // calculate objective value
            {
                float v = 0;
                int i;
                for (i = 0; i < problemSize; i++)
                    v += alpha[i] * (G[i] - 1);

                si.obj = v / 2;
            }


            si.upper_bound_p = C;
            si.upper_bound_n = C;
        }

        private void SetCudaData()
        {



            CudaHelpers.GetNumThreadsAndBlocks(problemSize, maxReductionBlocks, threadsPerBlock, ref reductionThreads, ref reductionBlocks);

            alphaPtr = cuda.CopyHostToDevice(alpha);
            gradPtr = cuda.CopyHostToDevice(G);
            yPtr = cuda.CopyHostToDevice(y);
            kernelDiagPtr = cuda.CopyHostToDevice(QD);

            //kernel columns i,j is simpler to copy array of zeros 
            kiPtr = cuda.CopyHostToDevice(alpha);
            kjPtr = cuda.CopyHostToDevice(alpha);

            reduceGrad = new float[reductionBlocks];
            gradRedPtr = cuda.CopyHostToDevice(reduceGrad);
            reduceIdx = new int[reductionBlocks];
            idxRedPtr = cuda.CopyHostToDevice(reduceIdx);


            constCPtr = cuda.GetModuleGlobal(cuModule, "C");
            float[] cData = new float[] { C };
            cuda.CopyHostToDevice(constCPtr, cData);

            SetCudaParams();

        }

        private void SetCudaParams()
        {

            #region Set cuda function parmeters for computing finding Max Idx

            cuda.SetFunctionBlockShape(cuFuncFindMaxI, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncFindMaxI, offset, yPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, gradPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, idxRedPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, gradRedPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, (uint)problemSize);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncFindMaxI, (uint)offset);
            #endregion


        }



        private Tuple<int, float> FindMaxPair()
        {
            cuda.Launch(cuFuncFindMaxI, reductionBlocks, 1);

            cuda.CopyDeviceToHost(gradRedPtr, reduceGrad);
            cuda.CopyDeviceToHost(idxRedPtr, reduceIdx);

            float max = float.NegativeInfinity;
            int idx = -1;
            for (int i = 0; i < reduceGrad.Length; i++)
            {
                if (max < reduceGrad[i])
                {
                    max = reduceGrad[i];
                    idx = reduceIdx[i];
                }
            }

            return new Tuple<int, float>(idx, max);

        }

        private int FindMinPair(int i, float GmaxI)
        {
            throw new NotImplementedException();
        }


        /// <summary>
        /// compute kernel and sets memory pointed by ptr (onec for i-th, once for j-th kernel column)
        /// </summary>
        /// <param name="i"></param>
        /// <param name="ptr"></param>
        private void ComputeKernel(int i,CUdeviceptr ptr)
        {
            gpuKernel.AllProductsGPU(i, ptr);

            float[] ki = new float[problemSize];
            cuda.CopyDeviceToHost(ptr, ki);

        }

        private float FindStoppingGradVal()
        {
            throw new NotImplementedException();
        }

        private void InitCudaModule()
        {
            cuda = new CUDA(0, true);
            var cuCtx = cuda.CreateContext(0, CUCtxFlags.MapHost);
            cuda.SetCurrentContext(cuCtx);

            string modluePath = Path.Combine(Environment.CurrentDirectory, cudaModuleName);
            if (!File.Exists(modluePath))
                throw new ArgumentException("Failed access to cuda module" + modluePath);

            cuModule = cuda.LoadModule(modluePath);
            cuFuncFindMaxI = cuda.GetModuleFunction(funcFindMaxI);
            cuFuncFindMinJ = cuda.GetModuleFunction(funcFindMinJ);
            cuFuncFindStopping = cuda.GetModuleFunction(funcFindStoping);
            cuFuncUpdateG = cuda.GetModuleFunction(funcUpdateGFunc);


        }




        
    }
}