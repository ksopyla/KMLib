using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using KMLib.Kernels;
using KMLib.SVMSolvers;
using GASS.CUDA;
using System.IO;

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

       

        #region variables from LibSVM
               protected sbyte[] y;
        protected float[] G;		// gradient of objective function
        
       
        private float[] alpha;
       
        protected float EPS = 0.001f;
        private float Cp, Cn;
        
        //private float[] p; == minus_once??
        private int[] active_set;
        private float[] G_bar;		// gradient, if we treat free variables as 0

        protected bool unshrink;	// XXX


        private float[] QD;
        
        protected const float INF = float.PositiveInfinity;
        #endregion


        private int problemSize;
        private CUDA cuda;
        private GASS.CUDA.Types.CUmodule cuModule;
        private GASS.CUDA.Types.CUfunction cuFuncFindMaxI;
        private string funcFindMaxI;
        private GASS.CUDA.Types.CUfunction cuFuncFindMinJ;
        private string funcFindMinJ;
        private GASS.CUDA.Types.CUfunction cuFuncFindStopping;
        private string funcFindStoping;
        private GASS.CUDA.Types.CUfunction cuFuncUpdateG;
        private string funcUpdateGFunc;



        public GPUSmoFanSolver(Problem<SparseVec> problem, IKernel<SparseVec> kernel, float C)
            : base(problem, kernel, C)
        {
            this.C = C;
            problemSize = problem.ElementsCount;
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
                G = new float[problemSize];
               
                int i;
                for (i = 0; i < problemSize; i++)
                {
                    G[i] = -1;
                }
                
            }


            InitCudaModule();


            // optimization step
            float GMaxI;
            float GMaxJ;
            int iter = 1;
          

            while (true)
            {

                //Find i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
                Tuple<int,float> maxPair =FindMaxPair();
                int i=maxPair.Item1;
                GMaxI=maxPair.Item2;

                if (i % 255 ==0)
                {
                    GMaxJ = FindStoppingGradVal();
                }

                //Compute i-th kernel collumn, set the specific memory region on GPU, 
                ComputeKernel(i);


                // j: mimimizes the decrease of obj value
            //    (if quadratic coefficeint <= 0, replace it with tau)
            //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
               int j= FindMinPair(i,GMaxI);
               



               //Compute j-th kernel collumn
               ComputeKernel(j);
                

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
                    v += alpha[i] * (G[i] -1);

                si.obj = v / 2;
            }

            
            si.upper_bound_p = Cp;
            si.upper_bound_n = Cn;
        }

        private int FindMinPair(int i, float GmaxI)
        {
            throw new NotImplementedException();
        }

        private void ComputeKernel(int i)
        {
            
            


        }

        private Tuple<int, float> FindMaxPair()
        {
            throw new NotImplementedException();
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




        public string cudaModuleName { get; set; }
    }
}