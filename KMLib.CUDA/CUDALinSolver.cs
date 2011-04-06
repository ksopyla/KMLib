using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using KMLib.SVMSolvers;
using System.Diagnostics;
using GASS.CUDA;
using GASS.CUDA.Types;

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
        /// cuda device pointer for output deltas
        /// </summary>
        protected CUdeviceptr deltasPtr;

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

            if (nr_class == 2)
            {
                model.W = new double[w_size];

                int e0 = start[0] + count[0];
                int k = 0;
                for (; k < e0; k++)
                    sub_prob.Y[k] = +1;
                for (; k < sub_prob.ElementsCount; k++)
                    sub_prob.Y[k] = -1;

                //train_one(sub_prob, param, model.w, weighted_C[0], weighted_C[1]);

                solve_l2r_l2_svc_cuda(model.W, epsilon, weighted_C[0], weighted_C[1]);
                //solve_l2r_l1l2_svc(model.W, epsilon, weighted_C[0], weighted_C[1], solverType);
            }
            else
            {
                model.W = new double[w_size * nr_class];
                double[] w = new double[w_size];

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

                    //train_one(sub_prob, param, w, weighted_C[i], param.C);
                    solve_l2r_l2_svc_cuda(w, epsilon, weighted_C[0], C);

                    for (j = 0; j < n; j++)
                        model.W[j * nr_class + i] = w[j];
                }
            }


            return model;
        }

        private void solve_l2r_l2_svc_cuda(double[] p, double epsilon, double p_2, double p_3)
        {
            throw new NotImplementedException();
        }


    }
}
