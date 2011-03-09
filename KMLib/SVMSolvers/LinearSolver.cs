using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using System.Diagnostics;

namespace KMLib.SVMSolvers
{

    /// <summary>
    /// Solver for linear SVM based on LIBLINEAR package
    /// http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    /// 
    /// Paper: "A Dual Coordinate Descent Method for Large-scale Linear SVM" Hsieh et al., ICML 2008
    /// 
    /// 
    /// </summary>
    public class LinearSolver : Solver<SparseVec>
    {
        public float bias=-1;

        /// <summary>
        /// contains labels which has different weight
        /// </summary>
        int[] labelWithWeight;

        /// <summary>
        /// contains weights for labels <see cref="labelWithWeight"/> array
        /// </summary>
        /// <remarks>
        /// 
        /// </remarks>
        double[] penaltyWeights;


        private SolverType solverType = SolverType.L2R_L2LOSS_SVC_DUAL;
        private double epsilon= 0.01;

        enum SolverType
        {
            /// <summary>
            ///L2-regularized L2-loss support vector classification (dual) (fka L2LOSS_SVM_DUAL)
            /// </summary>
            L2R_L2LOSS_SVC_DUAL,

            /// <summary>
            /// L2-regularized L2-loss support vector classification (primal)(fka L2LOSS_SVM)
            /// </summary>
            //L2R_L2LOSS_SVC,

            /// <summary>
            /// L2-regularized L1-loss support vector classification (dual) (fka L1LOSS_SVM_DUAL)
            /// </summary>
            L2R_L1LOSS_SVC_DUAL,


            /// <summary>
            /// L1-regularized L2-loss support vector classification (primal)
            /// </summary>
            //L1R_L2LOSS_SVC,
            MCSVM_CS,
        }




        /// <summary>
        /// Construct linear solver
        /// </summary>
        /// <param name="problem">trainning problem</param>
        /// <param name="C">penalty parameter</param>
        public LinearSolver(Problem<SparseVec> problem, float C)
            : base(problem, C)
        {
        }

        public LinearSolver(Problem<SparseVec> problem, float C,int[] weightedLabels,double[] weights):base(problem,C)
        {
            labelWithWeight = weightedLabels;
            penaltyWeights = weights;
        }

        public override Model<SparseVec> ComputeModel()
        {
            //throw new NotImplementedException();

            //check if features are in correct order
            //for (FeatureNode[] nodes : prob.x) {
            //    int indexBefore = 0;
            //    for (FeatureNode n : nodes) {
            //        if (n.index <= indexBefore) {
            //            throw new IllegalArgumentException("feature nodes must be sorted by index in ascending order");
            //        }
            //        indexBefore = n.index;
            //    }
            //}

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

            //    if (Bias >= 0)
            //    model.nr_feature = n - 1;
            //else
            //    model.nr_feature = n;


            //model.solverType = param.solverType;
            //model.bias = prob.bias;
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

            //todo: add inf about labels to Model
            //model class
            model.Labels = new float[nr_class];
            for (int i = 0; i < nr_class; i++)
                model.Labels[i] =(float) label[i];

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
            //FeatureNode[][] x = new FeatureNode[l][];
            Debug.Assert(l == problem.ElementsCount);
            for (int i = 0; i < l; i++)
            {
                //x[i] = prob.x[perm[i]];
                permVec[i] = problem.Elements[perm[i]];
            }


            Problem<SparseVec> sub_prob = new Problem<SparseVec>();
            sub_prob.ElementsCount = l;
            sub_prob.FeaturesCount = n;
            //we set labels below
            sub_prob.Y = new float[sub_prob.ElementsCount];
            sub_prob.Elements = permVec;
            //java version
            //for (int k = 0; k < sub_prob.ElementsCount; k++)
            //    sub_prob.x[k] = x[k];

            // multi-class svm by Crammer and Singer
            //for now not used
            if (solverType == SolverType.MCSVM_CS)
            {
                //model.w = new double[n * nr_class];
                model.W = new double[n * nr_class];
                for (int i = 0; i < nr_class; i++)
                {
                    for (j = start[i]; j < start[i] + count[i]; j++)
                    {
                        sub_prob.Y[j] = i;
                    }
                }

                //SolverMCSVM_CS solver = new SolverMCSVM_CS(sub_prob, nr_class, weighted_C, param.eps);
                //solver.solve(model.w);
            }
            else
            {
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
                    solve_l2r_l1l2_svc(model.W, epsilon, weighted_C[0], weighted_C[1], solverType);
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
                        solve_l2r_l1l2_svc(w, epsilon, weighted_C[0], C, solverType);

                        for (j = 0; j < n; j++)
                            model.W[j * nr_class + i] = w[j];
                    }
                }

            }
            return model;
        }

        private void SetClassWeights(int nr_class, int[] label, double[] weighted_C)
        {

            if (labelWithWeight == null || labelWithWeight.Length == 1)
                return;

            int j;
            for (int i = 0; i < labelWithWeight.Length; i++)
            {
                for (j = 0; j < nr_class; j++)
                    if (labelWithWeight[i] == label[j])
                        break;
                if (j == nr_class)
                    throw new ArgumentOutOfRangeException("class label " + labelWithWeight[i] + " specified in weight is not found");

                weighted_C[j] *= penaltyWeights[i];
            }
        }

        private void groupClasses(Problem<SparseVec> problem, out int nr_class, out int[] label, out int[] start, out int[] count, int[] perm)
        {
            int l = problem.ElementsCount; //prob.l;
            int max_nr_class = 16;
            nr_class = 0;

            label = new int[max_nr_class];
            count = new int[max_nr_class];
            int[] data_label = new int[l];
            int i;

            for (i = 0; i < l; i++)
            {
                int this_label = (int)problem.Y[i];//prob.y[i];
                int j;
                for (j = 0; j < nr_class; j++)
                {
                    if (this_label == label[j])
                    {
                        ++count[j];
                        break;
                    }
                }
                data_label[i] = j;
                if (j == nr_class)
                {
                    if (nr_class == max_nr_class)
                    {
                        max_nr_class *= 2;
                        label = label.CopyToNewArray(max_nr_class);// copyToNewArray(label, max_nr_class);
                        count = count.CopyToNewArray(max_nr_class);// copyToNewArray(count, max_nr_class);
                    }
                    label[nr_class] = this_label;
                    count[nr_class] = 1;
                    ++nr_class;
                }
            }

            start = new int[nr_class];
            start[0] = 0;
            for (i = 1; i < nr_class; i++)
                start[i] = start[i - 1] + count[i - 1];
            for (i = 0; i < l; i++)
            {
                perm[start[data_label[i]]] = i;
                ++start[data_label[i]];
            }
            start[0] = 0;
            for (i = 1; i < nr_class; i++)
                start[i] = start[i - 1] + count[i - 1];

            // return new GroupClassesReturn(nr_class, label, start, count);
        }





        /// <summary>
        /// this method corresponds to the following define in the C version:
        /// #define GETI(i) (y[i]+1)
        /// </summary>
        /// <remarks>
        /// Its only a helper for getting 0- if yi=-1 and 2- ifyi=1,
        /// 0 or 2 are indexes in 3-dim array containing penalty for positive and negative
        /// 
        /// To support weights for instances, use GETI(i) (i)
        /// </remarks>
        /// <param name="y"></param>
        /// <param name="i"></param>
        /// <returns></returns>
        private static int GETI(sbyte[] y, int i)
        {
            return y[i] + 1;
        }

        /// <summary>
        /// A coordinate descent algorithm for L1-loss and L2-loss SVM dual problems
        /// </summary>
        /// <remarks>
        ///   min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
        ///     s.t.      0 <= alpha_i <= upper_bound_i,
        /// 
        ///   where Qij = yi yj xi^T xj and
        ///   D is a diagonal matrix
        /// 
        ///  In L1-SVM case:
        ///      upper_bound_i = Cp if y_i = 1
        ///       upper_bound_i = Cn if y_i = -1
        ///       D_ii = 0
        ///  In L2-SVM case:
        ///       upper_bound_i = INF
        ///       D_ii = 1/(2*Cp) if y_i = 1
        ///       D_ii = 1/(2*Cn) if y_i = -1
        /// 
        ///  Given:
        ///  x, y, Cp, Cn
        ///  eps is the stopping tolerance
        /// 
        ///  solution will be put in w
        /// 
        ///  See Algorithm 3 of Hsieh et al., ICML 2008
        /// </remarks>

        /// <param name="w"></param>
        /// <param name="eps">epsilon solution accuracy</param>
        /// <param name="Cp">penalty for positive elements</param>
        /// <param name="Cn">penalty for negative elements</param>
        /// <param name="solver_type"></param>
        private void solve_l2r_l1l2_svc(double[] w, double eps, double Cp, double Cn, SolverType solver_type)
        {


            Random rand = new Random();

            int l = problem.ElementsCount;// prob.l;
            int w_size = problem.FeaturesCount;// prob.n;
            int i, s, iter = 0;
            double C, d, G;

            //diagonal cache QD=Qii+diag (for different  formulation L1 or L2 diag is different)
            double[] QD = new double[l];
            int max_iter = 1000;
            int[] index = new int[l];
            double[] alpha = new double[l];
            sbyte[] y = new sbyte[l];
            int active_size = l;


            // PG: projected gradient, for shrinking and stopping
            double PG;
            double PGmax_old = Double.PositiveInfinity;
            double PGmin_old = Double.NegativeInfinity;
            double PGmax_new, PGmin_new;

            // default solver_type: L2R_L2LOSS_SVC_DUAL
            double[] diag = new double[] { 0.5 / Cn, 0, 0.5 / Cp };

            double[] upper_bound = new double[] { Double.PositiveInfinity, 0, Double.PositiveInfinity };

            //for different svm formulation diag and upper bound is different
            if (solver_type == SolverType.L2R_L1LOSS_SVC_DUAL)
            {
                diag[0] = 0;
                diag[2] = 0;
                upper_bound[0] = Cn;
                upper_bound[2] = Cp;
            }

            #region some initialization


            for (i = 0; i < w_size; i++)
                w[i] = 0;

            for (i = 0; i < l; i++)
            {
                alpha[i] = 0;
                if (problem.Y[i] > 0)
                {
                    y[i] = +1;
                }
                else
                {
                    y[i] = -1;
                }

                QD[i] = diag[GETI(y, i)];

                //for (FeatureNode xi : prob.x[i]) {
                //    QD[i] += xi.value * xi.value;
                //}
                QD[i] += problem.Elements[i].DotProduct();


                index[i] = i;
            }
            #endregion
            
            while (iter < max_iter)
            {
                PGmax_new = Double.NegativeInfinity;
                PGmin_new = Double.PositiveInfinity;

                for (i = 0; i < active_size; i++)
                {
                    int j = i + rand.Next(active_size - i);// .nextInt(active_size - i);

                    //swap(index, i, j);
                    index.SwapIndex(i, j);
                }

                for (s = 0; s < active_size; s++)
                {
                    i = index[s];
                    G = 0;
                    sbyte yi = y[i];

                    //for (FeatureNode xi : prob.x[i]) {
                    //    G += w[xi.index - 1] * xi.value;
                    //}
                    //above changed into:
                    var element = problem.Elements[i];
                    for (int k = 0; k < element.Count; k++)
                    {
                        G += w[element.Indices[k] - 1] * element.Values[k];

                    }

                    G = G * yi - 1;

                    C = upper_bound[GETI(y, i)];
                    G += alpha[i] * diag[GETI(y, i)];

                    PG = 0;
                    if (alpha[i] == 0)
                    {
                        if (G > PGmax_old)
                        {
                            active_size--;
                            //swap(index, s, active_size);
                            index.SwapIndex(s, active_size);
                            s--;
                            continue;
                        }
                        else if (G < 0)
                        {
                            PG = G;
                        }
                    }
                    else if (alpha[i] == C)
                    {
                        if (G < PGmin_old)
                        {
                            active_size--;
                            //swap(index, s, active_size);
                            index.SwapIndex(s, active_size);
                            s--;
                            continue;
                        }
                        else if (G > 0)
                        {
                            PG = G;
                        }
                    }
                    else
                    {
                        PG = G;
                    }

                    PGmax_new = Math.Max(PGmax_new, PG);
                    PGmin_new = Math.Min(PGmin_new, PG);

                    if (Math.Abs(PG) > 1.0e-12)
                    {
                        double alpha_old = alpha[i];
                        alpha[i] = Math.Min(Math.Max(alpha[i] - G / QD[i], 0.0), C);

                        //update vector "w"
                        d = (alpha[i] - alpha_old) * yi;
                        var spVec = problem.Elements[i];
                        for (int k = 0; k < spVec.Count; k++)
                        {
                            w[spVec.Indices[k] - 1] += d * spVec.Values[k];
                        }
                        //above 
                        //for (FeatureNode xi : prob.x[i]) {
                        //    w[xi.index - 1] += d * xi.value;
                        //}
                    }
                }

                iter++;
                //if (iter % 10 == 0) info(".");

                if (PGmax_new - PGmin_new <= eps)
                {
                    if (active_size == l)
                        break;
                    else
                    {
                        active_size = l;
                        //  info("*");
                        PGmax_old = Double.PositiveInfinity;
                        PGmin_old = Double.NegativeInfinity;
                        continue;
                    }
                }
                PGmax_old = PGmax_new;
                PGmin_old = PGmin_new;
                if (PGmax_old <= 0) PGmax_old = Double.PositiveInfinity;
                if (PGmin_old >= 0) PGmin_old = Double.NegativeInfinity;
            }

            //info(NL + "optimization finished, #iter = %d" + NL, iter);
            Debug.WriteLine("optimization finished, #iter = {0}", iter);
            if (iter >= max_iter)
            {
                Debug.WriteLine("%nWARNING: reaching max number of iterations%nUsing -s 2 may be faster (also see FAQ)%n%n");
            }

            // calculate objective value

            double v = 0;
            int nSV = 0;
            for (i = 0; i < w_size; i++)
                v += w[i] * w[i];
            for (i = 0; i < l; i++)
            {
                v += alpha[i] * (alpha[i] * diag[GETI(y, i)] - 2);
                if (alpha[i] > 0) ++nSV;
            }
            Debug.WriteLine("Objective value = {0}", v / 2);
            Debug.WriteLine("nSV = {0}", nSV);
        }

    }
}
