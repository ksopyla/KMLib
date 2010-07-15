using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using KMLib.Kernels;

namespace KMLib.Helpers
{
    /// <summary>
    ///  An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918, implementation based on 
    /// SVM.net (http://matthewajohnson.org/software/svm.html) and original LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    ///  Solves:
    /// Min 0.5(\alpha^T Q \alpha) + p^T \alpha
    /// y^T \alpha = \delta
    /// y_i = +1 or -1
    /// 0 <= alpha_i <= Cp for y_i = 1
    /// 0 <= alpha_i <= Cn for y_i = -1
    /// solution will be put in \alpha, objective value will be put in obj
    /// </summary>
    /// <typeparam name="TProblemElement">Problem elements</typeparam>
    public class SmoFanSolver<TProblemElement> : Solver<TProblemElement>
    {

        /// <summary>
        /// Internal helper class, whitch store computed solution
        /// </summary>
        internal class SolutionInfo
        {
            /// <summary>
            /// objective function value
            /// </summary>
            public float obj;
            /// <summary>
            /// rho == b prameter in function
            /// </summary>
            public float rho;
            public float upper_bound_p;
            public float upper_bound_n;
            public float r;	// for Solver_NU
        }

        #region variables from LibSVM
        protected int active_size;
        protected sbyte[] y;
        protected float[] G;		// gradient of objective function
        private const byte LOWER_BOUND = 0;
        private const byte UPPER_BOUND = 1;
        private const byte FREE = 2;
        private byte[] alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
        private float[] alpha;
        //protected IQMatrix Q;
        //protected float[] QD;
        protected float EPS = 0.001f;
        private float Cp, Cn;
        private float[] p;
        private int[] active_set;
        private float[] G_bar;		// gradient, if we treat free variables as 0
       
        protected bool unshrink;	// XXX

        
        private float[] QD;
        private bool Shrinking;
        protected const float INF = float.PositiveInfinity;
        #endregion
       
        /// <summary>
        /// Cached kernel
        /// </summary>
        private CachedKernel<TProblemElement> Q;
        private int problemSize;
        

        public SmoFanSolver(Problem<TProblemElement> problem, IKernel<TProblemElement> kernel, float C)
            : base(problem, kernel, C)
        {

            //remeber that we have variable kernel in base class
            Q = new CachedKernel<TProblemElement>(problem,kernel);
            //  (CachedKernel)kernel;

            QD = Q.GetQD();

            Shrinking = false;
            //todo: change it, add array to base class with different penalty for labels
            Cp = C;
            Cn = C;
            problemSize = problem.ElementsCount;
        }


        /// <summary>
        /// Computes model by solving optimization problem
        /// </summary>
        /// <returns>Model</returns>
        public override Model<TProblemElement> ComputeModel()
        {
            int problemSize = problem.ElementsCount;
            float[] Minus_ones = new float[problemSize];
            sbyte[] y = new sbyte[problemSize];

            float[] alphaResult = new float[problem.ElementsCount];



            for (int i = 0; i < problemSize; i++)
            {
                alphaResult[i] = 0;
                Minus_ones[i] = -1;
                if (problem.Labels[i] > 0) y[i] = +1;
                else y[i] = -1;
            }

            SolutionInfo si = new SolutionInfo();
            Solve(Minus_ones, y, alphaResult, si, Shrinking);


            //for (int i = 0; i < problemSize; i++)
            //{
            //    alphaResult[i] *= y[i];
            //}

            Model<TProblemElement> model = new Model<TProblemElement>();
            model.NumberOfClasses = 2;
            model.Alpha = alphaResult;
            model.Rho = si.rho;


            List<TProblemElement> supportElements = new List<TProblemElement>();
            List<int> suporrtIndexes = new List<int>(alpha.Length);
            for (int j = 0; j < alphaResult.Length; j++)
            {
                if (Math.Abs(alphaResult[j]) > 0)
                {
                    supportElements.Add(problem.Elements[j]);
                    suporrtIndexes.Add(j);
                }

            }
            model.SupportElements = supportElements.ToArray();
            model.SupportElementsIndexes = suporrtIndexes.ToArray();

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
        private void Solve(float[] minusOnes, sbyte[] y_, float[] alpha_, SolutionInfo si, bool shrinking)
        {
            //this.l = l;
            //this.Q = Q;

            p = (float[])minusOnes.Clone();
            y = (sbyte[])y_.Clone();
            alpha = (float[])alpha_.Clone();
            //this.Cp = Cp;
            //this.Cn = Cn;

            this.unshrink = false;



            // initialize alpha_status
            {
                alpha_status = new byte[problemSize];
                for (int i = 0; i < problemSize; i++)
                    update_alpha_status(i);
            }

            // initialize active set (for shrinking)
            {
                active_set = new int[problemSize];
                for (int i = 0; i < problemSize; i++)
                    active_set[i] = i;
                active_size = problemSize;
            }

            // initialize gradient
            {
                G = new float[problemSize];
                G_bar = new float[problemSize];
                int i;
                for (i = 0; i < problemSize; i++)
                {
                    G[i] = p[i];
                    G_bar[i] = 0;
                }
                for (i = 0; i < problemSize; i++)
                    if (!is_lower_bound(i))
                    {
                        float[] Q_i = Q.GetQ(i, problemSize);
                        float alpha_i = alpha[i];
                        int j;
                        for (j = 0; j < problemSize; j++)
                            G[j] += alpha_i * Q_i[j];
                        if (is_upper_bound(i))
                            for (j = 0; j < problemSize; j++)
                                G_bar[j] += get_C(i) * Q_i[j];
                    }
            }

            // optimization step

            int iter = 0;
            int counter = Math.Min(problemSize, 1000) + 1;
            int[] working_set = new int[2];

            while (true)
            {
                // show progress and do shrinking

                if (--counter == 0)
                {
                    counter = Math.Min(problemSize, 1000);
                    if (shrinking) do_shrinking();
                    //Procedures.info(".");
                }

                if (select_working_set(working_set) != 0)
                {
                    // reconstruct the whole gradient
                    reconstruct_gradient();
                    // reset active set size and check
                    active_size = problemSize;
                    // Procedures.info("*");
                    if (select_working_set(working_set) != 0)
                        break;
                    else
                        counter = 1;	// do shrinking next iteration
                }

                int i = working_set[0];
                int j = working_set[1];

                ++iter;

                // update alpha[i] and alpha[j], handle bounds carefully

                float[] Q_i = Q.GetQ(i, active_size);
                float[] Q_j = Q.GetQ(j, active_size);

                float C_i = get_C(i);
                float C_j = get_C(j);

                float old_alpha_i = alpha[i];
                float old_alpha_j = alpha[j];

                if (y[i] != y[j])
                {
                    float quad_coef = Q_i[i] + Q_j[j] + 2 * Q_i[j];
                    if (quad_coef <= 0)
                        quad_coef = 1e-12f;
                    float delta = (-G[i] - G[j]) / quad_coef;
                    float diff = alpha[i] - alpha[j];
                    alpha[i] += delta;
                    alpha[j] += delta;

                    if (diff > 0)
                    {
                        if (alpha[j] < 0)
                        {
                            alpha[j] = 0;
                            alpha[i] = diff;
                        }
                    }
                    else
                    {
                        if (alpha[i] < 0)
                        {
                            alpha[i] = 0;
                            alpha[j] = -diff;
                        }
                    }
                    if (diff > C_i - C_j)
                    {
                        if (alpha[i] > C_i)
                        {
                            alpha[i] = C_i;
                            alpha[j] = C_i - diff;
                        }
                    }
                    else
                    {
                        if (alpha[j] > C_j)
                        {
                            alpha[j] = C_j;
                            alpha[i] = C_j + diff;
                        }
                    }
                }
                else
                {
                    float quad_coef = Q_i[i] + Q_j[j] - 2 * Q_i[j];
                    if (quad_coef <= 0)
                        quad_coef = 1e-12f;
                    float delta = (G[i] - G[j]) / quad_coef;
                    float sum = alpha[i] + alpha[j];
                    alpha[i] -= delta;
                    alpha[j] += delta;

                    if (sum > C_i)
                    {
                        if (alpha[i] > C_i)
                        {
                            alpha[i] = C_i;
                            alpha[j] = sum - C_i;
                        }
                    }
                    else
                    {
                        if (alpha[j] < 0)
                        {
                            alpha[j] = 0;
                            alpha[i] = sum;
                        }
                    }
                    if (sum > C_j)
                    {
                        if (alpha[j] > C_j)
                        {
                            alpha[j] = C_j;
                            alpha[i] = sum - C_j;
                        }
                    }
                    else
                    {
                        if (alpha[i] < 0)
                        {
                            alpha[i] = 0;
                            alpha[j] = sum;
                        }
                    }
                }

                // update G

                float delta_alpha_i = alpha[i] - old_alpha_i;
                float delta_alpha_j = alpha[j] - old_alpha_j;

                for (int k = 0; k < active_size; k++)
                {
                    G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
                }

                // update alpha_status and G_bar

                {
                    bool ui = is_upper_bound(i);
                    bool uj = is_upper_bound(j);
                    update_alpha_status(i);
                    update_alpha_status(j);
                    int k;
                    if (ui != is_upper_bound(i))
                    {
                        Q_i = Q.GetQ(i, problemSize);
                        if (ui)
                            for (k = 0; k < problemSize; k++)
                                G_bar[k] -= C_i * Q_i[k];
                        else
                            for (k = 0; k < problemSize; k++)
                                G_bar[k] += C_i * Q_i[k];
                    }

                    if (uj != is_upper_bound(j))
                    {
                        Q_j = Q.GetQ(j, problemSize);
                        if (uj)
                            for (k = 0; k < problemSize; k++)
                                G_bar[k] -= C_j * Q_j[k];
                        else
                            for (k = 0; k < problemSize; k++)
                                G_bar[k] += C_j * Q_j[k];
                    }
                }

            }

            // calculate rho

            si.rho = calculate_rho();

            // calculate objective value
            {
                float v = 0;
                int i;
                for (i = 0; i < problemSize; i++)
                    v += alpha[i] * (G[i] + p[i]);

                si.obj = v / 2;
            }

            // put back the solution
            {
                for (int i = 0; i < problemSize; i++)
                    alpha_[active_set[i]] = alpha[i];
            }

            si.upper_bound_p = Cp;
            si.upper_bound_n = Cn;

            // Procedures.info("\noptimization finished, #iter = " + iter + "\n");
        }


        private float get_C(int i)
        {
            return (y[i] > 0) ? Cp : Cn;
        }

        private void update_alpha_status(int i)
        {
            if (alpha[i] >= get_C(i))
                alpha_status[i] = UPPER_BOUND;
            else if (alpha[i] <= 0)
                alpha_status[i] = LOWER_BOUND;
            else alpha_status[i] = FREE;
        }

        protected bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
        protected bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }

        private bool is_free(int i) { return alpha_status[i] == FREE; }



        protected void swap_index(int i, int j)
        {
            Q.SwapIndex(i, j);
            y.SwapIndex(i, j);
            G.SwapIndex(i, j);
            alpha_status.SwapIndex(i, j);
            alpha.SwapIndex(i, j);
            p.SwapIndex(i, j);
            active_set.SwapIndex(i, j);
            G_bar.SwapIndex(i, j);
        }

        protected void reconstruct_gradient()
        {
            // reconstruct inactive elements of G from G_bar and free variables

            if (active_size == problemSize) return;

            int i, j;
            int nr_free = 0;

            for (j = active_size; j < problemSize; j++)
                G[j] = G_bar[j] + p[j];

            for (j = 0; j < active_size; j++)
                if (is_free(j))
                    nr_free++;

            /*
            if (2 * nr_free < active_size)
                Procedures.info("\nWarning: using -h 0 may be faster\n");
            */

            if (nr_free * problemSize > 2 * active_size * (problemSize - active_size))
            {
                for (i = active_size; i < problemSize; i++)
                {
                    float[] Q_i = Q.GetQ(i, active_size);
                    for (j = 0; j < active_size; j++)
                        if (is_free(j))
                            G[i] += alpha[j] * Q_i[j];
                }
            }
            else
            {
                for (i = 0; i < active_size; i++)
                    if (is_free(i))
                    {
                        float[] Q_i = Q.GetQ(i, problemSize);
                        float alpha_i = alpha[i];
                        for (j = active_size; j < problemSize; j++)
                            G[j] += alpha_i * Q_i[j];
                    }
            }
        }


        // return 1 if already optimal, return 0 otherwise
        int select_working_set(int[] working_set)
        {
            // return i,j such that
            // i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
            // j: mimimizes the decrease of obj value
            //    (if quadratic coefficeint <= 0, replace it with tau)
            //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

            float GMax = -INF;
            float GMax2 = -INF;
            int GMax_idx = -1;
            int GMin_idx = -1;
            float obj_diff_Min = INF;

            for (int t = 0; t < active_size; t++)
                if (y[t] == +1)
                {
                    if (!is_upper_bound(t))
                        if (-G[t] >= GMax)
                        {
                            GMax = -G[t];
                            GMax_idx = t;
                        }
                }
                else
                {
                    if (!is_lower_bound(t))
                        if (G[t] >= GMax)
                        {
                            GMax = G[t];
                            GMax_idx = t;
                        }
                }

            int i = GMax_idx;
            float[] Q_i = null;
            if (i != -1) // null Q_i not accessed: GMax=-INF if i=-1
                Q_i = Q.GetQ(i, active_size);

            for (int j = 0; j < active_size; j++)
            {
                if (y[j] == +1)
                {
                    if (!is_lower_bound(j))
                    {
                        float grad_diff = GMax + G[j];
                        if (G[j] >= GMax2)
                            GMax2 = G[j];
                        if (grad_diff > 0)
                        {
                            float obj_diff;
                            float quad_coef = (float)(Q_i[i] + QD[j] - 2.0 * y[i] * Q_i[j]);
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = (float)(-(grad_diff * grad_diff) / 1e-12);

                            if (obj_diff <= obj_diff_Min)
                            {
                                GMin_idx = j;
                                obj_diff_Min = obj_diff;
                            }
                        }
                    }
                }
                else
                {
                    if (!is_upper_bound(j))
                    {
                        float grad_diff = GMax - G[j];
                        if (-G[j] >= GMax2)
                            GMax2 = -G[j];
                        if (grad_diff > 0)
                        {
                            float obj_diff;
                            float quad_coef = (float)(Q_i[i] + QD[j] + 2.0 * y[i] * Q_i[j]);
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = (float)(-(grad_diff * grad_diff) / 1e-12);

                            if (obj_diff <= obj_diff_Min)
                            {
                                GMin_idx = j;
                                obj_diff_Min = obj_diff;
                            }
                        }
                    }
                }
            }

            if (GMax + GMax2 < EPS)
                return 1;

            working_set[0] = GMax_idx;
            working_set[1] = GMin_idx;
            return 0;
        }

        private bool be_shrunk(int i, float GMax1, float GMax2)
        {
            if (is_upper_bound(i))
            {
                if (y[i] == +1)
                    return (-G[i] > GMax1);
                else
                    return (-G[i] > GMax2);
            }
            else if (is_lower_bound(i))
            {
                if (y[i] == +1)
                    return (G[i] > GMax2);
                else
                    return (G[i] > GMax1);
            }
            else
                return (false);
        }

        void do_shrinking()
        {
            int i;
            float GMax1 = -INF;		// Max { -y_i * grad(f)_i | i in I_up(\alpha) }
            float GMax2 = -INF;		// Max { y_i * grad(f)_i | i in I_low(\alpha) }

            // find Maximal violating pair first
            for (i = 0; i < active_size; i++)
            {
                if (y[i] == +1)
                {
                    if (!is_upper_bound(i))
                    {
                        if (-G[i] >= GMax1)
                            GMax1 = -G[i];
                    }
                    if (!is_lower_bound(i))
                    {
                        if (G[i] >= GMax2)
                            GMax2 = G[i];
                    }
                }
                else
                {
                    if (!is_upper_bound(i))
                    {
                        if (-G[i] >= GMax2)
                            GMax2 = -G[i];
                    }
                    if (!is_lower_bound(i))
                    {
                        if (G[i] >= GMax1)
                            GMax1 = G[i];
                    }
                }
            }

            if (unshrink == false && GMax1 + GMax2 <= EPS * 10)
            {
                unshrink = true;
                reconstruct_gradient();
                active_size = problemSize;
            }

            for (i = 0; i < active_size; i++)
                if (be_shrunk(i, GMax1, GMax2))
                {
                    active_size--;
                    while (active_size > i)
                    {
                        if (!be_shrunk(active_size, GMax1, GMax2))
                        {
                            swap_index(i, active_size);
                            break;
                        }
                        active_size--;
                    }
                }
        }

        float calculate_rho()
        {
            float r;
            int nr_free = 0;
            float ub = INF, lb = -INF, sum_free = 0;
            for (int i = 0; i < active_size; i++)
            {
                float yG = y[i] * G[i];

                if (is_lower_bound(i))
                {
                    if (y[i] > 0)
                        ub = Math.Min(ub, yG);
                    else
                        lb = Math.Max(lb, yG);
                }
                else if (is_upper_bound(i))
                {
                    if (y[i] < 0)
                        ub = Math.Min(ub, yG);
                    else
                        lb = Math.Max(lb, yG);
                }
                else
                {
                    ++nr_free;
                    sum_free += yG;
                }
            }

            if (nr_free > 0)
                r = sum_free / nr_free;
            else
                r = (ub + lb) / 2;

            return r;
        }


        
    }
}