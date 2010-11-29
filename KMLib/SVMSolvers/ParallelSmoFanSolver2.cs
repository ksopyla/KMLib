using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using KMLib.Kernels;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Threading;

namespace KMLib.SVMSolvers
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
    /// <remarks>class use different thread utilization than <see cref="ParallelSmoFanSolver"/>
    /// it use ThreadPool, should be more efficeint than previous.
    /// Less memory allocation.
    /// </remarks>
    /// <typeparam name="TProblemElement">Problem elements</typeparam>
    public class ParallelSmoFanSolver2<TProblemElement> : Solver<TProblemElement>, IDisposable
    {

        /// <summary>
        /// Data passed to separete thread, for finding Max index 'i'
        /// </summary>
        internal class MaxFindingThreadData
        {


            public ManualResetEvent ResetEvent { get; set; }

            public Pair<int, float> Pair { get; set; }

            ///// <summary>
            ///// labels
            ///// </summary>
            //public sbyte[] Y { get; set; }

            ///// <summary>
            ///// gradient
            ///// </summary>
            //public float[] G { get; set; }

            /// <summary>
            /// Array range for processing
            /// </summary>
            public Tuple<int, int> Range { get; set; }
        }

        /// <summary>
        /// Data passed to separete thread, for finding min index 'j'
        /// </summary>
        internal class MinFindingThreadData
        {


            public ManualResetEvent ResetEvent { get; set; }

            /// <summary>
            /// min pair, min value and index
            /// </summary>
            public Pair<int, float> Pair { get; set; }

            public float GMax;

            public int GMaxIdx;

            public float[] Q_i;
            public float GMax2 = float.NegativeInfinity;

            /// <summary>
            /// Array range for processing
            /// </summary>
            public Tuple<int, int> Range { get; set; }
        }

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
        private OrderablePartitioner<Tuple<int, int>> partition;

        object lockObj = new object();

        #region thread objects
        /// <summary>
        /// for waiting on threads
        /// </summary>
        ManualResetEvent[] resetEvents;

        /// <summary>
        /// for finding max index "i" in svm solver
        /// </summary>
        Pair<int, float>[] maxPairs;

        /// <summary>
        /// for finding min index "j" in svm solver
        /// </summary
        Pair<int, float>[] minPairs;

        /// <summary>
        /// number of threads equal number of processors
        /// </summary>
        int numberOfThreads;

        /// <summary>
        /// Data for finding max Pair
        /// </summary>
        MaxFindingThreadData[] maxPairThreadsData;
        /// <summary>
        /// wait callback for finding maxPairs
        /// </summary>
        WaitCallback[] maxPairsWaitCallbacks;

        /// <summary>
        /// Data for finding min Pair
        /// </summary>
        MinFindingThreadData[] minPairThreadsData;
        /// <summary>
        /// wait callback for finding maxPairs
        /// </summary>
        WaitCallback[] minPairsWaitCallbacks;
        /// <summary>
        /// size of each problem chunk
        /// </summary>
        int rangeSize;

        #endregion

        public ParallelSmoFanSolver2(Problem<TProblemElement> problem, IKernel<TProblemElement> kernel, float C)
            : base(problem, kernel, C)
        {
            //todo: add checking if kernel is initialized

            //remeber that we have variable kernel in base class
            Q = new CachedKernel<TProblemElement>(problem, kernel);

            //get diagonal cache, kernel should compute that
            QD = Q.GetQD();

            //Shrinking = false;
            //todo: change it, add array to base class with different penalty for labels
            Cp = C;
            Cn = C;
            problemSize = problem.ElementsCount;

            numberOfThreads = Environment.ProcessorCount;

            rangeSize = (int)Math.Ceiling((problemSize + 0.0) / numberOfThreads);
            partition = Partitioner.Create(0, problemSize, rangeSize);
            resetEvents = new ManualResetEvent[numberOfThreads];
            //max data structures
            maxPairsWaitCallbacks = new WaitCallback[numberOfThreads];
            maxPairThreadsData = new MaxFindingThreadData[numberOfThreads];
            maxPairs = new Pair<int, float>[numberOfThreads];

            //min data structures
            minPairsWaitCallbacks = new WaitCallback[numberOfThreads];
            minPairThreadsData = new MinFindingThreadData[numberOfThreads];
            minPairs = new Pair<int, float>[numberOfThreads];

            int startRange = 0;
            int endRange = startRange + rangeSize;
            for (int i = 0; i < numberOfThreads; i++)
            {
                resetEvents[i] = new ManualResetEvent(false);
                maxPairs[i] = new Pair<int, float>(-1, float.NegativeInfinity);

                maxPairThreadsData[i] = new MaxFindingThreadData()
                {
                    ResetEvent = resetEvents[i],
                    Pair = maxPairs[i],
                    Range = new Tuple<int, int>(startRange, endRange)
                };

                maxPairsWaitCallbacks[i] = new WaitCallback(this.FindMaxPairInThread);

                minPairs[i] = new Pair<int, float>(-1, float.PositiveInfinity);
                minPairThreadsData[i] = new MinFindingThreadData()
                {
                    ResetEvent = resetEvents[i],
                    Pair = minPairs[i],
                    Range = new Tuple<int, int>(startRange, endRange)
                };

                minPairsWaitCallbacks[i] = new WaitCallback(this.FindMinPairInThread);


                //change the range
                startRange = endRange;
                int rangeSum = endRange + rangeSize;
                endRange = rangeSum < problemSize ? rangeSum : problemSize;


            }

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



            Model<TProblemElement> model = new Model<TProblemElement>();
            model.NumberOfClasses = 2;
            model.Alpha = alphaResult;
            model.Rho = si.rho;


            List<TProblemElement> supportElements = new List<TProblemElement>(alpha.Length);
            List<int> suporrtIndexes = new List<int>(alpha.Length);
            List<float> supportLabels = new List<float>(alpha.Length);
            for (int j = 0; j < alphaResult.Length; j++)
            {
                if (Math.Abs(alphaResult[j]) > 0)
                {
                    supportElements.Add(problem.Elements[j]);
                    suporrtIndexes.Add(j);
                    supportLabels.Add(problem.Labels[j]);
                }

            }
            model.SupportElements = supportElements.ToArray();
            model.SupportElementsIndexes = suporrtIndexes.ToArray();
            model.Labels = supportLabels.ToArray();
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

            #region initialization
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
            #endregion

            #region init data needef for thread processing
            //for (int i = 0; i < numberOfThreads; i++)
            //{
            //    maxPairThreadsData[i].Y = y;
            //    maxPairThreadsData[i].G = G;
            //}
            #endregion

            // optimization step
            int iter = 0;
            //int counter = Math.Min(problemSize, 1000) + 1;
            int[] working_set = new int[2];

            int processors = Environment.ProcessorCount;

            while (true)
            {
                if (select_working_set(working_set, processors) != 0)
                    break;

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



                //for (int k = 0; k < active_size; k++)
                //{
                //    G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
                //}


                //var partition = Partitioner.Create(0, active_size);

                Parallel.ForEach(partition, (range) =>
                {
                    int rangeEnd = range.Item2;
                    for (int k = range.Item1; k < rangeEnd; k++)
                    {
                        G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
                    }

                });

                //  Parallel.ForEach(partition, UpdateGradient);



                // update alpha_status and G_bar

                //{
                bool ui = is_upper_bound(i);
                bool uj = is_upper_bound(j);
                update_alpha_status(i);
                update_alpha_status(j);
                //    int k;
                //    if (ui != is_upper_bound(i))
                //    {
                //        Q_i = Q.GetQ(i, problemSize);
                //        if (ui)
                //            for (k = 0; k < problemSize; k++)
                //                G_bar[k] -= C_i * Q_i[k];
                //        else
                //            for (k = 0; k < problemSize; k++)
                //                G_bar[k] += C_i * Q_i[k];
                //    }

                //    if (uj != is_upper_bound(j))
                //    {
                //        Q_j = Q.GetQ(j, problemSize);
                //        if (uj)
                //            for (k = 0; k < problemSize; k++)
                //                G_bar[k] -= C_j * Q_j[k];
                //        else
                //            for (k = 0; k < problemSize; k++)
                //                G_bar[k] += C_j * Q_j[k];
                //    }
                //}

            }//end while

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

        

        // return 1 if already optimal, return 0 otherwise
        int select_working_set(int[] working_set, int pairsCount)
        {
            // return i,j such that
            // i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
            // j: mimimizes the decrease of obj value
            //    (if quadratic coefficeint <= 0, replace it with tau)
            //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

            float GMax = -INF;
            // float GNMax = -INF;

            float GMax2 = -INF;
            int GMax_idx = -1;
            int GMin_idx = -1;
            //float obj_diff_Min = INF;
            //float obj_diff_NMin = INF;
            
            #region find max i

            Pair<int, float> maxPair = FindMaxPair();

            
            GMax = maxPair.Second;
            GMax_idx = maxPair.First;

            #endregion

            int i = GMax_idx;
            float[] Q_i = null;
            if (i != -1) // null Q_i not accessed: GMax=-INF if i=-1
                Q_i = Q.GetQ(i, active_size);



            //find min
            Pair<int, float> minPair;
            GMax2 = FindMinPair(out minPair, i, GMax, Q_i);
            GMin_idx = minPair.First;

            if (GMax + GMax2 < EPS)
                return 1;

            working_set[0] = GMax_idx;
            working_set[1] = GMin_idx;
            return 0;
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

        /// <summary>
        /// finds Max index 'i' in svm solver
        /// </summary>
        /// <returns></returns>
        private Pair<int, float> FindMaxPair()
        {

            for (int i = 0; i < numberOfThreads; i++)
            {
                maxPairThreadsData[i].ResetEvent.Reset();
                maxPairThreadsData[i].Pair.First = -1;
                maxPairThreadsData[i].Pair.Second = float.NegativeInfinity;


                ThreadPool.QueueUserWorkItem(maxPairsWaitCallbacks[i], maxPairThreadsData[i]);
            }

            WaitHandle.WaitAll(resetEvents);

            Pair<int, float> maxPair = new Pair<int, float>(-1, float.NegativeInfinity);
            foreach (var item in maxPairs)
            {
                if (maxPair.Second < item.Second)
                {
                    maxPair.First = item.First;
                    maxPair.Second = item.Second;
                }
            }
            return maxPair;
        }

        /// <summary>
        /// find min index 'j'
        /// </summary>
        /// <param name="minPair"></param>
        /// <param name="GMaxIdx"></param>
        /// <param name="GMax"></param>
        /// <param name="Q_i"></param>
        /// <returns></returns>
        private float FindMinPair(out Pair<int, float> minPair, int GMaxIdx,float GMax,float[] Q_i)
        {

            for (int i = 0; i < numberOfThreads; i++)
            {
                minPairThreadsData[i].ResetEvent.Reset();
                minPairThreadsData[i].Pair.First = -1;
                minPairThreadsData[i].Pair.Second = float.PositiveInfinity;
                minPairThreadsData[i].GMaxIdx = GMaxIdx;
                minPairThreadsData[i].GMax = GMax;
                minPairThreadsData[i].Q_i = Q_i;


                ThreadPool.QueueUserWorkItem(minPairsWaitCallbacks[i], minPairThreadsData[i]);
            }

            WaitHandle.WaitAll(resetEvents);

            //find min pair
            minPair = new Pair<int, float>(-1, float.PositiveInfinity);

            foreach (var item in minPairs)
            {
                if (minPair.Second > item.Second)
                {
                    minPair.First = item.First;
                    minPair.Second = item.Second;
                }
            }

            //find GMax2
            float GMax2 = float.NegativeInfinity;
            for (int i = 0; i < numberOfThreads; i++)
            {
                if (GMax2 < minPairThreadsData[i].GMax2)
                    GMax2 = minPairThreadsData[i].GMax2;
            }
            return GMax2;
        }

        /// <summary>
        /// finds max 'i' in svm solver, its called in separate thread 
        /// and find it in specific range of array
        /// </summary>
        /// <param name="threadData"></param>
        private void FindMaxPairInThread(object threadData)
        {
            MaxFindingThreadData data = (MaxFindingThreadData)threadData;
            Pair<int, float> localMax = data.Pair;

            for (int t = data.Range.Item1; t < data.Range.Item2; t++)
            {

                if (y[t] == +1)
                {
                    if (!is_upper_bound(t))
                    {
                        if (-G[t] > localMax.Second) //wcześniej było większe lub równe
                        {
                            localMax.First = t;
                            localMax.Second = -G[t];
                        }
                    }
                }
                else
                {
                    if (!is_lower_bound(t))
                    {
                        if (G[t] > localMax.Second) //wcześniej było >=
                        {

                            localMax.First = t;
                            localMax.Second = G[t];
                        }
                    }
                }


            }
            //signal for thread that computation complete
            data.ResetEvent.Set();
        }


        /// <summary>
        /// finds min 'j' in svm solver, its called in separate thread 
        /// and find it in specific range of array
        /// </summary>
        /// <param name="threadData"></param>
        private void FindMinPairInThread(object threadData)
        {
            MinFindingThreadData data = (MinFindingThreadData)threadData;
            Pair<int, float> localMaxMin = data.Pair;

           
            float GMax2 = float.NegativeInfinity;

            int i = data.GMaxIdx;
            float obj_diff = 0;
            float quad_coef = 0;
            float grad_diff = 0;
            for (int j = data.Range.Item1; j < data.Range.Item2; j++)
            {
                if (y[j] == +1)
                {
                    if (!is_lower_bound(j))
                    {
                        grad_diff = data.GMax + G[j];
                        //save max value
                        if (G[j] >= GMax2)
                            GMax2 = G[j];



                        if (grad_diff > 0)
                        {

                            quad_coef = (float)(data.Q_i[i] + QD[j] - 2.0 * y[i] * data.Q_i[j]);
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = (float)(-(grad_diff * grad_diff) / 1e-12);

                            if (obj_diff < localMaxMin.Second)
                            {
                                localMaxMin.First = j;

                                localMaxMin.Second = obj_diff;
                            }
                        }
                    }
                }
                else
                {
                    if (!is_upper_bound(j))
                    {
                        grad_diff = data.GMax - G[j];
                        //save -max
                        if (-G[j] >= GMax2)
                            GMax2 = -G[j];

                        if (grad_diff > 0)
                        {

                            quad_coef = (float)(data.Q_i[i] + QD[j] + 2.0 * y[i] * data.Q_i[j]);
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = (float)(-(grad_diff * grad_diff) / 1e-12);

                            if (obj_diff < localMaxMin.Second)
                            {
                                localMaxMin.First = j;
                                localMaxMin.Second = obj_diff;
                            }
                        }
                    }
                }
            }

            data.GMax2 = GMax2;

            //signal for thread that computation complete
            data.ResetEvent.Set();
        }





        public void Dispose()
        {
            problem = null;

            Q = null;
            QD = null;

            this.active_set = null;
            this.alpha = null;
            this.alpha_status = null;
            this.G_bar = null;
            this.kernel = null;
            this.lockObj = null;
           

            for (int i = 0; i <  numberOfThreads; i++)
            {
                maxPairs[i]  = null;
                maxPairsWaitCallbacks[i] = null;
                maxPairThreadsData[i] = null;

                minPairs[i] = null;
                minPairsWaitCallbacks[i] = null;
                minPairThreadsData[i].Q_i = null;
                minPairThreadsData[i] = null;

                resetEvents[i].Dispose();

            }
            this.maxPairs = null;
            this.maxPairsWaitCallbacks = null;
            this.maxPairThreadsData = null;
            this.minPairs = null;
            this.minPairsWaitCallbacks = null;
            this.minPairThreadsData = null;
            this.p = null;
            this.partition = null;
            this.y = null;
            
            
        }
    }
}