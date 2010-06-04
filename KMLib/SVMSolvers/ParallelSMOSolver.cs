using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace KMLib.SVMSolvers
{
    /// <summary>
    /// Modified version of J. Plat SMO solver for SVM, 
    /// Simultaniously take step according to vectors which are in different direction,
    /// on each processor core its computed one step
    /// </summary>
    public class ParallelSMOSolver<TProblemElement> : Solver<TProblemElement>
    {

        static Random random = new Random();

        //float C = 0.05f;
        float tolerance = 0.01f;
        float epsilon = 0.01f;

        /// <summary>Array of Lagrange multipliers alpha_i</summary>
        protected float[] alpha;

        /// <summary>
        /// Threshold.
        /// </summary>
        protected float b = 0f;

        protected float[] errorCache;

        /// <summary>
        /// stores non zero alphas indexes
        /// </summary>
        List<int> nonZeroAlphas = new List<int>();

        /// <summary>Cache of Gram matrix diagonal.</summary>
        // protected float[] diagGramCache;


        public ParallelSMOSolver(Problem<TProblemElement> problem, IKernel<TProblemElement> kernel, float C)
            : base(problem, kernel, C)
        {

        }

        public override Model<TProblemElement> ComputeModel()
        {
            errorCache = new float[problem.ElementsCount];
            alpha = new float[problem.ElementsCount];
            // SMO algorithm
            int numChange = 0;
            int examineAll = 1;
            int kktViolatiors = 0;
            int cores = System.Environment.ProcessorCount;

            while (numChange > 0 || examineAll > 0)
            {
                numChange = 0;
                //if (examineAll > 0)
                //{
                for (int k = 0; k < problem.ElementsCount; k++)
                {

                    int i2 = -1;
                    //find first index
                    if (ExamineExample(k))
                    {
                        // numChange++;

                        ConcurrentBag<AlphaPair> newAlphas = new ConcurrentBag<AlphaPair>();

                        var loopRes = Parallel.ForEach(FindSubIndexes(cores, k), (index, loopState) =>
                        {
                            AlphaPair ap = TakeStep(k, index);

                            if (ap != null)
                            {

                                if (newAlphas.Count >= cores)
                                {
                                    //stop searching
                                    loopState.Stop();
                                    //loopState.Break();
                                }
                                else
                                    newAlphas.Add(ap);
                            }
                        });

                        //update alphas array
                        float avgAlpha1 = 0f;
                        float bnew = 0f;
                        foreach (var item in newAlphas)
                        {
                            avgAlpha1 += item.FirstAlpha;
                            alpha[item.SecondIndex] = item.SecondAlpha;

                            bnew += item.Threshold;
                        }

                        float countAlphas = newAlphas.Count;
                        avgAlpha1 = avgAlpha1 / countAlphas;
                        bnew = bnew / countAlphas;

                        alpha[k] = avgAlpha1;
                       
                        //update threshold b and calculate error cache
                        //todo: plus or minus?
                        float delta_b = bnew - b;
                        b = bnew;


                        //float t1 = y1 * (a1 - alph1);
                        //float t2 = y2 * (a2 - alph2);

                        //for (int i = 0; i < problem.ElementsCount; i++)
                        //{
                        //    if (0 < alpha[i] && alpha[i] < C)
                        //    {
                        //        errorCache[i] +=
                        //            t1 * Product(i1, i)
                        //            + t2 * Product(i2, i) - delta_b;
                        //    }
                        //}

                        //errorCache[i1] = 0f;
                        //errorCache[i2] = 0f;


                       
                        


                    }
                }
                #region Old code
                //else
                //{
                //    for (int k = 0; k < problem.ElementsCount; k++)
                //    {
                //        if (alpha[k] != 0 && alpha[k] != C)
                //        {
                //            if (ExamineExample(k)) numChange++;
                //        }
                //    }
                //}

                //if (examineAll == 1)
                //{
                //    examineAll = 0;
                //}
                //else if (numChange == 0)
                //{
                //    examineAll = 1;
                //}
                #endregion
            }

            // cleaning
            errorCache = null;

            #region Building Model
            Model<TProblemElement> model = new Model<TProblemElement>();
            model.NumberOfClasses = 2;
            model.Alpha = alpha;
            model.Rho = b;


            List<TProblemElement> supportElements = new List<TProblemElement>();
            List<int> suporrtIndexes = new List<int>(alpha.Length);
            for (int i = 0; i < alpha.Length; i++)
            {
                if (alpha[i] > 0)
                {
                    supportElements.Add(problem.Elements[i]);
                    suporrtIndexes.Add(i);
                }

            }
            model.SupportElements = supportElements.ToArray();
            model.SupportElementsIndexes = suporrtIndexes.ToArray();
            #endregion

            return model;
        }

        private IEnumerable<int> FindSubIndexes(int cores, int k)
        {
            int heuristicCount = 3;
            for (int i = 0; i < heuristicCount; i++)
            {
                switch (i)
                {
                    case 0:
                        foreach (var item in FindScnIndNMax(k,cores))
                        {
                            yield return item;    
                        }
                        
                        break;
                    case 1:
                        foreach (var item in FindScndIndSecondHeuristic(k))
                        {
                            yield return item;
                        }
                        break;
                    case 2:
                        foreach (var item in FindScndIndAll(k))
                        {
                            yield return item;
                        }
                        break;
                }
            }
            yield break;
        }


        /// <summary>Indicates if a step has been taken.</summary>
        /// <returns>Index of first alpha</returns>
        private bool ExamineExample(int i1)
        {
            float y1 = problem.Labels[i1],
                  alph1 = alpha[i1],
                  E1 = 0;
            E1 = ErrorForAlpha(i1);

            float r1 = y1 * E1;

            // Outer loop: choosing the first element.
            // First heuristic: testing for violation of KKT conditions
            if ((r1 < -tolerance && alph1 < C) || (r1 > tolerance && alph1 > 0))
            {

                return true;

                #region
                // Inner loop: choosing the second element

                // Second heuristing: testing a priori most interesting element
                // return FindSecondIndex(i1, E1);

                #endregion

            }

            return false;
        }

        private IEnumerable<int> FindScndIndSecondHeuristic(int index)
        {


            int k0 = random.Next(nonZeroAlphas.Count);
            for (int k = k0; k < nonZeroAlphas.Count + k0; k++)
            {
                int i2 = k % problem.ElementsCount;

                if (index == i2)
                    continue;

                yield return nonZeroAlphas[i2];
            }
            #region old code
            // Third heuristic: testing non-bound elements
            //int k0 = random.Next(problem.ElementsCount);
            //for (int k = k0; k < problem.ElementsCount + k0; k++)
            //{
            //    int i2 = k % problem.ElementsCount;

            //    if (alpha[i2] > 0 && alpha[i2] < C)
            //    {
            //        return i2;

            //        //if (TakeStep(i1, i2))
            //        //{
            //        //    return true;
            //        //}
            //    }
            //}
            #endregion

        }


        private IEnumerable<int> FindScndIndAll(int index)
        {

            // Finally: testing all elements
            int k0 = random.Next(problem.ElementsCount);
            for (int k = k0; k < problem.ElementsCount + k0; k++)
            {
                int i2 = k % problem.ElementsCount;

                if (index == i2)
                    continue;

                yield return i2;
                //if (TakeStep(i1, i2))
                //{
                //    return true;
                //}
            }
        }

        private IEnumerable<int> FindScnIndNMax(int index,int n)
        {
            int i2max = -1;
            float tmax = 0;
            float tmin = 0;
            
                        
            SortedList<float,int> s = new SortedList<float,int>(n);
            

            float E1 = ErrorForAlpha(index);
            for (int i = 0; i < nonZeroAlphas.Count; i++)
            {
                int i2 = nonZeroAlphas[i];

                float E2 = errorCache[i2];
                float temp = Math.Abs(E1 - E2);

                if (temp > tmax)
                {
                    tmax = temp;
                    s.Add(tmax, i2);

                    s.RemoveAt(n - 1);
                }
                else if (temp > tmin)
                {
                    tmin = temp;
                    s.Add(tmin, i2);
                    s.RemoveAt(n - 1);
                }
            }

            foreach (var item in s)
            {
                yield return item.Value;
            }
            

            #region
            //old version
            //for (int i2 = 0; i2 < problem.ElementsCount; i2++)
            //{
            //    if (alpha[i2] > 0 && alpha[i2] < C)
            //    {
            //        float E2 = errorCache[i2];
            //        float temp = Math.Abs(E1 - E2);

            //        if (temp > tmax)
            //        {
            //            tmax = temp;
            //            i2max = i2;
            //        }

            //    }
            //}
            #endregion

            // return i2max;

            //if (i2max >= 0)
            //{
            //    return i2max;

            //    //if (TakeStep(i1, i2max))
            //    //{
            //    //    return true;
            //    //}
            //}
            //return i2max;
        }


        /// <summary>
        /// Computes Error for alpha_i
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        private float ErrorForAlpha(int i)
        {
            float y1 = problem.Labels[i],
                  alph1 = alpha[i],
                  E1 = 0;

            if (alph1 > 0 && alph1 < C)
            {
                E1 = errorCache[i];
            }
            else
            {
                E1 = DecisionFunc(i) - y1;
            }
            return E1;
        }


        /// <summary>This method solves the two Lagrange multipliers problem.</summary>
        /// <returns>Indicates if the step has been taken.</returns>
        private AlphaPair TakeStep(int i1, int i2)
        {

            float y1 = 0, y2 = 0, s = 0;
            float alph1 = 0, alph2 = 0; /* old_values of alpha_1, alpha_2 */
            float a1 = 0, a2 = 0; /* new values of alpha_1, alpha_2 */
            float E1 = 0, E2 = 0, L = 0, H = 0, k11 = 0, k22 = 0, k12 = 0, eta = 0, Lobj = 0, Hobj = 0;

            if (i1 == i2) return null; // no step taken

            alph1 = alpha[i1];
            y1 = problem.Labels[i1];
            E1 = ErrorForAlpha(i1);

            alph2 = alpha[i2];
            y2 = problem.Labels[i2];
            E2 = ErrorForAlpha(i2);

            s = y1 * y2;


            if (y1 == y2)
            {
                float gamma = alph1 + alph2;
                if (gamma > C)
                {
                    L = gamma - C;
                    H = C;
                }
                else
                {
                    L = 0;
                    H = gamma;
                }
            }
            else
            {
                float gamma = alph2 - alph1;
                if (gamma > 0)
                {
                    L = gamma;
                    H = C;
                }
                else
                {
                    L = 0;
                    H = C - gamma;
                }
            }

            if (L == H)
            {
                return null; // no step take
            }

            k11 = Product(i1, i1);
            k12 = Product(i1, i2);
            k22 = Product(i2, i2);
            eta = 2 * k12 - k11 - k22;


            if (eta < 0)
            {
                //original version with plus  
                //a2 = alph2 + y2 * (E2 - E1) / eta;

                a2 = alph2 - y2 * (E1 - E2) / eta;

                if (a2 < L) a2 = L;
                else if (a2 > H) a2 = H;
            }
            else
            {
                {
                    float c1 = eta / 2;
                    float c2 = y2 * (E1 - E2) - eta * alph2;
                    Lobj = c1 * L * L + c2 * L;
                    Hobj = c1 * H * H + c2 * H;
                }

                if (Lobj > Hobj + epsilon) a2 = L;
                else if (Lobj < Hobj - epsilon) a2 = H;
                else a2 = alph2;
            }

            if (Math.Abs(a2 - alph2) < epsilon * (a2 + alph2 + epsilon))
            {
                return null; // no step taken
            }

            a1 = alph1 - s * (a2 - alph2);
            if (a1 < 0)
            {
                a2 += s * a1;
                a1 = 0;
            }
            else if (a1 > C)
            {
                float t = a1 - C;
                a2 += s * t;
                a1 = C;
            }


            float b1 = 0, b2 = 0, bnew = 0;

            if (a1 > 0 && a1 < C)
                bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
            else
            {
                if (a2 > 0 && a2 < C)
                    bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
                else
                {
                    b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
                    b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
                    bnew = (b1 + b2) / 2;
                }
            }

            return new AlphaPair() { FirstIndex = i1, FirstAlpha = a1, SecondIndex = i2, SecondAlpha = a2, Threshold = bnew };

            //todo: plus or minus?
            //float delta_b = bnew - b;
            //b = bnew;


            //float t1 = y1 * (a1 - alph1);
            //float t2 = y2 * (a2 - alph2);

            //for (int i = 0; i < problem.ElementsCount; i++)
            //{
            //    if (0 < alpha[i] && alpha[i] < C)
            //    {
            //        errorCache[i] +=
            //            t1 * Product(i1, i)
            //            + t2 * Product(i2, i) - delta_b;
            //    }
            //}

            //errorCache[i1] = 0f;
            //errorCache[i2] = 0f;


            //alpha[i1] = a1;
            //alpha[i2] = a2;

            //return true; // step taken
        }


        /// <summary>
        /// Helper method for computing Kernel Product
        /// </summary>
        /// <param name="i1"></param>
        /// <param name="i"></param>
        /// <returns></returns>
        private float Product(int i1, int i)
        {

            return kernel.Product(i1, i);

        }


        /// <summary>
        /// Compute decision function for k'th element
        /// </summary>
        /// <param name="k"></param>
        /// <returns></returns>
        private float DecisionFunc(int k)
        {

            float sum = 0;
            //sum( apha_i*y_i*K(x_i,problemElement)) + b
            //sum can by compute only on support vectors
            for (int i = 0; i < problem.Elements.Length; i++)
            {
                if (alpha[i] != 0)
                    sum += alpha[i] * problem.Labels[i] * Product(i, k);
            }

            sum -= b;
            return sum;

        }

    }
}