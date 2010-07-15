using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;

namespace KMLib.Helpers
{
    /// <summary>
    /// SMO solver for SVM, implemented according to John C. Plat article
    /// "Fast Training of Support Vector Machin using Sequential Minimal Optimization"
    /// some code based on NSvm library http://nsvm.sourceforge.net/
    /// </summary>
    public class SMOSolver<TProblemElement> : Solver<TProblemElement>
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

        /// <summary>Cache of Gram matrix diagonal.</summary>
       // protected float[] diagGramCache;


        public SMOSolver(Problem<TProblemElement> problem, IKernel<TProblemElement> kernel, float C)
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

            while (numChange > 0 || examineAll > 0)
            {
                numChange = 0;
                if (examineAll > 0)
                {
                    for (int k = 0; k < problem.ElementsCount; k++)
                    {
                        if (ExamineExample(k)) numChange++;
                    }
                }
                else
                {
                    for (int k = 0; k < problem.ElementsCount; k++)
                    {
                        if (alpha[k] != 0 && alpha[k] != C)
                        {
                            if (ExamineExample(k)) numChange++;
                        }
                    }
                }

                if (examineAll == 1)
                {
                    examineAll = 0;
                }
                else if (numChange == 0)
                {
                    examineAll = 1;
                }
            }

            // cleaning
            errorCache = null;
            


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


            return model;
        }

        /// <summary>Indicates if a step has been taken.</summary>
        private bool ExamineExample(int i1)
        {
            float y1 = problem.Labels[i1],
                  alph1 = alpha[i1],
                  E1 = 0;
            
           
            if (alph1 > 0 && alph1 < C)
            {
                E1 = errorCache[i1];
            }
            else
            {
                E1 = DecisionFunc(i1) - y1;
            }

            float r1 = y1 * E1;

            // Outer loop: choosing the first element.
            // First heuristic: testing for violation of KKT conditions
            if ((r1 < -tolerance && alph1 < C) || (r1 > tolerance && alph1 > 0))
            {
                // Inner loop: choosing the second element

                // Second heuristing: testing a priori most interesting element
                int i2max = -1;
                float tmax = 0;

                for (int i2 = 0; i2 < problem.ElementsCount; i2++)
                {
                    if (alpha[i2] > 0 && alpha[i2] < C)
                    {
                        float E2 = errorCache[i2];
                        float temp = Math.Abs(E1 - E2);

                        if (temp > tmax)
                        {
                            tmax = temp;
                            i2max = i2;
                        }
                    }
                }

                if (i2max >= 0)
                {
                    if (TakeStep(i1, i2max))
                    {
                        return true;
                    }
                }

                // Third heuristic: testing non-bound elements
                int k0 = random.Next(problem.ElementsCount);
                for (int k = k0; k < problem.ElementsCount + k0; k++)
                {
                    int i2 = k % problem.ElementsCount;

                    if (alpha[i2] > 0 && alpha[i2] < C)
                    {
                        if (TakeStep(i1, i2))
                        {
                            return true;
                        }
                    }
                }

                // Finally: testing all elements
                k0 = random.Next(problem.ElementsCount);
                for (int k = k0; k < problem.ElementsCount + k0; k++)
                {
                    int i2 = k % problem.ElementsCount;

                    if (TakeStep(i1, i2))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        /*
        private void BuildCacheForRow(int i1)
        {
            for (int i2 = 0; i2 < problem.ElementsCount; i2++)
            {
                I1Cache[i2] = kernel.Product(problem.Elements[i1], problem.Elements[i2]);
            }
        }
        
         */ 

        /// <summary>This method solves the two Lagrange multipliers problem.</summary>
        /// <returns>Indicates if the step has been taken.</returns>
        private bool TakeStep(int i1, int i2)
        {

            float y1 = 0, y2 = 0, s = 0;
            float alph1 = 0, alph2 = 0; /* old_values of alpha_1, alpha_2 */
            float a1 = 0, a2 = 0; /* new values of alpha_1, alpha_2 */
            float E1 = 0, E2 = 0, L = 0, H = 0, k11 = 0, k22 = 0, k12 = 0, eta = 0, Lobj = 0, Hobj = 0;

            if (i1 == i2) return false; // no step taken

            alph1 = alpha[i1];
            y1 = problem.Labels[i1];
            if (alph1 > 0 && alph1 < C)
            {
                E1 = errorCache[i1];
            }
            else
            {
                E1 = DecisionFunc(i1) - y1;
            }

            alph2 = alpha[i2];
            y2 = problem.Labels[i2];
            if (alph2 > 0 && alph2 < C)
            {
                E2 = errorCache[i2];
            }
            else
            {
                E2 = DecisionFunc(i2) - y2;
            }

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
                    H = C-gamma;
                }
                /*
                float gamma = alph1 - alph2;
                if (gamma > 0)
                {
                    L = 0;
                    H = C - gamma;
                }
                else
                {
                    L = -gamma;
                    H = C;
                }
                 */ 
            }

            if (L == H)
            {
                return false; // no step take
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
                return false; // no step taken
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


            //todo: plus or minus?
            float delta_b = bnew - b;
            b = bnew;





            float t1 = y1 * (a1 - alph1);
            float t2 = y2 * (a2 - alph2);

            for (int i = 0; i < problem.ElementsCount; i++)
            {
                if (0 < alpha[i] && alpha[i] < C)
                {
                    errorCache[i] +=
                        t1 * Product(i1, i)
                        + t2 * Product(i2, i) - delta_b;
                }
            }

            errorCache[i1] = 0f;
            errorCache[i2] = 0f;


            alpha[i1] = a1;
            alpha[i2] = a2;

            return true; // step taken
        }


        /// <summary>
        /// Helper method for computing Kernel Product
        /// </summary>
        /// <param name="i1"></param>
        /// <param name="i"></param>
        /// <returns></returns>
        private float Product(int i1, int i)
        {

            return kernel.Product(i1,i);

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
                if(alpha[i]!=0)
                    sum += alpha[i] * problem.Labels[i] * Product(i,k);
            }

            sum -= b;
            return sum;

        }

    }
}