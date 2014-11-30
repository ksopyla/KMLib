using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using System.Diagnostics;

namespace KMLib.SVMSolvers
{
    public class ConjugateLinSolver : LinearSolver
    {
        private Problem<SparseVec> train;
        
        /// <summary>
        /// step for computing hessian
        /// </summary>
        private float hess_step;
        private int maxInnerIter;



        public ConjugateLinSolver(Problem<SparseVec> problem, float C)
            : base(problem, C)
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
            int nr_class = 0;
            int[] label;
            int[] start;
            int[] count;

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

                float[] w = new float[w_size];

                int e0 = start[0] + count[0];
                int k = 0;
                for (; k < e0; k++)
                    sub_prob.Y[k] = +1;
                for (; k < sub_prob.ElementsCount; k++)
                    sub_prob.Y[k] = -1;



                solve_l2r_l2_svc_parallel(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);

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

                    solve_l2r_l2_svc_parallel(sub_prob, w, epsilon, weighted_C[i], C);

                    for (j = 0; j < n; j++)
                        model.W[j * nr_class + i] = w[j];
                }
            }



            return model;
        }

        private void solve_l2r_l2_svc_parallel(Problem<SparseVec> sub_prob, float[] w, double epsilon, double Cp, double Cn)
        {


            double obj = Double.PositiveInfinity;
            int maxIter = 200000;
            float stepScaling = 1;

            //todo: remove it QD is not needed
            float[] QD = new float[sub_prob.ElementsCount];
            for (int i = 0; i < sub_prob.ElementsCount; i++)
            {
                QD[i] = sub_prob.Elements[i].DotProduct();
            }

            float[] alpha = new float[sub_prob.ElementsCount];

            float[] alpha_temp = new float[sub_prob.ElementsCount];

            float[] w_temp = new float[sub_prob.FeaturesCount];

            float[] deltas = new float[sub_prob.ElementsCount];
            float[] vals = new float[sub_prob.ElementsCount];
            float[] diag = new float[] { (float)(0.5 / Cn), 0, (float)(0.5 / Cp) };

            //gradient and previous gradient
            float[] projGrad = new float[sub_prob.ElementsCount];
            float[] oldGrad = new float[sub_prob.ElementsCount];

            //directional vecotr
            float[] pi = new float[sub_prob.ElementsCount];
            float[] ri = new float[sub_prob.ElementsCount];

            //array stroing hessian vector product
            float[] hessProd = new float[sub_prob.ElementsCount];
            
            ComputeGradient(sub_prob, w, alpha, diag,ref projGrad);

            //copy gradient values to dir vector

            

            Buffer.BlockCopy(projGrad, 0, pi, 0, projGrad.Length*sizeof(float));
            Buffer.BlockCopy(projGrad, 0, ri, 0, projGrad.Length*sizeof(float));

            // todo:check if its correct
            float rsqr = ri.Sum(x => x * x);

            int iter = 0;

            float step = 0.01f;
            hess_step = 0.01f;
            


            Stopwatch st = new Stopwatch();
            st.Start();

            //we do as many steps
            maxIter = sub_prob.ElementsCount;

            obj = ComputeObj(w, alpha, sub_prob, diag);
            while (iter <= maxIter)
            {

                Buffer.BlockCopy(projGrad, 0, oldGrad, 0, projGrad.Length*sizeof(float));

                //compute temp solution for comuting aproximate hessian vector product
                // xtemp = xi+hess_step*pi;
                UpdateWandAlpha(alpha_temp, w_temp, alpha, w, hess_step, pi,sub_prob);
                obj = ComputeObj(w_temp, alpha_temp, sub_prob, diag);
                ComputeGradient(sub_prob, w_temp, alpha_temp, diag,ref projGrad);

                ComputeHessianVectorProduct(oldGrad, projGrad, hess_step, ref hessProd);

                float piHess = DotProduct(pi, hessProd);

                step = rsqr / piHess;

                //do normal conjugate step, x=x+step*pi;
                UpdateWandAlpha(alpha, w, alpha, w, step, pi, sub_prob);

                //update ri vector ri=ri-step*Ap;
                UpdateVector(ri,-step, hessProd, ref ri);
                
                // rsqr2= ri'*ri
                float rsqr2 = DotProduct(ri, ri);
                float beta = rsqr2 / rsqr;

                //update pi, pi = ri+beta*pi
                UpdateVector(ri, beta, pi, ref pi);

                ComputeGradient(sub_prob, w, alpha, diag, ref projGrad);

                rsqr = rsqr2;
                obj = ComputeObj(w, alpha, sub_prob, diag);
                iter++;
            }

            st.Stop();
           obj= ComputeObj(w, alpha, sub_prob, diag);
           

            Console.WriteLine("Objective value = {0} time={1} ms={2}", obj,st.Elapsed, st.ElapsedMilliseconds);
        }


        /// <summary>
        /// updates vector by adding some step in dirVec direction
        /// 
        /// v=base+step*dirVec;
        /// </summary>
        /// <param name="step">step</param>
        /// <param name="dirVec">directional vector</param>
        /// <param name="vector">updated vector</param>
        private void UpdateVector(float[] baseVec ,float step, float[] dirVec, ref float[] vector)
        {
            if (vector.Length != dirVec.Length)
                throw new ArgumentException("vectors 'vector' and 'dirVec' should have same dimensions");
            if (vector.Length != baseVec.Length)
                throw new ArgumentException("vectors 'vector' and 'basevec' should have same dimensions");


            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = baseVec[i]+ step * dirVec[i];
            }     
        }


        /// <summary>
        /// computse dense dot product between two vectors
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        private float DotProduct(float[] v1, float[] v2)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("vectors should have same dimensions");

            float sum = 0;
            for (int k = 0; k <v1.Length ; k++)
            {
                sum += v1[k] * v2[k];
            }
            return sum;
        }
        /// <summary>
        /// Computes aproximte hessian vector product, it use formula
        /// H*p ~ (grad(x+delta*p)-grad(x))/step;
        /// </summary>
        /// <param name="grad">gradient in point x</param>
        /// <param name="gradDelta">gradient in point x+delta*p</param>
        /// <param name="hess_step"></param>
        /// <param name="hessProd"></param>
        private void ComputeHessianVectorProduct(float[] grad, float[] gradDelta, float hess_step, ref float[] hessProd)
        {
            for (int i = 0; i < hessProd.Length; i++)
            {

                hessProd[i] = -(gradDelta[i] - grad[i]) / hess_step;
            } 
        }

        

        
        /// <summary>
        /// updates alpha and 'w' vector by step in dir direction
        /// </summary>
        /// <param name="update_alpha">alpha to update</param>
        /// <param name="update_w">vector to update</param>
        /// <param name="base_alpha">base value of alpha</param>
        /// <param name="base_w"></param>
        /// <param name="step">update step</param>
        /// <param name="dir">step in dir direction</param>
        private void UpdateWandAlpha(float[] update_alpha, float[] update_w, float[] base_alpha, float[] base_w, float step, float[] dir, Problem<SparseVec> sub_prob)
        {


            //copy w vector to update_w
            Buffer.BlockCopy(base_w, 0, update_w, 0, base_w.Length * sizeof(float));


            for (int p = 0; p < base_alpha.Length; p++)
            {
                
                float old_alpha = base_alpha[p];

                float alphaStep = step*dir[p];

                update_alpha[p] = Math.Max(base_alpha[p] + alphaStep, 0);// base_alpha[p] + alphaStep;// 
                
                var spVec = sub_prob.Elements[p];
                float d = (update_alpha[p] - old_alpha); // we multiply by *y_i  4 lines lower
                
                if (Math.Abs(d) < 1e-10)
                    continue;
                sbyte y_i = (sbyte)sub_prob.Y[p];
                d *= y_i;
                int idx=-1;
                for (int k = 0; k < spVec.Count; k++)
                {
                    idx = spVec.Indices[k] - 1;
                    update_w[idx] +=d * spVec.Values[k];
                }

            }


        }

       

        /// <summary>
        /// Computes -gradient based on "w" i "alpha", result is in 
        /// </summary>
        /// <param name="sub_prob"></param>
        /// <param name="w"></param>
        /// <param name="alpha"></param>
        /// <param name="diag"></param>
        /// <param name="grad"></param>
        private static void ComputeGradient(Problem<SparseVec> sub_prob, float[] w, float[] alpha, float[] diag,ref float[] grad)
        {
            
        
            for (int i = 0; i < grad.Length; i++)
            {
                //intialized gradient i-th row
                grad[i] = 0;

                var element = sub_prob.Elements[i];
                //computes dot product between W and all elements
                for (int k = 0; k < element.Count; k++)
                {
                    grad[i] += w[element.Indices[k] - 1] * element.Values[k];

                }

                sbyte y_i = (sbyte)sub_prob.Y[i];
                //"minus" -gradient
                //grad[i] =1- grad[i] * y_i - alpha[i]*diag[y_i+1];

                //normal gradient
                grad[i] = grad[i] * y_i + alpha[i] * diag[y_i + 1] - 1;

                //projection
                if (alpha[i] == 0)
                {
                    grad[i] = Math.Min(0, grad[i]);
                }

                //minus gradient - descent direction
                grad[i] = -grad[i];

            }

        }

        
    }
}
