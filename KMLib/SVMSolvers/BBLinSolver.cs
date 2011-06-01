using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using System.Diagnostics;

namespace KMLib.SVMSolvers
{


    /// <summary>
    /// This solver use Barzilai-Borwein projected method for solving QP constrained problem
    /// 
    /// </summary>
 
    public class BBLinSolver : LinearSolver
    {
        private Problem<SparseVec> train;
        
       
        private int maxInnerIter;
        private Random rnd;

        /// <summary>
        /// probability of choosing step2 in BB step
        /// </summary>
        private double probStep2=0.6;
        
        
        
        /// <summary>
        /// current solver iteration
        /// </summary>
        private int iter;



        public BBLinSolver(Problem<SparseVec> problem, float C)
            : base(problem, C)
        {

            rnd =new  Random();
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

                float[] w = new float[w_size];
                //for (int z = 0; z < w.Length; z++)
                //{
                //    w[z] = 1.0f;
                //}


                int e0 = start[0] + count[0];
                int k = 0;
                for (; k < e0; k++)
                    sub_prob.Y[k] = +1;
                for (; k < sub_prob.ElementsCount; k++)
                    sub_prob.Y[k] = -1;



                solve_l2r_l2_svc_bb(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);
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
                    solve_l2r_l2_svc_bb(sub_prob, w, epsilon, weighted_C[i], C);

                    for (j = 0; j < n; j++)
                        model.W[j * nr_class + i] = w[j];
                }
            }



            return model;
        }

        private void solve_l2r_l2_svc_bb(Problem<SparseVec> sub_prob, float[] w, double epsilon, double Cp, double Cn)
        {


            double obj = Double.PositiveInfinity;
            int maxIter = 200000;
          
            float[] alpha = new float[sub_prob.ElementsCount];

            float[] alphaOld = new float[sub_prob.ElementsCount];


            //float[] deltas = new float[sub_prob.ElementsCount];
            //float[] vals = new float[sub_prob.ElementsCount];

            float[] diag = new float[] { (float)(0.5 / Cn), 0, (float)(0.5 / Cp) };

            //gradient and previous gradient
            float[] projGrad = new float[sub_prob.ElementsCount];
            float[] oldGrad = new float[sub_prob.ElementsCount];

            float gradNorm = float.MaxValue; 
            gradNorm = ComputeGradient(sub_prob, w, alpha, diag, ref projGrad);
                       
            iter = 0;
            float step = 0.1f;
           
            Stopwatch st = new Stopwatch();
            st.Start();

            //we do as many steps
            maxIter = 10000;

          //  obj = ComputeObj(w, alpha, sub_prob, diag);
          

            while (iter <= maxIter)
            {

                //remember old alpha
                Buffer.BlockCopy(alpha, 0, alphaOld, 0, alpha.Length * sizeof(float));

                //do  projected sep step
                //x_new = Proj( x_old-step*grad)
                UpdateWandAlpha(alpha, w, -step, projGrad,sub_prob);

#if DEBUG

                obj = ComputeObj(w, alpha, sub_prob, diag);
#endif
                Buffer.BlockCopy(projGrad, 0, oldGrad, 0, projGrad.Length * sizeof(float));
                //computes -gradient
                //grad = b-A*xtemp, 
                gradNorm= ComputeGradient(sub_prob, w, alpha, diag,ref projGrad);

                //stop condition
                if(gradNorm < epsilon)
                {
                    break;
                }

                step = ComputeBBStep(alpha, alphaOld, projGrad, oldGrad);

                
                iter++;
            }

            st.Stop();
           obj= ComputeObj(w, alpha, sub_prob, diag);
           

            Console.WriteLine("Objective value = {0} time={1} ms={2} iter={3}", obj,st.Elapsed, st.ElapsedMilliseconds,iter);
            //Debug.WriteLine("nSV = {0}", nSV);



        }

      
        /// <summary>
        /// Computes Barzilai-Borwein step
        /// 
        /// step1 = (x_new - x_old)'*(x_new - x_old) / (x_new - x_old)'*(grad-grad_old)
        /// step2 = (x_new - x_old)'*(grad-grad_old) / (grad-grad_old)'*(grad-grad_old) 
        /// </summary>
        /// <param name="xNew"></param>
        /// <param name="xOld"></param>
        /// <param name="grad"></param>
        /// <param name="gradOld"></param>
        /// <returns></returns>
        private float ComputeBBStep(float[] xNew, float[] xOld, float[] grad, float[] gradOld)
        {
           

            bool check = (xNew.Length == xOld.Length && xNew.Length==grad.Length &&  grad.Length == gradOld.Length);

            if (!check)
                throw new ArgumentException("Arrays have different sizes");

            float step1=0;
            float step2 = 0;
            float step =0;


            //contains partial results for each parts in BB formula

            //(x_new - x_old)'*(x_new - x_old)
            float xxPart = 0;

            //(x_new - x_old)'*(grad-grad_old)
            float xgPart = 0;

            //grad-grad part (grad-grad_old)'*(grad-grad_old) 
            float ggPart = 0;

            for (int i = 0; i < xNew.Length; i++)
            {
                float xi = xNew[i] - xOld[i];
                float gi = grad[i] - gradOld[i];

                xxPart += xi * xi;

                xgPart += xi * gi;

                ggPart += gi * gi;

            }
            step1 = xxPart / xgPart;
            step2 = xgPart / ggPart;

            step = step1;
            //todo: try different schemes for choosing step
            //if (iter % 3 == 0)
            //    step = step2;

            //random step works better then modulo step (alternating iter%2)
            if (rnd.NextDouble() > probStep2)
                step = step2;




            return step;
        }


      
             

        
        /// <summary>
        /// updates alpha and 'w' vector by step in dir direction
        /// </summary>
        /// <param name="alpha">alpha to update</param>
        /// <param name="w">vector to update</param>
        /// <param name="step">update step</param>
        /// <param name="dir">step in dir direction</param>
        private void UpdateWandAlpha(float[] alpha, float[] w, float step, float[] dir, Problem<SparseVec> sub_prob)
        {

            for (int p = 0; p < alpha.Length; p++)
            {
                
                float old_alpha = alpha[p];

                float alphaStep = step*dir[p];

                //projected update, all alphas>=0
                alpha[p] = Math.Max(alpha[p] + alphaStep, 0);// base_alpha[p] + alphaStep;// 
                
                var spVec = sub_prob.Elements[p];
                
                //real alpha update
                float d = (alpha[p] - old_alpha); // we multiply by *y_i  4 lines lower
                
                //if update is small
                if (Math.Abs(d) < 1e-10)
                    continue;

                sbyte y_i = (sbyte)sub_prob.Y[p];
                d *= y_i;
                int idx=-1;

                //if alpha[p] has changed then we should 
                //change w- vector
                for (int k = 0; k < spVec.Count; k++)
                {
                    idx = spVec.Indices[k] - 1;
                    w[idx] +=d * spVec.Values[k];
                }

            }


        }

       

        /// <summary>
        /// Computes projected -gradient based on "w" i "alpha", result is in 
        /// </summary>
        /// <param name="sub_prob"></param>
        /// <param name="w"></param>
        /// <param name="alpha"></param>
        /// <param name="diag"></param>
        /// <param name="grad"></param>
        private static float ComputeGradient(Problem<SparseVec> sub_prob, float[] w, float[] alpha, float[] diag,ref float[] grad)
        {

            float max = float.NegativeInfinity;

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
                //else
                //{
                //    grad[i] = grad[i];
                //    // projGrad_i[i] = grad_i[i];
                //}

                //minus gradient - descent direction
               // grad[i] = -grad[i];

                //maximum norm
                max = Math.Max(max, Math.Abs(grad[i]));
            }

            return max;

        }

        
    }
}
