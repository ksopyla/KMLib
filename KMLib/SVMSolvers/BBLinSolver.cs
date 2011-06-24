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

        int maxIter = 100000;

        /// <summary>
        /// probability of choosing step2 in BB step
        /// </summary>
        private double probStep2=0.5;
        
        
        
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

                double[] w = new double[w_size];
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



               // solve_l2r_l2_svc_bb(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);
                solve_l2r_l2_svc_nm_bb(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);
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
                    solve_l2r_l2_svc_bb(sub_prob, w, epsilon, weighted_C[i], C);

                    for (j = 0; j < n; j++)
                        model.W[j * nr_class + i] = w[j];
                }
            }



            return model;
        }

        private void solve_l2r_l2_svc_bb(Problem<SparseVec> sub_prob, double[] w, double epsilon, double Cp, double Cn)
        {


            double obj = Double.PositiveInfinity;
           
          
            double[] alpha = new double[sub_prob.ElementsCount];

            double[] alphaOld = new double[sub_prob.ElementsCount];


            //float[] deltas = new float[sub_prob.ElementsCount];
            //float[] vals = new float[sub_prob.ElementsCount];

            double[] diag = new double[] { (double)(0.5 / Cn), 0, (double)(0.5 / Cp) };

            //gradient and previous gradient
            double[] projGrad = new double[sub_prob.ElementsCount];
            double[] oldGrad = new double[sub_prob.ElementsCount];

            double gradNorm = double.MaxValue; 
            gradNorm = ComputeGradient(sub_prob, w, alpha, diag, ref projGrad);
                       
            iter = 0;
            double step = 0.01f;
           
            Stopwatch st = new Stopwatch();
            st.Start();


          //  obj = ComputeObj(w, alpha, sub_prob, diag);
          

            while (iter <= maxIter)
            {
                

                //remember old alpha
                Buffer.BlockCopy(alpha, 0, alphaOld, 0, alpha.Length * sizeof(double));

                //do  projected sep step
                //x_new = Proj( x_old-step*grad)
                UpdateWandAlpha(alpha, w, -step, projGrad,sub_prob);

#if DEBUG

                obj = ComputeObj(w, alpha, sub_prob, diag);
#endif
                Buffer.BlockCopy(projGrad, 0, oldGrad, 0, projGrad.Length * sizeof(double));
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

        private void solve_l2r_l2_svc_nm_bb(Problem<SparseVec> sub_prob, double[] w, double epsilon, double Cp, double Cn)
        {


            double obj = Double.PositiveInfinity;
            

            double[] alpha = new double[sub_prob.ElementsCount];

            double[] alphaOld = new double[sub_prob.ElementsCount];

            double[] alpha_tmp = new double[sub_prob.ElementsCount];
            double[] w_tmp = new double[sub_prob.FeaturesCount];
            
            double[] diag = new double[] { (double)(0.5 / Cn), 0, (double)(0.5 / Cp) };

            //gradient and previous gradient
            double[] projGrad = new double[sub_prob.ElementsCount];
            double[] oldGrad = new double[sub_prob.ElementsCount];

            double gradNorm = double.MaxValue;
            gradNorm = ComputeGradient(sub_prob, w, alpha, diag, ref projGrad);

            iter = 0;
            double step = 0.01f;

            int M = 10;
            double sig1 = 0.1;
            double sig2 = 0.9;
            double gamma = 10e-4;
            double lambda = 0;
            double l_min = 10e-20;
            double l_max = 10e20;

            double[] func_vals = new double[M];
            double max_funcVal = 0;
            Stopwatch st = new Stopwatch();
            st.Start();

           

            //  obj = ComputeObj(w, alpha, sub_prob, diag);


            while (iter <= maxIter)
            {

                max_funcVal = func_vals.Max();

                lambda = step;

                for (int i = 0; i < 10; i++)
                {
                    Buffer.BlockCopy(alpha, 0, alpha_tmp, 0, alpha.Length * sizeof(double));
                    Buffer.BlockCopy(w, 0, w_tmp, 0, w.Length * sizeof(double));

                    UpdateWandAlpha(alpha_tmp, w_tmp, -lambda, projGrad, sub_prob);

                    obj = ComputeObj(w_tmp, alpha_tmp, sub_prob, diag);

                    double linPart = gamma * ComputeDiff(alpha_tmp, alpha, projGrad);
                    if (obj <= (max_funcVal + linPart))
                    {
                        int idx = (iter+1) % M;
                        func_vals[idx] = obj;
                        break;

                    }
                    lambda = (sig1 * lambda + sig2 * lambda) / 2;

                }


                //remember old alpha

                //Buffer.BlockCopy(alpha, 0, alphaOld, 0, alpha.Length * sizeof(double));
                //Buffer.BlockCopy(alpha_tmp, 0, alpha, 0, alpha.Length * sizeof(double));
                var tmpPtr = alphaOld;
                alphaOld = alpha;
                alpha = alpha_tmp;
                alpha_tmp = tmpPtr;

                //Buffer.BlockCopy(w_tmp, 0, w, 0, w.Length * sizeof(double));
                var w_tmpPtr = w;
                w = w_tmp;
                w_tmp = w_tmpPtr;

               
               // Buffer.BlockCopy(projGrad, 0, oldGrad, 0, projGrad.Length * sizeof(double));
                var grad_tmpPtr = oldGrad;
                oldGrad = projGrad;
                projGrad = grad_tmpPtr;


                //computes -gradient
                //grad = b-A*xtemp, 
                gradNorm = ComputeGradient(sub_prob, w, alpha, diag, ref projGrad);

                //stop condition
                if (gradNorm < epsilon)
                {
                    break;
                }

                step = ComputeBBStep(alpha, alphaOld, projGrad, oldGrad);

                iter++;
            }

            st.Stop();
            obj = ComputeObj(w, alpha, sub_prob, diag);

            Console.WriteLine("Objective value = {0} time={1} ms={2} iter={3}", obj, st.Elapsed, st.ElapsedMilliseconds, iter);
            //Debug.WriteLine("nSV = {0}", nSV);

        }

        /// <summary>
        /// computes dot product = (vec1-vec2)*mulVec
        /// </summary>
        /// <param name="vec1"></param>
        /// <param name="vec2"></param>
        /// <param name="mullVec"></param>
        /// <returns></returns>
        private double ComputeDiff(double[] vec1, double[] vec2, double[] mullVec)
        {
            double diff = 0;
            for (int i = 0; i < vec1.Length; i++)
            {
                diff += (vec1[i] - vec2[i]) * mullVec[i];
                
            }
            return diff;
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
        private double ComputeBBStep(double[] xNew, double[] xOld, double[] grad, double[] gradOld)
        {
           

            bool check = (xNew.Length == xOld.Length && xNew.Length==grad.Length &&  grad.Length == gradOld.Length);

            if (!check)
                throw new ArgumentException("Arrays have different sizes");

            double step1=0;
            double step2 = 0;
            double step =0;

            //contains partial results for each parts in BB formula
            //(x_new - x_old)'*(x_new - x_old)
            double xxPart = 0;

            //(x_new - x_old)'*(grad-grad_old)
            double xgPart = 0;

            //grad-grad part (grad-grad_old)'*(grad-grad_old) 
            double ggPart = 0;

            for (int i = 0; i < xNew.Length; i++)
            {
                double xi = xNew[i] - xOld[i];
                double gi = grad[i] - gradOld[i];

                xxPart += xi * xi;

                xgPart += xi * gi;

                ggPart += gi * gi;

            }
            step1 = xxPart / xgPart;
            step2 = xgPart / ggPart;

            step = step1;
            //todo: try different schemes for choosing step
            if (iter % 2 == 0)
            {
                step = step2;
            }

            //random step works better then modulo step (alternating iter%2)

            double rndProb = rnd.NextDouble();
            if (rndProb > probStep2)
            {
                //step = step2;
            }


            return step;
        }


      
             

        
        /// <summary>
        /// updates alpha and 'w' vector by step in dir direction
        /// </summary>
        /// <param name="alpha">alpha to update</param>
        /// <param name="w">vector to update</param>
        /// <param name="step">update step</param>
        /// <param name="dir">step in dir direction</param>
        private void UpdateWandAlpha(double[] alpha, double[] w, double step, double[] dir, Problem<SparseVec> sub_prob)
        {

            for (int p = 0; p < alpha.Length; p++)
            {
                
                double old_alpha = alpha[p];
                double alphaStep = step*dir[p];

                //projected update, all alphas>=0
                double alpha_new = Math.Max(alpha[p] + alphaStep, 0);// base_alpha[p] + alphaStep;// 
                
                //real alpha update
                double d = (alpha_new - old_alpha); // we multiply by *y_i  4 lines lower
                
                //if update is small
                if (Math.Abs(d) < 1e-12)
                    continue;

                sbyte y_i = (sbyte)sub_prob.Y[p];
                d *= y_i;
                int idx=-1;

                var spVec = sub_prob.Elements[p];
                //if alpha[p] has changed then we should 
                //change w- vector
               
                for (int k = 0; k < spVec.Count; k++)
                {
                    idx = spVec.Indices[k] - 1;
                    w[idx] +=(double) d * spVec.Values[k];
                }

                alpha[p] = (double)alpha_new;

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
        private static double ComputeGradient(Problem<SparseVec> sub_prob, double[] w, double[] alpha, double[] diag,ref double[] grad)
        {

            double max = double.NegativeInfinity;

            for (int i = 0; i < grad.Length; i++)
            {
                //intialized gradient i-th row
                grad[i] = 0;

                var element = sub_prob.Elements[i];
                //computes dot product between W and all elements
                double dot = 0;
                for (int k = 0; k < element.Count; k++)
                {
                    dot += w[element.Indices[k] - 1] * element.Values[k];
                }

                sbyte y_i = (sbyte)sub_prob.Y[i];
                //"minus" -gradient
                //grad[i] =1- grad[i] * y_i - alpha[i]*diag[y_i+1];

                //normal gradient

                grad[i] =(double) (dot * y_i + alpha[i] * diag[y_i + 1] - 1);
                

                //projection
                if (alpha[i] == 0)
                {
                   // grad[i] = Math.Min(0, grad[i]);
                }
                //else
                //{
                //    grad[i] = grad[i];
                //    // projGrad_i[i] = grad_i[i];
                //}

                //minus gradient - descent direction
               // grad[i] = -grad[i];

                //projected maximum norm
                if (Math.Abs(alpha[i]) > 10e-10)
                {
                    max = Math.Max(max, Math.Abs(grad[i]));
                }
                
            }

            return max;

        }

        
    }
}
