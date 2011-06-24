using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using System.Diagnostics;

namespace KMLib.SVMSolvers
{
    public class ParallelLinSolver : LinearSolver
    {
        private Problem<SparseVec> train;
        private int scaling_step;
        private int maxInnerIter;



        public ParallelLinSolver(Problem<SparseVec> problem, float C)
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



                solve_l2r_l2_svc_parallel(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);
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

            float[] QD = new float[sub_prob.ElementsCount];

            for (int i = 0; i < sub_prob.ElementsCount; i++)
            {
                QD[i] = sub_prob.Elements[i].DotProduct();
            }

            float[] alpha = new float[sub_prob.ElementsCount];
            float[] deltas = new float[sub_prob.ElementsCount];
            float[] vals = new float[sub_prob.ElementsCount];
            float[] diag = new float[] { (float)(0.5 / Cn), 0, (float)(0.5 / Cp) };

            //gradient and previous gradient
            float[] projGrad = new float[sub_prob.ElementsCount];
            float[] oldGrad;

            //directional vecotr
            float[] dir = new float[sub_prob.ElementsCount];
            
            ComputeGradient(sub_prob, w, alpha, diag, projGrad);

            //copy gradient values to dir vector
            Buffer.BlockCopy(projGrad, 0, dir, 0, projGrad.Length);

            int iter = 0;

            scaling_step = 10;
            maxInnerIter = 10;


            Stopwatch st = new Stopwatch();
            st.Start();

            while (iter < maxIter)
            {

                float step = ComputeLineStep(dir, w, alpha);


                #region computing steps and vals
                

                for (int p = 0; p < sub_prob.ElementsCount; p++)
                {
                    sbyte y_i = (sbyte)sub_prob.Y[p];
                    float old_alpha = alpha[p];

                    float alphaStep = 0;


                    float qii = QD[p] + diag[y_i + 1];

                    //alpha[p] = Math.Max(alpha[p] - projGrad[p] /qii, 0);

                    deltas[p] = Math.Max(alpha[p] - projGrad[p] / qii, 0) - alpha[p];
                    vals[p] = -projGrad[p] * projGrad[p] / (2 * qii);
                }
                #endregion

                #region find min val
                
                float min = float.PositiveInfinity;
                int minIdx = -1;
                for (int i = 0; i < vals.Length; i++)
                {
                    if (vals[i] < min)
                    {
                        min = vals[i];
                        minIdx = i;
                    }

                }
                #endregion


                #region update func in min val direction
                
                var spVec = sub_prob.Elements[minIdx];
                float d = deltas[minIdx];

                alpha[minIdx] = alpha[minIdx] + d;

                d = d * sub_prob.Y[minIdx];

                for (int k = 0; k < spVec.Count; k++)
                {
                    w[spVec.Indices[k] - 1] += d * spVec.Values[k];
                }
                #endregion



#if DEBUG
               // obj = ComputeObj(w, alpha, sub_prob, diag);

#endif

                //float minPG = float.PositiveInfinity;
                //float maxPG = float.NegativeInfinity;
                //for (int i = 0; i < projGrad.Length; i++)
                //{
                //    minPG = Math.Min(minPG, projGrad[i]);
                //    maxPG = Math.Max(maxPG, projGrad[i]);
                //}
                //if (maxPG < 0)
                //    maxPG = float.NegativeInfinity;
                //if (minPG > 0)
                //    minPG = float.PositiveInfinity;

                //if (Math.Abs(maxPG - minPG) <= epsilon)
                //    break;

                iter++;
            }

            st.Stop();
           obj= ComputeObj(w, alpha, sub_prob, diag);
            //int l = sub_prob.ElementsCount;// prob.l;
            //int w_size = sub_prob.FeaturesCount;// prob.n;
            //double v = 0;
            //int nSV = 0;
            //for (int i = 0; i < w_size; i++)
            //    v += w[i] * w[i];
            //for (int i = 0; i < l; i++)
            //{
            //    sbyte y_i =(sbyte) sub_prob.Y[i];
            //    v += alpha[i] * (alpha[i] * diag[y_i+1] - 2);
            //    if (alpha[i] > 0) ++nSV;
            //}


            Console.WriteLine("Objective value = {0} time={1} ms={2}", obj,st.Elapsed, st.ElapsedMilliseconds);
            //Debug.WriteLine("nSV = {0}", nSV);



        }

        /// <summary>
        /// Compute step in <see cref="dir"/> direction, 
        /// it moves by some step in dir direction, computes min function value
        /// and returns step for min value
        /// </summary>
        /// <param name="dir"></param>
        /// <param name="w"></param>
        /// <param name="alpha"></param>
        /// <returns></returns>
        private float ComputeLineStep(float[] dir, float[] w, float[] alpha)
        {
            // max vector length should be 1, so if it is longer we take 1
            
            //todo: compute dir norm
            float dirNorm = 1.0f;
            
            float maxVecLen = Math.Max(1, dirNorm);

            float prevVal = 0;
            float step = 0f;

            for (int i = 0; i < maxInnerIter; i++)
            {
                //small step
                step = i / (scaling_step * maxVecLen);

                float[] step_alpha = new float[alpha.Length];
                Buffer.BlockCopy(alpha, 0, step_alpha, 0, alpha.Length);

                //update alpha solution by step*dir
                // xx=xi+st* dir;
                for (int k = 0; k < dir.Length; k++)
                {
                    step_alpha[k] += step * dir[k];
                }



            }
        }

        private static void ComputeGradient(Problem<SparseVec> sub_prob, float[] w, float[] alpha, float[] diag, float[] projGrad)
        {
            //computes dot product between W and all elements
            #region computing gradient

            float[] dots = new float[sub_prob.ElementsCount];

            for (int i = 0; i < dots.Length; i++)
            {

                var element = sub_prob.Elements[i];
                for (int k = 0; k < element.Count; k++)
                {
                    dots[i] += w[element.Indices[k] - 1] * element.Values[k];

                }
                dots[i] = dots[i] * sub_prob.Y[i] - 1;

            }



            for (int i = 0; i < dots.Length; i++)
            {
                sbyte y_i = (sbyte)sub_prob.Y[i];
                dots[i] += alpha[i] * diag[y_i + 1];

                if (alpha[i] == 0)
                {
                    projGrad[i] = Math.Min(0, dots[i]);
                }
                else
                {
                    projGrad[i] = dots[i];
                    // projGrad_i[i] = grad_i[i];
                }

            }
            #endregion
        }

        private double ComputeObj(float[] w, float[] alpha, Problem<SparseVec> sub_prob, float[] diag)
        {
            double v = 0, v1 = 0;
            int nSV = 0;
            for (int i = 0; i < w.Length; i++)
            {
                v += w[i] * w[i];
                //v1 += 0.5 * w[i] * w[i];
            }
            for (int i = 0; i < alpha.Length; i++)
            {
                sbyte y_i = (sbyte)sub_prob.Y[i];

                //original line
                //v += alpha[i] * (alpha[i] * diag[GETI(y_i, i)] - 2);
                v += alpha[i] * (alpha[i] * diag[y_i + 1] - 2);
               // v1 += 0.5 * alpha[i] * (alpha[i] * diag[y_i + 1] - 2);
                if (alpha[i] > 0) ++nSV;
            }

            v = v / 2;
            //  Debug.WriteLine("Objective value = {0}", v);
            //  Debug.WriteLine("nSV = {0}", nSV);

            return v;
        }
    }
}
