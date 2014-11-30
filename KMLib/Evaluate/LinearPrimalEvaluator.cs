using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using System.Threading.Tasks;

namespace KMLib.Evaluate
{

    /// <summary>
    /// Evaluator for linear SVM
    /// Predict based on primal
    /// </summary>
    public class LinearPrimalEvaluator : Evaluator<SparseVec>
    {
        public override float[] Predict(SparseVec[] elements)
        {
            if (!IsInitialized)
                throw new ApplicationException("Evaluator is not initialized. Call init method");

            float[] predictions = new float[elements.Length];

            for (int i = 0; i < elements.Length; i++)
            {
                predictions[i] = Predict(elements[i]);
            }

            return predictions;
        }

        public override void Init()
        {
            IsInitialized = true;
        }

        public new float Predict(SparseVec element)
        {

            double[] dec_values = ComputeDecisions(element);

            
            if (TrainedModel.NumberOfClasses == 2)
            {
                //add multiplication by first label
                var lab0 = TrainedModel.Labels[0];
                
                return (dec_values[0] > 0) ? TrainedModel.Labels[0] : TrainedModel.Labels[1];
            }
            else
            {
                int dec_max_idx = 0;
                for (int i = 1; i < TrainedModel.NumberOfClasses; i++)
                {
                    if (dec_values[i] > dec_values[dec_max_idx]) dec_max_idx = i;
                }
                return TrainedModel.Labels[dec_max_idx];
            }
        }

        private double[] ComputeDecisions(SparseVec element)
        {
            double[] w = TrainedModel.W;
            double[] dec_values = new double[TrainedModel.NumberOfClasses];


            int nr_w;
            if (TrainedModel.NumberOfClasses == 2)
                nr_w = 1;
            else
                nr_w = TrainedModel.NumberOfClasses;

            for (int i = 0; i < nr_w; i++)
                dec_values[i] = 0;

            int n = TrainedModel.FeaturesCount;
            for (int k = 0; k < element.Count; k++)
            {
                int idx = element.Indices[k];
                if (idx <= n)
                {
                    for (int i = 0; i < nr_w; i++)
                    {
                        dec_values[i] += w[(idx - 1) * nr_w + i] * element.Values[k];
                    }
                }

            }
            return dec_values;
        }

        public override float PredictVal(SparseVec element)
        {
            //return only first value of decision
            return (float)ComputeDecisions(element)[0];
        }

    }
}
