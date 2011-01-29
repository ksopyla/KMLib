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
    public class LinearPrimalEvaluator : EvaluatorBase<SparseVec>
    {
        public override float[] Predict(SparseVec[] elements)
        {
            if (!IsInitialized)
                throw new ApplicationException("Evaluator is not initialized. Call init method");

            float[] predictions = new float[elements.Length];

            Parallel.For(0, elements.Length, i =>
            {

                predictions[i] = Predict(elements[i]);
            });

            //for (int i = 0; i < elements.Length; i++)
            //{
            //    predictions[i] = Predict(elements[i]);
            //}

            return predictions;
        }

        public override void Init()
        {
            IsInitialized = true;
        }

        public new float Predict(SparseVec element)
        {

            int n = TrainedModel.FeaturesCount;

            //if (model.bias >= 0)
            //    n = model.nr_feature + 1;
            //else
            //    n = model.nr_feature;

            double[] w = TrainedModel.W;
            double[] dec_values = new double[TrainedModel.NumberOfClasses];


            int nr_w;
            if (TrainedModel.NumberOfClasses == 2)// && TrainedModel.solverType != SolverType.MCSVM_CS)
                nr_w = 1;
            else
                nr_w = TrainedModel.NumberOfClasses;

            for (int i = 0; i < nr_w; i++)
                dec_values[i] = 0;

            //for (FeatureNode lx : x) {
            //    int idx = lx.index;
            //    // the dimension of testing data may exceed that of training
            //    if (idx <= n) {
            //        for (int i = 0; i < nr_w; i++) {
            //            dec_values[i] += w[(idx - 1) * nr_w + i] * lx.value;
            //        }
            //    }
            //}

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

            
            if (TrainedModel.NumberOfClasses == 2)
            {
                //odwróciłem znak
                return (dec_values[0] < 0) ? TrainedModel.Labels[0] : TrainedModel.Labels[1];
            }
            else
            {
                int dec_max_idx = 0;
                for (int i = 1; i < TrainedModel.NumberOfClasses; i++)
                {
                    if (dec_values[i] > dec_values[dec_max_idx]) dec_max_idx = i;
                }
                return TrainedModel.Labels[dec_max_idx];// model.label[dec_max_idx];
            }

            return float.NegativeInfinity;
        }

    }
}
