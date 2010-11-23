using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using System.Threading.Tasks;

namespace KMLib.Evaluate
{
    /// <summary>
    /// represents sequential evaluation(prediction) for new unseen vector elements,
    /// </summary>
    /// <remarks>It is not so sequential, because it works on many CPU cores, but
    /// word sequential means that elements are predicted one by one
    /// </remarks>
    public class SequentialEvaluator<TProblemElement> : EvaluatorBase<TProblemElement>
    {
        
        //#region IEvaluator<SparseVector> Members

        ///// <summary>
        ///// Predicts the class of specified elements.
        ///// </summary>
        ///// <remarks>Computes this on many CPU cores</remarks>
        ///// <param name="elements">Array with elements to predict.</param>
        ///// <returns>array with predicted class for each element</returns>
        //public override float[] Predict(SparseVector[] elements)
        //{
        //    float[] predictions = new float[elements.Length];

        //    Parallel.For(0, elements.Length, i =>
        //    {

        //        predictions[i] = Predict(elements[i]);
        //    });

        //    return predictions;
        //}

        ///// <summary>
        ///// Predicts the specified element.
        ///// </summary>
        ///// <param name="element">The element.</param>
        ///// <returns>predicted class</returns>
        //public override float Predict(SparseVector element)
        //{
        //    float sum = 0;

        //    int index = -1;

        //    for (int k = 0; k < TrainedModel.SupportElementsIndexes.Length; k++)
        //    {
        //        index = TrainedModel.SupportElementsIndexes[k];
        //        sum += TrainedModel.Alpha[index] * TrainningProblem.Labels[index] *
        //                            Kernel.Product(TrainningProblem.Elements[index], element);
        //    }

        //    sum -= TrainedModel.Rho;

        //    float ret = sum > 0 ? 1 : -1;
        //    return ret;
        //}

        //#endregion

        public override void Init()
        {
            //we don't have to initialized anything
            IsInitialized = true;
        }
        /// <summary>
        /// Predicts the class of specified elements.
        /// </summary>
        /// <remarks>Computes this on many CPU cores</remarks>
        /// <param name="elements">Array with elements to predict.</param>
        /// <returns>array with predicted class for each element</returns>
        public override float[] Predict(TProblemElement[] elements)
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

        /// <summary>
        /// Predicts the specified element.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <returns>predicted class</returns>
        //public override float Predict(TProblemElement element)
        //{
        //    float sum = 0;

        //    int index = -1;

        //    for (int k = 0; k < TrainedModel.SupportElementsIndexes.Length; k++)
        //    {
        //        index = TrainedModel.SupportElementsIndexes[k];
        //        sum += TrainedModel.Alpha[index] * TrainningProblem.Labels[index] *
        //                            Kernel.Product(TrainningProblem.Elements[index], element);
        //    }

        //    sum -= TrainedModel.Rho;

        //    float ret = sum > 0 ? 1 : -1;
        //    return ret;
        //}
    }
}
