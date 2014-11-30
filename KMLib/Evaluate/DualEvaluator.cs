using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;
using System.Threading.Tasks;

namespace KMLib.Evaluate
{
    /// <summary>
    /// represents sequential evaluation(prediction) for new unseen vector elements,
    /// It use dual form of SVM, all prediction are based on alpha coeficients in model
    /// </summary>
    /// <remarks>It is not so sequential, because it works on many CPU cores, but
    ///  "sequential" means that elements are predicted one by one
    /// </remarks>
    public class DualEvaluator<TProblemElement> : Evaluator<TProblemElement>
    {
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

            for (int i = 0; i < elements.Length; i++)
            {
                predictions[i] = Predict(elements[i]);
            }

            return predictions;
        }
    }
}
