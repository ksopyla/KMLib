using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;

namespace KMLib.Evaluate
{




    /// <summary>
    /// Base class for all the evaluators, used for prediction
    /// </summary>
    /// <typeparam name="TProblemElement">The type of the problem element.</typeparam>
    public abstract class EvaluatorBase<TProblemElement>
    {
        protected bool IsInitialized = false;
        /// <summary>
        /// Trained model
        /// </summary>
        public Model<TProblemElement> TrainedModel { get; set; }

        /// <summary>
        /// Gets or sets the trainning problem.
        /// </summary>
        /// <value>The trainning problem.</value>
       // public Problem<TProblemElement> TrainningProblem { get; set; }

        public IKernel<TProblemElement> Kernel { get; set; }

        /// <summary>
        /// Predicts the class of specified elements.
        /// </summary>
        /// <param name="elements">Array with elements to predict</param>
        /// <returns></returns>
        public abstract float[] Predict(TProblemElement[] elements);



        public abstract void Init();
        /// <summary>
        /// Predicts the class of specified element.
        /// </summary>
        /// <param name="element">The element to predict.</param>
        /// <returns></returns>
        //public abstract float Predict(TProblemElement element);



        /// <summary>
        ///  Predicts the class of specified element.
        /// </summary>
        /// <param name="element">The element to predict.</param>
        /// <returns>predicted class</returns>
        public virtual  float Predict(TProblemElement element)
        {
            float sum = PredictVal(element);

           float ret = sum < 0 ? -1 : 1;


            return ret;
        }

        public virtual float PredictVal(TProblemElement element)
        {
            float sum = 0;

            int index = -1;

            for (int k = 0; k < TrainedModel.SupportElementsIndexes.Length; k++)
            {
                index = TrainedModel.SupportElementsIndexes[k];
                sum += TrainedModel.Alpha[index] * TrainedModel.Y[k] *
                                    Kernel.Product(TrainedModel.SupportElements[k], element);
                // Kernel.Product(TrainningProblem.Elements[index], element);
            }

            sum -= TrainedModel.Bias;
            return sum;
        }

    }
}
