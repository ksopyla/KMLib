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
        /// <summary>
        /// Trained model
        /// </summary>
        public Model<TProblemElement> TrainedModel { get; set; }

        /// <summary>
        /// Gets or sets the trainning problem.
        /// </summary>
        /// <value>The trainning problem.</value>
        public Problem<TProblemElement> TrainningProblem { get; set; }

        public IKernel<TProblemElement> Kernel { get; set; }

        /// <summary>
        /// Predicts the class of specified elements.
        /// </summary>
        /// <param name="elements">Array with elements to predict</param>
        /// <returns></returns>
        public abstract float[] Predict(TProblemElement[] elements);

        /// <summary>
        /// Predicts the class of specified element.
        /// </summary>
        /// <param name="element">The element to predict.</param>
        /// <returns></returns>
        public abstract float Predict(TProblemElement element);

    }
}
