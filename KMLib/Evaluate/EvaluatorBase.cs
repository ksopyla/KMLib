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
    public abstract class EvaluatorBase<TProblemElement>
    {
        /// <summary>
        /// Trained model
        /// </summary>
        public Model<TProblemElement> TrainedModel { get; set; }

        public Problem<TProblemElement> TrainningProblem { get; set; }

        public IKernel<TProblemElement> Kernel { get; set; }

        public abstract float[] Predict(TProblemElement[] elements);

        public abstract float Predict(TProblemElement element);

    }
}
