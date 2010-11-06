using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;

namespace KMLib.GPU
{
    public class CudaEvaluator: EvaluatorBase<SparseVector>
    {


        public override float[] Predict(SparseVector[] elements)
        {
            throw new NotImplementedException();
        }

        public override float Predict(SparseVector element)
        {
            throw new NotImplementedException();
        }
    }
}
