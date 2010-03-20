using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;

namespace KMLib.Kernels
{
    public abstract class MatrixKernel: IKernel<Matrix>
    {

        protected Matrix[] problemElements;

        public Matrix[] ProblemElements
        {
            get { return problemElements; }
            set
            {
                problemElements = value;
                ComputeDiagonalDotCache();
            }
        }

        protected void ComputeDiagonalDotCache()
        {
            DiagonalDotCache = new float[ProblemElements.Length];
            for (int i = 0; i < DiagonalDotCache.Length; i++)
            {
                DiagonalDotCache[i] = Product(ProblemElements[i], ProblemElements[i]);
            }
        }


        public float[] DiagonalDotCache
        {
            get; protected set;
        }


        public abstract float Product(Matrix element1, Matrix element2);
        public abstract float Product(int element1, int element2);

        
    }
}
