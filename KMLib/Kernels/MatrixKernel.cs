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
            get { throw new NotImplementedException(); }
            set { throw new NotImplementedException(); }
        }

        public float[] DiagonalDotCache
        {
            get; protected set;
        }

        public MatrixKernel(Matrix[] elements)
        {
            problemElements = elements;
        }

        

        public abstract float Product(Matrix element1, Matrix element2);
        public abstract float Product(int element1, int element2);

        
    }
}
