using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;

namespace KMLib.Kernels
{

    /// <summary>
    /// Base class for vector kernels
    /// </summary>
    public abstract class VectorKernel: IKernel<Vector>
    {
        protected Vector[] problemVectors;

        public float[] DiagonalDotCache
        {
            get;
            protected set;
        }

        public VectorKernel(Vector[] vectors)
        {
            problemVectors = vectors;
        }

        #region IKernel<Vector> Members

        public abstract float Product(Vector element1, Vector element2);

        public abstract float Product(int element1, int element2);

        #endregion
    }
}
