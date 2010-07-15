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
    public abstract class VectorKernel<T>: IKernel<T> where T:Vector
    {

        protected bool DiagonalDotCacheBuilded = false;
       
        protected T[] problemElements;
        public virtual T[] ProblemElements
        {
            get { return problemElements; }
            set
            {
                DiagonalDotCacheBuilded = false;
                problemElements = value;
                ComputeDiagonalDotCache();

            }
        }

        protected void ComputeDiagonalDotCache()
        {
            DiagonalDotCache = new float[problemElements.Length];
            for (int i = 0; i < DiagonalDotCache.Length; i++)
            {
              // DiagonalDotCache[i] = Product(ProblemElements[i], ProblemElements[i]);

              DiagonalDotCache[i] = Product(i,i);
            }
            DiagonalDotCacheBuilded = true;
        }

        public float[] DiagonalDotCache
        {
            get;
            protected set;
        }

        //public VectorKernel(Vector[] vectors)
        //{
        //    problemVectors = vectors;
        //}


        #region IKernel<Vector> Members



        public abstract float Product(T element1, T element2);

        public abstract float Product(int element1, int element2);

        #endregion


        public abstract ParameterSelection<T> CreateParameterSelection();
    }
}
