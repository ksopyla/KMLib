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
        protected bool Initilized = false;

        public float[] Labels
        {
            get;
            set;
        }

        protected T[] problemElements;
       
        public virtual T[] ProblemElements
        {
            get { return problemElements; }
            set
            {
                DiagonalDotCacheBuilded = false;
                problemElements = value;
              //  ComputeDiagonalDotCache();

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

        #region IKernel<T> Members

        /// <summary>
        /// computes product between <see cref="element1"/> and rest of vectors
        /// </summary>
        /// <param name="element1"></param>
        /// <returns></returns>
        public virtual void AllProducts(int element1,ref float[] results)
        {
            if (!Initilized)
            {
                throw new ApplicationException("Kernel not initialized");
            }
            
            
            if (results == null)
                throw new ApplicationException("result array should not be null");
            for (int j = 0; j < results.Length; j++)
                results[j] = (Labels[element1] * Labels[j] * Product(element1, j));


            //var partition = Partitioner.Create(0, problem.ElementsCount);

            //Parallel.ForEach(partition, (range) =>
            //{
            //    for (int k = range.Item1; k < range.Item2; k++)
            //    {
            //        data[k] = (y[i] * y[k] * kernel.Product(i, k));
            //    }

            //});

           // return data;

        }

        #endregion



        #region IKernel<T> Members


        public virtual void Init()
        {
            ComputeDiagonalDotCache();

            Initilized = true;
        }

        #endregion
    }
}
