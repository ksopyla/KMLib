using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    /// Base class for vector kernels
    /// </summary>
    public abstract class VectorKernel<T>: IKernel<T> where T:Vector
    {

        protected bool DiagonalDotCacheBuilded = false;

        public bool IsInitialized
        {
            get;
            protected set;
        }

        public float[] Y
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
              
                //insted of computing Diagonal Dot Cache here 
                //it is done in Init() method
                //  ComputeDiagonalDotCache();
            }
        }

        protected void ComputeDiagonalDotCache()
        {
            DiagonalDotCache = new float[problemElements.Length];
            DiagonalDotCacheBuilded = false;
            for (int i = 0; i < DiagonalDotCache.Length; i++)
            {
                DiagonalDotCache[i] = Product(i,i);
            }
            DiagonalDotCacheBuilded = true;
        }

        public float[] DiagonalDotCache
        {
            get;
            protected set;
        }

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
        public virtual void AllProducts(int element1, float[] results)
        {
            if (!IsInitialized)
            {
                throw new ApplicationException("Kernel not initialized");
            }
            
            
            if (results == null)
                throw new ArgumentNullException("result array should not be null");

            var partition = Partitioner.Create(0, results.Length);

            Parallel.ForEach(partition, (range) =>
            {
                for (int k = range.Item1; k < range.Item2; k++)
                {
                    results[k] = (Y[element1] * Y[k] * Product(element1, k));
                }

            });

           

        }

        #endregion



        #region IKernel<T> Members


        public virtual void Init()
        {
            ComputeDiagonalDotCache();

            IsInitialized = true;
        }

        #endregion

        public virtual void SwapIndex(int i, int j)
        {
            Y.SwapIndex(i, j);
            DiagonalDotCache.SwapIndex(i, j);
            problemElements.SwapIndex(i, j);
        }


        public virtual void AllProducts(int i, int j, float[][] results)
        {
            AllProducts(i, results[0]);
            AllProducts(j, results[1]);
        }
    }
}
