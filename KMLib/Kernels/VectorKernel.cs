using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;
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
        
        //public bool IsInitilized = false;
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
        public virtual void AllProducts(int element1, float[] results)
        {
            if (!IsInitialized)
            {
                throw new ApplicationException("Kernel not initialized");
            }
            
            
            if (results == null)
                throw new ArgumentNullException("result array should not be null");
            //for (int j = 0; j < results.Length; j++)
            //    results[j] = (Y[element1] * Y[j] * Product(element1, j));


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

        #region IKernel<T> Members


        //public float[] Predict(Model<T> model, T[] predictElements)
        //{
        //    float[]  sum = new float[predictElements.Length];

        //    int index = -1;
        //    for (int i = 0; i < predictElements.Length; i++)
        //    {


        //        for (int k = 0; k < model.SupportElementsIndexes.Length; k++)
        //        {
        //            index = model.SupportElementsIndexes[k];
                    
        //            sum[i] += model.Alpha[index] * Y[index] *
        //                                Product(problemElements[index], predictElements[i]);
        //        }


        //        sum[i] -= model.Bias;
        //        sum[i] = sum[i] > 0 ? 1 : -1;
        //    }

            

        //    return sum;
        //}

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
