using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using KMLib.Helpers;

namespace KMLib.Kernels
{
    public class LinearKernel: VectorKernel<SparseVector> 
    {
       // protected Vector[] problemVectors;

       // public float[] DiagonalDotCache { get; private set; }

        private LRUCache<Point2D, float> cache;
        
        /// <summary>
        /// decide if we use cache, for easy kernles and sparse vectors
        /// cache is slower than computing Product, but when vectors are dense and big, cache may improve
        /// computation
        /// </summary>
        private bool useCache;


        /// <summary>
        /// Construct kernel, vectors are nesseery for some precomputing eg. DiagonalDotCache and
        /// for easiest Product computing  
        /// </summary>
        /// <param name="vectors"></param>
        //public LinearKernel(Vector[] vectors):base(vectors)
        //{
        //  //  problemVectors = vectors;

        //    DiagonalDotCache=new float[vectors.Length];
        //    for (int i = 0; i < DiagonalDotCache.Length; i++)
        //    {
        //        DiagonalDotCache[i] = this.Product(vectors[i], vectors[i]);
        //    }

        //    useCache = false;
        //}

        public LinearKernel() 
        {
            useCache = false;
        }


        /// <summary>
        /// constructor with cache construction
        /// </summary>
        /// <param name="vectors"></param>
        /// <param name="cacheSize"></param>
        //public LinearKernel(Vector[] vectors,int cacheSize):this(vectors)
        //{
        //    useCache = true;
        //    cache = new LRUCache<Point2D, float>(cacheSize);
        //}

        #region IKernel<Vector> Members

        /// <summary>
        /// virtual method because other kernels can inherit from linear 
        /// in other kernels we have to comput normal dot product (linear)
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        
        
        public override float Product(SparseVector element1, SparseVector element2)
        {
            return  (float) element1.DotProduct(element2);

            //todo: test which version is faster
             /*
            var iter1 = element1.GetIndexedEnumerator().GetEnumerator();
            var iter2 = element2.GetIndexedEnumerator().GetEnumerator();
            float prod = 0;

            bool go = iter1.MoveNext() && iter2.MoveNext();
            if (!go)
                return prod;


            while (go)
            {
                var cur1 = iter1.Current;
                var cur2 = iter2.Current;

                int index1 = cur1.Key;
                int index2 = cur2.Key;

                if (index1 == index2)
                {
                    prod += (float)(cur1.Value * cur2.Value);

                    //go to next elements
                    go = iter1.MoveNext() && iter2.MoveNext();
                }
                else
                {
                    if (index1 < index2)
                        go = iter1.MoveNext();
                    else
                        go = iter2.MoveNext();
                }
            }



            return prod;
             */
 
           /*

            * old implementation witch uses DataStructures.Vector
            float sum = 0;

            uint i1=0, i2=0;

            while (i1<element1.Data.Length && i2<element2.Data.Length)
            {
                int index1 = element1.Data[i1].Index;
                int index2 = element2.Data[i2].Index;

                if(index1==index2)
                {
                    sum += element1.Data[i1].Value * element2.Data[i2].Value;
                    i1++;
                    i2++;
                }else
                {
                    if (index1 < index2)
                        i1++;
                    else
                        i2++;
                }
                
            }
            return sum;
            */ 
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        public override float  Product(int element1, int element2)
        {
            if (element1 >= ProblemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= ProblemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");


            if (element1 == element2 && (DiagonalDotCacheBuilded))
                return DiagonalDotCache[element1];

            if (!useCache)
            {
                return this.Product(ProblemElements[element1], ProblemElements[element2]);
            }

            Point2D indexes;

            int x=element1, y=element2;
            //we ensure that Point is created in that way X<Y, product is symetric
            //so it don't change the value, but it Point.GetHashCode() compute same hash for cache
            if(element2<element1)
            {
                x = element1;
                y = element2;
            }
            indexes = new Point2D(x,y);

            float prod = 0;
            
            if (cache.ContainsKey(indexes))
            {
                //if we compute produce earlier, then we retriv it from cache
                prod = cache[indexes];
               // Console.WriteLine("{0}-{1}={2} from cache", x, y, prod);
            }
            else
            {
                //
                prod = this.Product(ProblemElements[x], ProblemElements[y]);
                cache.Add(indexes,prod);
                //Console.WriteLine("{0}-{1}={2}", x, y, prod);
               
            }
           
            return prod;
        }

        public override ParameterSelection<SparseVector> CreateParameterSelection()
        {
            return new LinearParameterSelection();
        }

        #endregion

        public override string ToString()
        {
            return "Linear Kernel";
        }
    }
}
