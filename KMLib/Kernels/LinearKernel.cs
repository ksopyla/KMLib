using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;

namespace KMLib.Kernels
{


    /// <summary>
    /// Represents Linear Kernel for computing linear product between 
    /// two sparse vectors
    /// </summary>
    public class LinearKernel: VectorKernel<SparseVec> , IDisposable
    {


        /// <summary>
        /// Cache for computed products
        /// </summary>
       // private LRUCache<Point2D, float> cache;
        
        /// <summary>
        /// decide if we use cache, for easy kernles and sparse vectors
        /// cache is slower than computing Product, but when vectors are dense and big, cache may improve
        /// computation
        /// </summary>
       // private bool useCache;




        /// <summary>
        /// Initializes a new instance of the <see cref="LinearKernel"/> class.
        /// </summary>
        public LinearKernel() 
        {
           // useCache = false;
        }


        #region IKernel<Vector> Members

        /// <summary>
        /// virtual method because other kernels can inherit from linear 
        /// in other kernels we have to comput normal dot product (linear)
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns>linear product between elements</returns>
        public override float Product(SparseVec element1, SparseVec element2)
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
        /// compute linear kernel between two element given by indexes
        /// </summary>
        /// <param name="element1">index of element in <see cref="ProblemElements"/> array</param>
        /// <param name="element2">index of element in <see cref="ProblemElements"/> array</param>
        /// <returns>linear product between elements</returns>
        public override float  Product(int element1, int element2)
        {
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");

            if (problemElements == null)
                throw new ApplicationException("Problem elements are null");

            if (element1 == element2)
            {
                return problemElements[element1].DotProduct();
                //return DiagonalDotCache[element1];
            }

            //if (element1 == element2 && (DiagonalDotCacheBuilded))
            //{
            //    return DiagonalDotCache[element1];
            //}
            return this.Product(problemElements[element1], problemElements[element2]);
 /*
            if (!useCache)
            {
                return this.Product(problemElements[element1], problemElements[element2]);
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
                prod = this.Product(problemElements[x], problemElements[y]);
                cache.Add(indexes,prod);
                //Console.WriteLine("{0}-{1}={2}", x, y, prod);
               
            }
           return prod;
           */
            
        }


        /// <summary>
        /// Creates the parameter selection class for finding the best parameter for 
        /// this kernel, it finds only SVM penalty "C" parameter.
        /// </summary>
        /// <returns>Instance of parameter selection class</returns>
        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            return new LinearParameterSelection();
        }

        #endregion

        /// <summary>
        /// Returns a <see cref="System.String"/> that represents this instance.
        /// </summary>
        /// <returns>
        /// A <see cref="System.String"/> that represents this instance.
        /// </returns>
        public override string ToString()
        {
            return "Linear Kernel";
        }

        public void Dispose()
        {
            problemElements = null;
            Y = null;
            DiagonalDotCache = null;
            IsInitialized = false;
        }
    }
}
