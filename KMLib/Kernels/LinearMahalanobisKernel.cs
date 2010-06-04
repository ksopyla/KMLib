using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using KMLib.Helpers;
using CenterSpace.NMath.Matrix;

namespace KMLib.Kernels
{

    /// <summary>
    /// Computes Linear Mahalanobis Kernel 
    /// 
    /// </summary>
    /// <remarks>
    ///  K(x,y)= X'* invC*Y
    /// where
    /// X,Y - are a feature vector
    /// invC - is inverted covariance Matrix
    /// </remarks>
    public class LinearMahalanobisKernel<T>: VectorKernel<T> where T:Vector
    {
        /// <summary>
        /// Inverted covariance matrix
        /// </summary>
        private FloatSymmetricMatrix invertedCovMatrix;

      

        /// <summary>
        /// Construct kernel, vectors are nesseery for some precomputing eg. DiagonalDotCache and
        /// for easiest Product computing  
        /// </summary>
        /// <param name="vectors"></param>
        public LinearMahalanobisKernel(FloatSymmetricMatrix invertedCovarianceMatrix)
        {
          //  problemVectors = vectors;


            invertedCovMatrix = invertedCovarianceMatrix;

        }



        #region IKernel<Vector> Members

        /// <summary>
        /// virtual method because other kernels can inherit from linear 
        /// in other kernels we have to comput normal dot product (linear)
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        public override float Product(T element1, T element2)
        {
           // return  (float) element1.DotProduct(element2);

            
            var iter1 = element1.GetIndexedEnumerator().GetEnumerator();
            var iter2 = element2.GetIndexedEnumerator().GetEnumerator();
            double prod = 0;

            bool goForward1 = iter1.MoveNext();
            bool goForward2= iter2.MoveNext();
            
            //if vectors contains all zeros
            if (! (goForward1&&goForward2))
                return 0f;

            //
            while (goForward1)
            {
                var cur1 = iter1.Current;
                int index1 = cur1.Key;

                while (goForward2)
                {
                    var cur2 = iter2.Current;
                    int index2 = cur2.Key;

                    prod += cur1.Value*invertedCovMatrix[index1, index2]*cur2.Value;

                    goForward2 = iter2.MoveNext();
                }
                
                //throw an Exception "NotSuportedException" he?
                //iter2.Reset();
                iter2 = element2.GetIndexedEnumerator().GetEnumerator();
                goForward2 = iter2.MoveNext();

                goForward1 = iter1.MoveNext();
            }


            return (float) prod;
 
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
            if (element1 == element2 && (DiagonalDotCache!=null))
                return DiagonalDotCache[element1];


            return this.Product(ProblemElements[element1], ProblemElements[element2]);
            
        }

        public override ParameterSelection<T> CreateParameterSelection()
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
