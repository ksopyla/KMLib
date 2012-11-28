using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    /// Represents Chi^2 Kernel for computing product between two histograms l1 normalised histograms
    /// 
    /// K(x,y)= Sum( (xi*yi)/(xi+yi))
    /// 
    /// vectors should contains positive numbers(like histograms does) and should be normalized
    /// sum(xi)=1
    /// </summary>
    public class ChiSquaredNormKernel : VectorKernel<SparseVec>, IDisposable
    {
        public override float Product(SparseVec element1, SparseVec element2)
        {

            float result = ChiSquareNormDist(element1, element2);
            return result;

        }


        /// <summary>
        /// Computes Chi-Square distance
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        public static float ChiSquareNormDist(SparseVec element1, SparseVec element2)
        {
            double result = 0.0f;
            int i1 = 0, i2 = 0;
            while (i1 < element1.Count && i2 < element2.Count)
            {
                int idx1 = element1.Indices[i1];
                int idx2 = element2.Indices[i2];

                if (idx1 == idx2)
                {
                    float mul = element1.Values[i1] * element2.Values[i2];
                    float sum = element1.Values[i1] + element2.Values[i2];

                    result += mul / sum;

                    i1++;                     
                    i2++;
                }
                else if (idx1 < idx2)
                {
                    i1++;
                }
                else
                {
                    i2++;
                }
            }

            return (float)result;
        }

        public override float Product(int element1, int element2)
        {
            
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");

            if (problemElements == null)
                throw new ApplicationException("Problem elements are null");

            return this.Product(problemElements[element1], problemElements[element2]);
        }

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
        }

        public override string ToString()
        {
            return "Chi Squared Norm Kernel";
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
