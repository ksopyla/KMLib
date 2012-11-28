using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    /// Represents Chi^2 Kernel for computing product between two histograms
    /// 
    /// K(x,y)= 1 -2* Sum( (xi-yi)^2/(xi+yi))
    /// 
    /// vectors should contains positive numbers(like histograms does) and should be normalized
    /// sum(xi)=1
    /// </summary>
    public class ChiSquaredKernel : VectorKernel<SparseVec>, IDisposable
    {
        public override float Product(SparseVec element1, SparseVec element2)
        {

            float result = ChiSquareDist(element1, element2);
            return 1.0f-2*result;

        }


        /// <summary>
        /// Computes Chi-Square distance
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        public static float ChiSquareDist(SparseVec element1, SparseVec element2)
        {
            double result = 0.0f;
            int i1 = 0, i2 = 0;
            while (i1 < element1.Count && i2 < element2.Count)
            {
                int idx1 = element1.Indices[i1];
                int idx2 = element2.Indices[i2];

                if (idx1 == idx2)
                {
                    float div = element1.Values[i1] - element2.Values[i2];
                    float sum = element1.Values[i1] + element2.Values[i2];

                    result += div * div / sum;

                    i1++;                     
                    i2++;
                }
                else if (idx1 < idx2)
                {
                    // xi!=0 yi=0 than
                    //  (xi-yi)^2/(xi-yi)=(xi-0)^2/(xi+0)=xi^2/xi=xi;
                    result += element1.Values[i1];
                    i1++;
                }
                else
                {
                    // xi=0 yi!=0 than
                    //  (0-yi)^2/(0+yi)=yi^2/yi=yi;
                    result += element2.Values[i2];
                    i2++;
                }
            }

            while (i1 < element1.Count)
                result += element1.Values[i1++];

            while (i2 < element2.Count)
                result += element2.Values[i2++];


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

            if (element1 == element2)
                return 1.0f;

            return this.Product(problemElements[element1], problemElements[element2]);
        }

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
        }

        public override string ToString()
        {
            return "Chi Square Kernel";
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
