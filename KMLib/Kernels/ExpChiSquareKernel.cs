using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    ///  Exponential Chi Square Kernel
    /// K(x,y)  = exp( -gamma * ChiSquareDistance(x,y))
    /// ChiSquareDistance(x,y)= Sum( (xi-yi)^2/(xi+yi))
    /// </summary>
    public class ExpChiSquareKernel : VectorKernel<SparseVec>, IDisposable
    {

        public readonly float Gamma = 0.5f;


        public ExpChiSquareKernel(float gamma)
        {
            Gamma = gamma;
        }


        public override float Product(SparseVec element1, SparseVec element2)
        {
            float chiSquare =2* ChiSquareKernel.ChiSquareDist(element1, element2);

            return (float)Math.Exp(-Gamma * chiSquare);
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
            return String.Format("Exponential Chi-Square, gamma={0}",Gamma);
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
