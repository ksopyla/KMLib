using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
//using dnAnalytics.LinearAlgebra;

namespace KMLib.Kernels
{

    /// <summary>
    /// Represents polinominal kernel (Gamma*dot(x,y)+Coef)^deg
    /// </summary>
    public class PolinominalKernel:VectorKernel<SparseVec>
    {
        public readonly double Degree = 3;
        public readonly double Coef = 0;
        public readonly double Gamma = 0.5;


        private LinearKernel linKernel;

        public override SparseVec[] ProblemElements
        {
            get { return ProblemElements; }
            set
            {
                linKernel.ProblemElements = value;

                base.ProblemElements = value;

                //ComputeDiagonalDotCache();

            }
        }

        /// <summary>
        /// Creates Polinominal Kernel,
        /// </summary>
        /// <param name="vectors">array of vectors</param>
        /// <param name="degree">polinominal Degree</param>
        /// <param name="coef">coeficient</param>
        /// <param name="gamma">Gamma</param>
        public PolinominalKernel(double degree, double coef, double gamma)
        {
            Degree = degree;
            Coef = coef;
            Gamma = gamma;

            linKernel = new LinearKernel();

            
        }


        public override float Product(SparseVec element1, SparseVec element2)
        {
            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Pow((Gamma * dot + Coef), Degree);

            return prod;
        }

        public override float Product(int element1, int element2)
        {
            if (element1 == element2 && (DiagonalDotCacheBuilded))
                return DiagonalDotCache[element1];


            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Pow((Gamma *dot+Coef),Degree);

            return prod;
        }

        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            throw new NotImplementedException();
        }

        public override string ToString()
        {
            return string.Format("Polinominal kernel, Gamma={0}", Gamma);
        }
    }
}
