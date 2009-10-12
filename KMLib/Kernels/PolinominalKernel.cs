using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;

namespace KMLib.Kernels
{

    /// <summary>
    /// Represents polinominal kernel (Gamma*dot(x,y)+Coef)^deg
    /// </summary>
    public class PolinominalKernel:VectorKernel
    {
        public readonly double Degree = 3;
        public readonly double Coef = 0;
        public readonly double Gamma = 0.5;


        private LinearKernel linKernel;


        /// <summary>
        /// Creates Polinominal Kernel,
        /// </summary>
        /// <param name="vectors">array of vectors</param>
        /// <param name="degree">polinominal Degree</param>
        /// <param name="coef">coeficient</param>
        /// <param name="gamma">Gamma</param>
        public PolinominalKernel(double degree, double coef, double gamma, Vector[] vectors)
            : base(vectors)
        {
            Degree = degree;
            Coef = coef;
            Gamma = gamma;

            linKernel = new LinearKernel(vectors);

            DiagonalDotCache = new float[vectors.Length];
            for (int i = 0; i < DiagonalDotCache.Length; i++)
            {
                DiagonalDotCache[i] = Product(vectors[i], vectors[i]);
            }


        }


        public override float Product(Vector element1, Vector element2)
        {
            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Pow((Gamma * dot + Coef), Degree);

            return prod;
        }

        public override float Product(int element1, int element2)
        {
            if (element1 == element2 && (DiagonalDotCache != null))
                return DiagonalDotCache[element1];


            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Pow((Gamma *dot+Coef),Degree);

            return prod;
        }
    }
}
