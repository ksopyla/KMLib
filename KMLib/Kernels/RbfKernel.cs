using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using dnAnalytics.LinearAlgebra;


namespace KMLib.Kernels
{
    /// <summary>
    /// Rbf kernel, use Linear kernel to compute normal dot products
    /// </summary>
    public class RbfKernel: VectorKernel<SparseVector>
    {
        public readonly float Gamma = 0.5f;

        private LinearKernel linKernel;


        public override SparseVector[] ProblemElements
        {
            set
            {
                if (value == null) throw new ArgumentNullException("value");
                linKernel.ProblemElements = value;

                base.ProblemElements = value;
                //problemVectors = value;

                //ComputeDiagonalDotCache();

            }
        }

        //public RbfKernel(float gamma,Vector[] vectors):base(vectors)
        //{
        //    Gamma = gamma;
        //    linKernel = new LinearKernel(vectors);

        //    DiagonalDotCache = new float[vectors.Length];
        //    for (int i = 0; i < DiagonalDotCache.Length; i++)
        //    {
        //        DiagonalDotCache[i] = Product(vectors[i], vectors[i]);
        //    }
        //}

        public RbfKernel(float gamma)
        {
            Gamma = gamma;
            linKernel = new LinearKernel();

        }

        #region IKernel<Vector> Members


        /// <summary>
        /// Computes rbf product, use base calass LinearKernel to compute normal dot product
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        public override float Product(SparseVector element1, SparseVector element2)
        {
            // epx(-g*|x-y|^2) =exp(-g*dot(x-y,x-y))= exp(-g*[ (x1-y1)^2+ ....(xN-yN)^2])
            //=exp(-g*( x1^2+...+xN^2 + y1^2+...+yN^2 -2x1y1+...+ -2xNyN))
            //float x1Squere = base.Product(element1, element1);
            //float x2Squere = base.Product(element2, element2);
            //float dot = base.Product(element1, element2);

            float x1Squere = linKernel.Product(element1, element1);
            float x2Squere = linKernel.Product(element2, element2);

            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));

         

            return prod;
            
        }

        #endregion

        public override float Product(int element1, int element2)
        {
            if (element1 >= problemElements.Length)
                throw new IndexOutOfRangeException("element1 out of range");

            if (element2 >= problemElements.Length)
                throw new IndexOutOfRangeException("element2 out of range");


            if (element1 == element2 && (DiagonalDotCacheBuilded))
                return DiagonalDotCache[element1];


            float x1Squere = linKernel.Product(element1, element1);
            float x2Squere = linKernel.Product(element2, element2);

            float dot = linKernel.Product(element1, element2);

            float prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));

            return prod;
        }


      
        /// <summary>
        /// Creates object for parameters selection for RBF kernel
        /// </summary>
        /// <returns></returns>
        public override ParameterSelection<SparseVector> CreateParameterSelection()
        {
            return new RbfParameterSelection();
        }

        public override string ToString()
        {
            string s = string.Format("RBF kernel, Gamma={0}", Gamma);
            return s;
        }
    }
}
