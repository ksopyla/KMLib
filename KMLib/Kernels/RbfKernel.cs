using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using KMLib.Helpers;
//using dnAnalytics.LinearAlgebra;


namespace KMLib.Kernels
{
    /// <summary>
    /// Rbf kernel, compute product 
    /// K(x,y)  = exp( ||x-y||^2/gamma)
    /// </summary>
    /// <remarks> use Linear kernel to compute normal dot products</remarks>
    public class RbfKernel: VectorKernel<SparseVec>, IDisposable
    {
        public readonly float Gamma = 0.5f;

        private LinearKernel linKernel;


        public override SparseVec[] ProblemElements
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


        /// <summary>
        /// Initializes a new instance of the <see cref="RbfKernel"/> class.
        /// </summary>
        /// <param name="gamma">The gamma parameter .</param>
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
        public override float Product(SparseVec element1, SparseVec element2)
        {
            // epx(-g*|x-y|^2) =exp(-g*dot(x-y,x-y))= exp(-g*[ (x1-y1)^2+ ....(xN-yN)^2])
            //=exp(-g*( x1^2+...+xN^2 + y1^2+...+yN^2 -2x1y1+...+ -2xNyN))
            
            //float x1Squere = linKernel.Product(element1, element1);
            //float x2Squere = linKernel.Product(element2, element2);

            float x1Squere =(float)element1.DotProduct();
            float x2Squere =(float)element2.DotProduct();

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


            float x1Squere = 0f, x2Squere = 0f, dot = 0f, prod = 0f;

            if (element1 == element2)
            {
                if (DiagonalDotCacheBuilded)
                    return DiagonalDotCache[element1];
                else
                {
                    //all parts are the same
                   // x1Squere = x2Squere = dot = linKernel.Product(element1, element1);
                    //prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));
                   // (x1Squere + x2Squere - 2 * dot)==0 this expresion is equal zero
                    //so we can prod set to 1 beceause exp(0)==1
                    prod = 1f;
                }
            }
            else
            {
                //when element1 and element2 are different we have to compute all parts
                x1Squere = linKernel.Product(element1, element1);
                x2Squere = linKernel.Product(element2, element2);
                dot = linKernel.Product(element1, element2);
                prod = (float)Math.Exp(-Gamma * (x1Squere + x2Squere - 2 * dot));
            }
            

            return prod;
        }

        public override void Init()
        {
            linKernel.Init();
            base.Init();
        }

      
        /// <summary>
        /// Creates object for parameters selection for RBF kernel
        /// </summary>
        /// <returns></returns>
        public override ParameterSelection<SparseVec> CreateParameterSelection()
        {
            return new RbfParameterSelection();
        }

        public override string ToString()
        {
            return string.Format("RBF kernel, Gamma={0}", Gamma);
        }

        public void Dispose()
        {
            linKernel.Dispose();

            problemElements = null;
            Labels = null;
            DiagonalDotCache = null;
            IsInitialized = false;
        }
    }
}
