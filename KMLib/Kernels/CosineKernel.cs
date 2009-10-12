using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;


namespace KMLib.Kernels
{
    /// <summary>
    /// computes cosine similarity
    /// dot(x,y)/norm(x)*norm(y);
    ///
    /// not very usefull when Vectors have norm=1, then its equal to LinearKernel dot product
    /// 
    ///  
    /// </summary>
    public class CosineKernel: VectorKernel
    {
        private LinearKernel linKernel;

        public CosineKernel(Vector[] vectors) : base(vectors)
        {
            linKernel = new LinearKernel(vectors);

            DiagonalDotCache = new float[vectors.Length];
            for (int i = 0; i < DiagonalDotCache.Length; i++)
            {
                DiagonalDotCache[i] = Product(vectors[i], vectors[i]);
            }
        }

        //public CosineKernel(Vector[] vectors, int cacheSize) : base(vectors, cacheSize)
        //{
        //}

        /// <summary>
        /// computes cosine similarity cos = dot(x,y)/norm(x)*norm(y)
        /// </summary>
        /// <param name="element1">First vector</param>
        /// <param name="element2">Second vector</param>
        /// <returns></returns>
        public override float Product(Vector element1, Vector element2)
        {
            float dot = linKernel.Product(element1, element2);

            float norm1 =(float) Math.Sqrt(linKernel.Product(element1, element1));
            float norm2 = (float) Math.Sqrt(linKernel.Product(element2, element2));

            float prod = dot/norm1*norm2;

            return prod;
        }


        /// <summary>
        /// computes cosine similarity cos = dot(x,y)/norm(x)*norm(y)
        /// </summary>
        /// <param name="element1">index of first vector in ProblemElement array</param>
        /// <param name="element2">index of second vector in ProblemElement array</param>
        /// <returns>Cosine between two vectors</returns>
        public override float Product(int element1, int element2)
        {
            if (element1 == element2 && (DiagonalDotCache != null))
                return DiagonalDotCache[element1];

            float norm1 = (float) Math.Sqrt(linKernel.Product(element1, element1));
            float norm2 = (float) Math.Sqrt(linKernel.Product(element2, element2));

            float dot = linKernel.Product(element1, element2);

            float prod = dot/(norm1*norm2);

            return prod;
        }
    }
}
