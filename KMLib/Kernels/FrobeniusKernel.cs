using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using dnAnalytics.LinearAlgebra;

namespace KMLib.Kernels
{

    /// <summary>
    /// Linear matrix kelrnel, 
    /// </summary>
    public class FrobeniusKernel: MatrixKernel
    {

        public override sealed float Product(Matrix element1, Matrix element2)
        {

            if((element1.Columns!= element2.Columns) || (element1.Rows!=element2.Rows) )
            {
                throw new RankException("Matrix have different sizes");
            }

            float product = 0f;

            for (int i = 0; i <element1.Rows; i++)
            {
                product += (float) element1.GetRow(i).DotProduct(element2.GetRow(i));

            }

            return product;
        }

        public override float Product(int element1, int element2)
        {
            if (element1 == element2 && (DiagonalDotCache != null))
                return DiagonalDotCache[element1];


            float prod = 0;
            prod = Product(problemElements[element1], problemElements[element2]);

            return prod;

        }
    }
}
