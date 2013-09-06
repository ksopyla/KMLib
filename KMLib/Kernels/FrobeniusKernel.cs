using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using dnAnalytics.LinearAlgebra;
using System.Diagnostics;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    /// Linear matrix kelrnel, 
    /// </summary>
    public class FrobeniusKernel: MatrixKernel
    {
        public override ParameterSelection<Matrix> CreateParameterSelection()
        {
            return new FrobeniusParameterSelection();
        }

        public override sealed float Product(Matrix element1, Matrix element2)
        {

            if((element1.Columns!= element2.Columns) || (element1.Rows!=element2.Rows) )
            {
                throw new RankException("Matrix have different sizes");
            }
            
            double product = 0;
            
            //(from item in element1.GetEnumerator()
            //                     let row = item.A
            //                     let col = item.B
            //                     let val = item.C
            //                     select val * element2[row, col]).Sum();

           

            foreach (var item in element1.GetEnumerator())
            {

                int row = item.A;
                int col = item.B;
                double val = item.C;

                product += val * element2[row, col];
            }
            //for (int i = 0; i <element1.Rows; i++)
            //{
            //    product += (float) element1.GetRow(i).DotProduct(element2.GetRow(i));

            //}


            return (float) product;
        }

        public override float Product(int element1, int element2)
        {
            if (element1 == element2 && (DiagonalDotCacheBuilded))
                return DiagonalDotCache[element1];

            float prod = 0;
           // Debug.Write(string.Format("Product {0}-{1} ", element1, element2));

            prod = Product(problemElements[element1], problemElements[element2]);

            return prod;

        }
    }
}
