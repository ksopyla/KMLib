using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using GASS.CUDA;




namespace KMLib.Kernels.GPU
{
    class CuLinearKernel : VectorKernel<SparseVector> , IDisposable
    {

        /// <summary>
        /// linear kernel for normal product
        /// </summary>
        private LinearKernel linKernel;


        /// <summary>
        /// Cuda .net class for cuda opeation
        /// </summary>
        private CUDA cuda;


        public override SparseVector[] ProblemElements
        {
            set
            {
                if (value == null) throw new ArgumentNullException("value");
                linKernel.ProblemElements = value;

                base.ProblemElements = value;

                //transform elements to specific array forman eg. CSR

                //for (int i = 0; i < value.Length; i++)
                //{
                //    var vec = value[i];

                    
                //}

            }
        }


        public CuLinearKernel()
        {

            linKernel = new LinearKernel();

            cuda = new CUDA(true);
            


        }

        public override float Product(SparseVector element1, SparseVector element2)
        {
            return linKernel.Product(element1, element2);
        }

        public override float Product(int element1, int element2)
        {
            return linKernel.Product(element1, element2);
        }

        public override ParameterSelection<SparseVector> CreateParameterSelection()
        {
            return linKernel.CreateParameterSelection();
        }

        public override float[] AllProducts(int element1)
        {

            //cuda calculation


            return base.AllProducts(element1);
        }

        #region IDisposable Members

        public void Dispose()
        {
            if (cuda != null)
            {
                cuda.Dispose();
                cuda = null;
            }
        }

        #endregion
    }
}
