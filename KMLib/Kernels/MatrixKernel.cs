using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;

namespace KMLib.Kernels
{

    /// <summary>
    /// abstract class for matrix kernles
    /// </summary>
    public abstract class MatrixKernel: IKernel<Matrix>
    {

        protected Matrix[] problemElements;
        protected bool DiagonalDotCacheBuilded = false;
        public Matrix[] ProblemElements
        {
            get { return problemElements; }
            set
            {
                DiagonalDotCacheBuilded = false;
                problemElements = value;
               // ComputeDiagonalDotCache();
            }
        }

        protected void ComputeDiagonalDotCache()
        {
            DiagonalDotCache = new float[ProblemElements.Length];
            for (int i = 0; i < DiagonalDotCache.Length; i++)
            {
                //DiagonalDotCache[i] = Product(ProblemElements[i], ProblemElements[i]);
                DiagonalDotCache[i] = Product(i,i);
            }
            DiagonalDotCacheBuilded = true;
        }


        public float[] DiagonalDotCache
        {
            get; protected set;
        }

        public abstract ParameterSelection<Matrix> CreateParameterSelection();


        public abstract float Product(Matrix element1, Matrix element2);
        public abstract float Product(int element1, int element2);



        #region IKernel<Matrix> Members


        public float[] Labels
        {
            get
           ;
            set;
        }

       

        #endregion

        #region IKernel<Matrix> Members


        public void Init()
        {
            ComputeDiagonalDotCache();
            
        }

        #endregion

        #region IKernel<Matrix> Members


        public void AllProducts(int element1,  float[] results)
        {
            throw new NotImplementedException();
        }

        #endregion

        #region IKernel<Matrix> Members


        public float[] Predict(Model<Matrix> model, Matrix[] predictElements)
        {
            throw new NotImplementedException();
        }

        #endregion


        public bool IsInitialized
        {
            get { throw new NotImplementedException(); }
        }
    }
}
