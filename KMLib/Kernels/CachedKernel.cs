using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    public class CachedKernel<TProblemElement>
    {
        /// <summary>
        /// problem labels
        /// </summary>
        private sbyte[] y;


        /// <summary>
        /// cache for kernel matrix, 
        /// </summary>
        private BlockCache cache;

        /// <summary>
        /// kernel matrix diagonal elements array
        /// </summary>
         private float[] QD;

        /// <summary>
        /// 
        /// </summary>
        private float CacheSize = 140; //in MB


        private Problem<TProblemElement> problem;

        private readonly IKernel<TProblemElement> kernel;


        /// <summary>
        /// 
        /// </summary>
        /// <param name="problem"></param>
        /// <param name="problemkernel"></param>
        public CachedKernel(Problem<TProblemElement> problem, IKernel<TProblemElement> problemkernel)
        {

            this.problem = problem;
            kernel = problemkernel;

            y = new sbyte[problem.ElementsCount];


            //rewrite labels from proble to y[]
            for (int i = 0; i < problem.ElementsCount; i++)
            {
                if (problem.Labels[i] > 0) y[i] = +1;
                else y[i] = -1;
            }


            cache = new BlockCache(problem.ElementsCount, (long)(CacheSize * (1 << 20)));

            //computation for diagonal element array(cache), should be stored in temporary array, 
            //because Product(i,j) method use this diagonal cache
            //float[] tempQD = new float[problem.ElementsCount];
            //for (int i = 0; i < problem.ElementsCount; i++)
            //    tempQD[i] = kernel.Product(i, i);

            //QD = tempQD;
            QD = kernel.DiagonalDotCache;
        }


        /// <summary>
        /// Get row from kernel matrix 
        /// </summary>
        /// <param name="i">i-th row</param>
        /// <param name="len">block length</param>
        /// <returns></returns>
        public float[] GetQ(int i, int len)
        {
            float[] data = null;
            int start, j;
            if ((start = cache.GetData(i, ref data, len)) < len)
            {
                for (j = start; j < len; j++)
                    data[j] = (y[i] * y[j] * kernel.Product(i, j));
            }
            return data;
        }

        public float[] GetQD()
        {
            return QD;
        }

        public void SwapIndex(int i, int j)
        {
            cache.SwapIndex(i, j);

            //ty była zamiana x[i] z x[i] oraz x_squere
            //base.SwapIndex(i, j);

            //_x.SwapIndex(i, j);
            y.SwapIndex(i, j);
            QD.SwapIndex(i, j);
        }

        /* old implementation with computing kernel product
                /// <summary>
                /// linear dot product
                /// </summary>
                /// <param name="element1"></param>
                /// <param name="element2"></param>
                /// <returns></returns>
                public float Product(Vector element1, Vector element2)
                {
                    float sum = 0;

                    uint i1 = 0, i2 = 0;

                    while (i1 < element1.Data.Length && i2 < element2.Data.Length)
                    {
                        int index1 = element1.Data[i1].Index;
                        int index2 = element2.Data[i2].Index;

                        if (index1 == index2)
                        {
                            sum += element1.Data[i1].Value * element2.Data[i2].Value;
                            i1++;
                            i2++;
                        }
                        else
                        {
                            if (index1 < index2)
                                i1++;
                            else
                                i2++;
                        }

                    }
                    return sum;
                }

                public float Product(int element1, int element2)
                {
                    if (element1 == element2 && (QD != null))
                        return QD[element1];

                    return this.Product(problem.Elements[element1], problem.Elements[element2]);
           
                }
         */
    }
}
