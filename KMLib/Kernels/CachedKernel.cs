using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;
using System.Threading.Tasks;
using System.Collections.Concurrent;

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
            int start = 0, j;


            //data = new float[problem.ElementsCount];
            //for (j = start; j < len; j++)
            //    data[j] = (y[i] * y[j] * kernel.Product(i, j));

            //with cache
            if ((start = cache.GetData(i, ref data, len)) < len)
            {
                
                //todo: error we change referecne
                kernel.AllProducts(i,ref data);

                // for (j = start; j < len; j++)
                //     data[j] = (y[i] * y[j] * kernel.Product(i, j));


                //var  data2 = kernel.AllProducts(i);

                //for (int k = 0; k < len; k++)
                //{
                //    if (data[k] != data2[k])
                //        throw new InvalidOperationException(string.Format("different val on {0} position", k));
                //}

                //var partition = Partitioner.Create(start, len);

                //Parallel.ForEach(partition, (range) =>
                //{
                //    for (int k = range.Item1; k < range.Item2; k++)
                //    {
                //        data[k] = (y[i] * y[k] * kernel.Product(i, k));
                //    }

                //});



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


    }
}
