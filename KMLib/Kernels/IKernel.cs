using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace KMLib
{
    /// <summary>
    /// Interface for kernels
    /// </summary>
    /// <typeparam name="TProblemElement"></typeparam>
    public interface IKernel<TProblemElement>
    {
        /// <summary>
        /// Product of 2 elements, methods of messure similarity
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        float Product(TProblemElement element1,TProblemElement element2);

        /// <summary>
        /// Product of 2 elements, element accessed by index
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns></returns>
        float Product(int element1, int element2);

        float[] DiagonalDotCache { get; }
    }
}