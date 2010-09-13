namespace KMLib.Kernels
{
    /// <summary>
    /// Interface for kernels
    /// </summary>
    /// <typeparam name="TProblemElement"></typeparam>
    public interface IKernel<TProblemElement>
    {

        TProblemElement[] ProblemElements { get; set; }

        float[] Labels { get; set; }
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

        /// <summary>
        /// Cache for kenrle products between two the same vectors
        /// </summary>
        float[] DiagonalDotCache { get; }

        /// <summary>
        /// Kernel initialization
        /// </summary>
        void Init();

        /// <summary>
        /// Products between one element and all of the rest vectors
        /// </summary>
        /// <param name="element1"></param>
        /// <returns></returns>
        void AllProducts(int element1, float[] results);

        ParameterSelection<TProblemElement> CreateParameterSelection();

    }
}