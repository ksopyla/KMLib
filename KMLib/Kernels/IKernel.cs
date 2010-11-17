namespace KMLib.Kernels
{
    /// <summary>
    /// Interface for kernels
    /// </summary>
    /// <typeparam name="TProblemElement"></typeparam>
    public interface IKernel<TProblemElement>
    {

        /// <summary>
        /// Gets or sets the problem elements.
        /// </summary>
        /// <value>The problem elements.</value>
        TProblemElement[] ProblemElements { get; set; }

        /// <summary>
        /// Gets or sets the labels.
        /// </summary>
        /// <value>The labels.</value>
        float[] Labels { get; set; }

        /// <summary>
        /// Product of 2 elements, methods of messure similarity
        /// </summary>
        /// <param name="element1">first element</param>
        /// <param name="element2">second element</param>
        /// <returns>Kernel product between two elements </returns>
        float Product(TProblemElement element1,TProblemElement element2);

        /// <summary>
        /// Product of 2 elements, element accessed by index
        /// </summary>
        /// <param name="element1"></param>
        /// <param name="element2"></param>
        /// <returns>Kernel product between tow elements</returns>
        float Product(int element1, int element2);

        /// <summary>
        /// Cache for kernel products between two the same vectors
        /// </summary>
        float[] DiagonalDotCache { get; }


         bool IsInitialized { get;  }
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

        /// <summary>
        /// Creates the parameter selection class for finding the best parameter for 
        /// this kernel.
        /// </summary>
        /// <returns>Instance of parameter selection class</returns>
        ParameterSelection<TProblemElement> CreateParameterSelection();

    }
}