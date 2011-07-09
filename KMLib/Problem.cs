using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib
{


    /// <summary>
    /// Problem, holds <see cref="Elements"> array of data</see> (type of <see cref="TProblemElement"/>) and 
    /// for each element its <see cref="Y"> label</see>
    /// This class is only container for data.
    /// </summary>
    /// <typeparam name="TProblemElement">Type of stored data can be Vectors, Matrix, strings etc.</typeparam>
    public class Problem<TProblemElement>: IDisposable
    {
        //public IKernel<TProblemElement> Kernel;

        /// <summary>
        /// Array of labels, on specific position "i" stands label for i-th <see cref="Elements">element</see>
        /// </summary>
        public float[] Y;

        /// <summary>
        /// Array of problem data of type <see cref="TProblemElement"/>
        /// </summary>
        public TProblemElement[] Elements;

        /// <summary>
        /// How many elements contains our problem
        /// </summary>
        public int ElementsCount; // { get { return Elements.Length;} }


        /// <summary>
        /// How many features contain one problem element
        /// </summary>
        /// <remarks>
        /// Maximum index for each problem element
        /// </remarks>
        public int FeaturesCount;


        public int NumberOfClasses;

        public float[] ElementLabels;
       
        /// <summary>
        /// Empty constructor,nothing initialized
        /// </summary>
        public Problem()
        {
            
        }

        /// <summary>
        /// constructor with ProblemElements array and labels for elements
        /// </summary>
        /// <param name="elements">array of elements</param>
        /// <param name="labels">array of elements labels</param>
        public Problem(TProblemElement[] elements,float[] labels)
        {
            Elements = elements;
            Y = labels;
            ElementsCount = elements.Length;

            NumberOfClasses = 2;
            ElementLabels = new float[2]{-1,1};
        }

        /// <summary>
        /// constructor with ProblemElements array and labels for elements
        /// </summary>
        /// <param name="elements">array of elements</param>
        /// <param name="labels">array of elements labels</param>
        /// <param name="featuresCount">number of features</param>
        public Problem(TProblemElement[] elements, float[] labels,int featuresCount)
        {
            Elements = elements;
            Y = labels;
            ElementsCount = elements.Length;
            FeaturesCount = featuresCount;

            NumberOfClasses = 2;
            ElementLabels = new float[2] { -1, 1 };
        }

        public Problem(TProblemElement[] sparseVec, float[] labels, int numberOfFeatures, int numberOfClasses, float[] elementClasses)
            :this(sparseVec,labels,numberOfFeatures)
        {
            NumberOfClasses = numberOfClasses;
            ElementLabels = elementClasses;
        }

        public void Dispose()
        {
            Elements = null;
            Y = null;
            ElementsCount = -1;
            FeaturesCount = -1;
            ElementLabels = null;
            NumberOfClasses = -1;
        }
    }


}