using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib
{


    /// <summary>
    /// Problem, holds <see cref="Elements"> array of data</see> (type of <see cref="TProblemElement"/>) and 
    /// for each element its <see cref="Labels"> label</see>
    /// This class is only container for data.
    /// </summary>
    /// <typeparam name="TProblemElement">Type of stored data can be Vectors, Matrix, strings etc.</typeparam>
    public class Problem<TProblemElement>
    {
        //public IKernel<TProblemElement> Kernel;

        /// <summary>
        /// Array of labels, on specific position "i" stands label for i-th <see cref="Elements">element</see>
        /// </summary>
        public float[] Labels;

        /// <summary>
        /// Array of problem data of type <see cref="TProblemElement"/>
        /// </summary>
        public TProblemElement[] Elements;

        /// <summary>
        /// How many elements contains our problem
        /// </summary>
        public int ElementsCount; // { get { return Elements.Length;} }


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
            Labels = labels;
            ElementsCount = elements.Length;
        }



    }


}