using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;

namespace KMLib.SVMSolvers
{

    /// <summary>
    /// Represents base class for different binary clasification SVM solvers
    /// </summary>
    /// <typeparam name="TProblemElement"></typeparam>
    public abstract class Solver<TProblemElement>
    {

        /// <summary>
        /// Trainning problem
        /// </summary>
        public Problem<TProblemElement> problem;

        /// <summary>
        /// Penalty parameter C
        /// </summary>
        public float C;

        /// <summary>
        /// SVM kernel for computing producst
        /// </summary>
        public IKernel<TProblemElement> kernel;

        /// <summary>
        /// Construct the solver
        /// </summary>
        /// <param name="problem">trainning problem</param>
        /// <param name="kernel">kernel</param>
        /// <param name="C">penalty parameter</param>
        public Solver(Problem<TProblemElement> problem, IKernel<TProblemElement> kernel, float C)
        {
            this.problem = problem;
            this.kernel = kernel;
            this.C = C;
        }
        /// <summary>
        /// Construct the solver when kernel is not needed
        /// </summary>
        /// <param name="problem">trainning problem</param>
        /// <param name="C">penalty parameter</param>
        public Solver(Problem<TProblemElement> problem, float C)
        {
            this.problem = problem;
            this.kernel = null;
            this.C = C;
        }

        /// <summary>
        /// Abstract method for computing trainde model
        /// </summary>
        /// <returns>trained model</returns>
        public abstract Model<TProblemElement> ComputeModel();


       // public abstract void Init() { }
        
    }
}