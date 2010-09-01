using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;
using KMLib.Helpers;
using System.Diagnostics;

namespace KMLib
{

    /// <summary>
    /// Solves C-SVM problem,
    /// Class is generic so you can use any type (string,Vector, Matrix etc) but you have to remember to
    /// implement custom Kernel witch meassure similarity between your custom objects
    /// </summary>
    /// <typeparam name="TProblemElement">Problem elements, can use Vector, Matrix or any custom type</typeparam>
    public class CSVM<TProblemElement>
    {

        private Problem<TProblemElement> problem;

        private float C = 0.5f;


        private IKernel<TProblemElement> kernel;

        private Model<TProblemElement> model;

        /// <summary>
        /// Solver, solves C-SVM optimization problem 
        /// </summary>
        protected Solver<TProblemElement> Solver;

        public CSVM(Problem<TProblemElement> trainProblem, IKernel<TProblemElement> kernel, float C)
        {
            this.problem = trainProblem;

            this.kernel = kernel;
            this.C = C;

            //=======================================================================//
            //  solver = new SMOSolver<TProblemElement>(problem, kernel, C);         //
            // solver = new SmoFanSolver<TProblemElement>(trainProblem, kernel, C);  //
            //=======================================================================//

        }


        /// <summary>
        /// Initialize clasifficator, initialize kernel and solver
        /// </summary>
        public void Init()
        {
            kernel.ProblemElements = problem.Elements;
            kernel.Labels = problem.Labels;
            kernel.Init();

            //solver = new ParallelSMOSolver<TProblemElement>(problem, kernel, C);
            //solver = new ModSMOSolver<TProblemElement>(problem, kernel, C);
            //solver = new SMOSolver<TProblemElement>(problem, kernel, C); 

            Solver = new ParallelSmoFanSolver<TProblemElement>(problem, kernel, C);
           // Solver = new SmoFanSolver<TProblemElement>(problem, kernel, C);
        }

        public void Train()
        {
            if (kernel.ProblemElements == null)
                throw new ArgumentNullException("Not initialized, should call Init method");
           

           

            Console.WriteLine("User solver {0}", Solver.ToString());

            Stopwatch timer = Stopwatch.StartNew();
            model = Solver.ComputeModel();
            Console.WriteLine("Model computed {0}", timer.Elapsed);
        }

        public float Predict(TProblemElement problemElement)
        {
            float sum = 0;
            //sum( apha_i*y_i*K(x_i,problemElement)) + b
            //sum can by compute only on support vectors


            //for (int i = 0; i < problem.Elements.Length; i++)
            //{
            //    //todo: alpha array is sparse, try to exploit it
            //    sum += model.Alpha[i] * problem.Labels[i] * kernel.Product(problem.Elements[i], problemElement);
            //}


            int index = -1;


            for (int k = 0; k < model.SupportElementsIndexes.Length; k++)
            {
                index = model.SupportElementsIndexes[k];

                //todo: alpha array is sparse, try to exploit it
                sum += model.Alpha[index] * problem.Labels[index] * kernel.Product(problem.Elements[index], problemElement);
            }


            sum -= model.Rho;

            float ret = sum > 0 ? 1 : -1;

            return ret;

        }




    }
}