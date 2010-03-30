﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;
using KMLib.SVMSolvers;

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
        /// <remarks>Now KMLib implements two solvers clasic <see cref="SMOSolver{TProblemElement}"/>
        /// and <see cref="SmoFanSolver{TProblemElement}"/> solver from LibSVM library
        /// </remarks>
        private Solver<TProblemElement> solver;

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


        public void Train()
        {
            kernel.ProblemElements = problem.Elements;
            solver = new SmoFanSolver<TProblemElement>(problem, kernel, C);
            model = solver.ComputeModel();
        }

        public float Predict(TProblemElement problemElement)
        {
            float sum = 0;
            //sum( apha_i*y_i*K(x_i,problemElement)) + b
            //sum can by compute only on support vectors
            for (int i = 0; i < problem.Elements.Length; i++)
            {

                sum += model.Alpha[i] * problem.Labels[i] * kernel.Product(problem.Elements[i], problemElement);
            }

            sum -= model.Rho;

            float ret = sum > 0 ? 1 : -1;

            return ret;

        }




    }
}