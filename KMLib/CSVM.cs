using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Kernels;
using KMLib.Helpers;
using System.Diagnostics;
using KMLib.Evaluate;
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

        /// <summary>
        /// type of SVM solver
        /// </summary>
        public SolverVariant solverType = SolverVariant.ParallelSmoFanSolver2;
        /// <summary>
        /// 
        /// </summary>
        private Problem<TProblemElement> problem;


        /// <summary>
        /// Penalty parameter C in SVM
        /// </summary>
        private float C = 0.5f;

        /// <summary>
        /// Kernel for computing product
        /// </summary>
        private IKernel<TProblemElement> kernel;

        /// <summary>
        /// Evaluator for prediction
        /// </summary>
        private EvaluatorBase<TProblemElement> evaluator;

        /// <summary>
        /// trained model
        /// </summary>
        private Model<TProblemElement> model;

        /// <summary>
        /// Solver, solves C-SVM optimization problem 
        /// </summary>
        protected Solver<TProblemElement> svmSolver;
        // private Problem<TProblemElement> trainSubprob;
        //  private IKernel<TProblemElement> Kernel;
        // private EvaluatorBase<TProblemElement> Evaluator;




        /// <summary>
        /// Initializes a new instance of the <see cref="CSVM&lt;TProblemElement&gt;"/> class.
        /// </summary>
        /// <param name="trainProblem">The train problem.</param>
        /// <param name="kernel">The kernel for computing product.</param>
        /// <param name="C">Parameter C.</param>
        /// <param name="evaluator">The evaluator class for prediction.</param>
        public CSVM(Problem<TProblemElement> trainProblem, IKernel<TProblemElement> kernel,
                    float C, EvaluatorBase<TProblemElement> evaluator)
        {
            this.problem = trainProblem;

            this.kernel = kernel;
            this.C = C;
            //default evaluator
            this.evaluator = evaluator;
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
            //kernel.ProblemElements = problem.Elements;
            //kernel.Labels = problem.Labels;
            //kernel.Init();


            //solver = new ParallelSMOSolver<TProblemElement>(problem, kernel, C);
            //solver = new ModSMOSolver<TProblemElement>(problem, kernel, C);
            //solver = new SMOSolver<TProblemElement>(problem, kernel, C); 

            // Solver = new ParallelSmoFanSolver<TProblemElement>(problem, kernel, C);
            // Solver = new SmoFanSolver<TProblemElement>(problem, kernel, C);
        }

        public void Train()
        {

            kernel.ProblemElements = problem.Elements;
            kernel.Y = problem.Y;
            kernel.Init();

            //svmSolver = new SmoFanSolver<TProblemElement>(problem, kernel, C);
           // svmSolver = new ParallelSmoFanSolver<TProblemElement>(problem, kernel, C);
            //this solver works a bit faster and use less memory
            svmSolver = new ParallelSmoFanSolver2<TProblemElement>(problem, kernel, C);

            
            Debug.WriteLine("User solver {0} and kernel {1}", svmSolver.ToString(), kernel.ToString());

            Stopwatch timer = Stopwatch.StartNew();
            model = svmSolver.ComputeModel();
            Debug.WriteLine("Model computed,  {0}  miliseconds={1}", timer.Elapsed, timer.ElapsedMilliseconds);
            Console.WriteLine("model obj={0} rho={1} nSV={2} iter={3}", model.Obj, model.Bias, model.SupportElements.Length,model.Iter);

            var disKernel = kernel as IDisposable;
            if (disKernel != null)
                disKernel.Dispose();
            
          

            evaluator.Kernel = kernel;
            evaluator.TrainedModel = model;
            //evaluator.TrainningProblem = problem;

        }

        /// <summary>
        /// Predicts 
        /// </summary>
        /// <param name="problemElement"></param>
        /// <returns></returns>
        public float Predict(TProblemElement problemElement)
        {

            return evaluator.Predict(problemElement);
            //float sum = 0;

            //int index = -1;

            //for (int k = 0; k < model.SupportElementsIndexes.Length; k++)
            //{
            //    index = model.SupportElementsIndexes[k];
            //    sum += model.Alpha[index] * problem.Labels[index] *
            //                        kernel.Product(problem.Elements[index], problemElement);
            //}


            //sum -= model.Rho;

            //float ret = sum > 0 ? 1 : -1;

            //return ret;

        }

        public float[] Predict(TProblemElement[] predictElements)
        {
            //evaluator.Kernel = kernel;
            if (model == null)
                throw new ApplicationException("Model not computed, call train method or read model from file");

            if (problem == null)
                throw new ApplicationException("Train problem not set");


            evaluator.Init();
            Stopwatch t = Stopwatch.StartNew();
            float[] predictions = evaluator.Predict(predictElements);

            //toremove: only for tests
            Console.WriteLine("prediction takes {0} ms", t.ElapsedMilliseconds);

            //todo: Free evaluator memories

            var disposeEvaluator = evaluator as IDisposable;
            if (disposeEvaluator != null)
                disposeEvaluator.Dispose();


            return predictions;

        }




    }
}