using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.SVMSolvers;
using KMLib.Kernels;
using KMLib.Evaluate;
using System.Diagnostics;

namespace KMLib
{
	public class MultiClassSVM<TProblemElement>
	{

		/// <summary>
		/// type of SVM solver
		/// </summary>
		public SolverVariant solverType = SolverVariant.ParallelSmoFanSolver2;
		/// <summary>
		/// 
		/// </summary>
		private Problem<TProblemElement> mainProblem;


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
		private Evaluator<TProblemElement> evaluator;

		/// <summary>
		/// trained model
		/// </summary>
		private Model<TProblemElement>[] models;

		/// <summary>
		/// Solver, solves C-SVM optimization problem 
		/// </summary>
		protected Solver<TProblemElement> svmSolver;


		/// <summary>
		/// Initializes a new instance of the class.
		/// </summary>
		/// <param name="trainProblem">The train problem.</param>
		/// <param name="kernel">The kernel for computing product.</param>
		/// <param name="C">Parameter C.</param>
		/// <param name="evaluator">The evaluator class for prediction.</param>
		public MultiClassSVM(Problem<TProblemElement> trainProblem, IKernel<TProblemElement> kernel,
					float C, Evaluator<TProblemElement> evaluator)
		{
			this.mainProblem = trainProblem;

			this.kernel = kernel;
			this.C = C;
			//default evaluator
			this.evaluator = evaluator;
			models = new Model<TProblemElement>[trainProblem.NumberOfClasses];
			
		}




		/// <summary>
		/// Initialize clasifficator, initialize kernel and solver
		/// </summary>
		public void Init()
		{
		}

		public void Train()
		{
			int nrClasses = mainProblem.NumberOfClasses;
			var problemClasses = mainProblem.ElementLabels;
			
			kernel.ProblemElements = mainProblem.Elements;

			for (int i = 0; i <nrClasses; i++)
			{
				float[] subLabels = new float[mainProblem.ElementsCount];

				

				float mainLabel = problemClasses[i];

				GroupLabels(subLabels, mainLabel);
				kernel.Y = subLabels;// mainProblem.Y;
				kernel.Init();

				var subProblem = new Problem<TProblemElement>(mainProblem.Elements, subLabels, mainProblem.FeaturesCount);
				//
				//Solver = new ParallelSmoFanSolver<TProblemElement>(problem, kernel, C);
				//this solver works a bit faster and use less memory
				svmSolver = new ParallelSmoFanSolver2<TProblemElement>(subProblem, kernel, C);

				Console.WriteLine("User solver {0} and kernel {1}", svmSolver.ToString(), kernel.ToString());

				Stopwatch timer = Stopwatch.StartNew();
				models[i] = svmSolver.ComputeModel();
				Console.WriteLine("Model computed {0}  miliseconds={1}", timer.Elapsed, timer.ElapsedMilliseconds);

			}
			var disKernel = kernel as IDisposable;
			if (disKernel != null)
				disKernel.Dispose();
		   
		   // evaluator.Kernel = kernel;
		 //   evaluator.TrainedModel = model;
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
			if (models == null)
				throw new ApplicationException("Models not computed, call train method or read model from file");

			if (mainProblem == null)
				throw new ApplicationException("Train problem not set");

			int nrLabels = mainProblem.NumberOfClasses;
			float[] decisionVals = new float[nrLabels];
			float[] predictions = new float[predictElements.Length];

			for (int k = 0; k < predictElements.Length; k++)
			{


				int maxDecIdx = 0;
				for (int i = 0; i < nrLabels; i++)
				{
					evaluator.Kernel = kernel;
					evaluator.TrainedModel = models[i];

					decisionVals[i] = evaluator.PredictVal(predictElements[k]);

					if (i > 0 && decisionVals[i]>decisionVals[maxDecIdx])
						maxDecIdx = i;
				}
				predictions[k] = mainProblem.ElementLabels[maxDecIdx];

			}
			return predictions;

		}


		private void GroupLabels(float[] subLabels, float mainLabel)
		{
			for (int i = 0; i < mainProblem.ElementsCount; i++)
			{
				if (mainProblem.Y[i] != mainLabel)
				{
					subLabels[i] = -1;
				}
				else
					subLabels[i] = +1;
			}
		}

	
	}
}
