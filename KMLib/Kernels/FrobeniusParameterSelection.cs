using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    /// Class for searching best prams for matrix Frobenius kernel
    /// </summary>
    public class FrobeniusParameterSelection : ParameterSelection<Matrix>
    {


        public override void SearchParams(Problem<Matrix> problem, out float C, out IKernel<Matrix> kernel)
        {
            //throw new NotImplementedException();
            Debug.WriteLine("Starting Parameter Selection for Frobenius kernel");

            //create range for penalty parameter C
            IList<double> rangeC = PowerRange(MinCPower, MaxCPower, PowerBase, PowerStep);

            double crossValidation = double.MinValue;
            double maxC = double.MinValue;
            
            object lockObj = new object();


            List<Matrix>[] foldsElements;
            List<float>[] foldsLabels;

            Validation<Matrix>.MakeFoldsSplit(problem, NrFolds, out foldsElements, out foldsLabels);


            //we don't have to specify more kernels, because we only searching for C parameter
            FrobeniusKernel bestKernel  = new FrobeniusKernel();

            Parallel.ForEach(rangeC, paramC =>
            {
                //do cross validation
                Stopwatch timer = Stopwatch.StartNew();

                Validation<Matrix> valid = new Validation<Matrix>();
                valid.Evaluator = new SequentialDualEvaluator<Matrix>();
                valid.TrainingProblem = problem;

                //but here kernel should be different because 
                // in CSVM.Init we set the problem to kernel
                valid.Kernel = new FrobeniusKernel();
                valid.C = (float)paramC;

                double acc = valid.CrossValidateOnFolds(problem.ElementsCount, foldsElements, foldsLabels);


                lock (lockObj)
                {
                    if (acc > crossValidation)
                    {
                        crossValidation = acc;
                        maxC = paramC;
                       // bestKernel = tmpKernel;
                    }
                }

                Debug.WriteLine(
                    string.Format(
                        "CrossValidation time={0},C={1}->{2:0.#####}",
                        timer.Elapsed, paramC, acc));

            });

            C = (float)maxC;
            kernel = bestKernel;


            Debug.WriteLine("\n");
            Debug.WriteLine("-------------- Grid Search summary ------------");
            Debug.WriteLine(string.Format("Max accuracy={0} c={1} ", crossValidation, C));
        }
    }
}
