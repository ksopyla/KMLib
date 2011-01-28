using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    /// Class for finding best parameters for Linear kernel, in this case 
    /// <see cref="SearchParams"/> search only one parameter C
    /// </summary>
    public class LinearParameterSelection: ParameterSelection<SparseVec> 
    {
        public override void SearchParams(Problem<SparseVec> problem, 
            out float C, 
            out IKernel<SparseVec> kernel)
        {
         //   throw new NotImplementedException();
            Debug.WriteLine("Starting Parameter Selection for Linear kernel");

            //create range for penalty parameter C
            IList<double> rangeC = PowerRange(MinCPower, MaxCPower, PowerBase, PowerStep);

            double crossValidation = double.MinValue;
            double maxC = double.MinValue;

            //we can set this kernel, because we only looking for one parameter C 
            LinearKernel bestKernel = new LinearKernel();
            object lockObj = new object();


            List<SparseVec>[] foldsElements;
            List<float>[] foldsLabels;
            Validation<SparseVec>.MakeFoldsSplit(problem, NrFolds, out foldsElements, out foldsLabels);

            Parallel.ForEach(rangeC, paramC =>
            {
                //do cross validation
                    Stopwatch timer = Stopwatch.StartNew();

                    Validation<SparseVec> valid = new Validation<SparseVec>();
                    valid.Evaluator = new SequentialDualEvaluator<SparseVec>();
                    valid.TrainingProblem = problem;

                    //but here kernel should be different because 
                    // in CSVM.Init we set the problem to kernel
                    valid.Kernel = new LinearKernel();
                    valid.C = (float)paramC;

                    double acc = valid.CrossValidateOnFolds(problem.ElementsCount, foldsElements, foldsLabels);

                    //old code
                    //var tmpKernel = new LinearKernel();

                    //double acc = Validation.CrossValidateOnFolds(problem.ElementsCount,
                    //                                             foldsElements,
                    //                                             foldsLabels, tmpKernel,
                    //                                             (float)paramC);

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
