using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
//using dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;
using KMLib.Helpers;

namespace KMLib.Kernels
{

    /// <summary>
    /// Clas for searching best parameters "C" and "Gamma" for RbfKernel
    /// </summary>
    public class RbfParameterSelection : ParameterSelection<SparseVec>
    {

        /// <summary>
        /// Finds best params C and Gamma
        /// </summary>
        /// <param name="problem"></param>
        /// <param name="C"></param>
        /// <param name="kernel"></param>
        public override void SearchParams(Problem<SparseVec> problem, out float C, out IKernel<SparseVec> kernel)
        {

            Debug.WriteLine("Starting Parameter Selection for RBF kernel");

            //create range for penalty parameter C
            IList<double> rangeC = PowerRange(MinCPower, MaxCPower, PowerBase, PowerStep);

            //create range for gamma parameter in Rbf Kernel
            IList<double> gammaRange = PowerRange(MinGammaPower, MaxGammaPower, PowerBase, PowerStep);

            ////creates Rbf kernels with different gamma's
            //IList<RbfKernel> rbfKernels = new List<RbfKernel>(gammaRange.Count);
            //foreach (var gamma in gammaRange)
            //{
            //    var tmpkernel = new RbfKernel((float)gamma);
            //    //  tmpkernel.ProblemElements = problem.Elements;
            //    rbfKernels.Add(tmpkernel);
            //}

            double crossValidation = double.MinValue;
            double maxC = double.MinValue;
            float bestGamma = -2f;
            object lockObj = new object();


            List<SparseVec>[] foldsElements;
            List<float>[] foldsLabels;
            Validation<SparseVec>.MakeFoldsSplit(problem, NrFolds, out foldsElements, out foldsLabels);


            //paralle search for best C and gamma, for each kernel try different C

            //foreach (var rbfKernel in rbfKernels)
            //{

            //Parallel.ForEach(rbfKernels, (rbfKernel) =>
            //{
            Parallel.ForEach(gammaRange,gamma=>{

                Validation<SparseVec> valid = new Validation<SparseVec>();
                valid.Evaluator = new DualEvaluator<SparseVec>();
                valid.TrainingProblem = problem;

                //but here kernel should be different because 
                // in CSVM.Init we set the problem to kernel
                valid.Kernel = new RbfKernel((float) gamma);
                
                for (int j = 0; j < rangeC.Count; j++)
                {
                    valid.C =(float) rangeC[j];

                    //do cross validation
                    Stopwatch timer = Stopwatch.StartNew();

                    double acc = valid.CrossValidateOnFolds(problem.ElementsCount, foldsElements, foldsLabels);
                    
                    lock (lockObj)
                    {
                        if (acc > crossValidation)
                        {
                            crossValidation = acc;
                            maxC = rangeC[j];
                            
                            bestGamma =(float) gamma;
                        }
                    }

                    Debug.WriteLine(
                        string.Format(
                            "CrossValidation time={0},C={1},gamma={2}->{3:0.#####}",
                            timer.Elapsed, rangeC[j], gamma, acc));

                }

            });

            C = (float)maxC;
            kernel = new RbfKernel(bestGamma);


            Debug.WriteLine("\n");
            Debug.WriteLine("-------------- Grid Search summary ------------");
            Debug.WriteLine(string.Format("Max accuracy={0} c={1} gamma={2}  ", crossValidation, C, bestGamma));
        }
    }
}


