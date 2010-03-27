using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using dnAnalytics.LinearAlgebra;
using System.Threading;
using KMLib.Kernels;

namespace KMLib
{
    public class RBFParameterSelection : ParameterSelection<Vector>
    {


        public override void SearchParams(Problem<Vector> problem, out float C, out IKernel<Vector> kernel)
        {

            Debug.WriteLine("Starting Parameter Selection for RBF kernel");

            //create range for penalty parameter C
            IList<double> rangeC = PowerRange(MinCPower, MaxCPower, PowerBase, PowerStep);

            //create range for gamma parameter in Rbf Kernel
            IList<double> rangeGamma = PowerRange(MinGammaPower, MaxGammaPower, PowerBase, PowerStep);

            //creates Rbf kernels with different gamma's
            IList<RbfKernel> rbfKernels = new List<RbfKernel>(rangeGamma.Count);
            foreach (var gamma in rangeGamma)
            {
                var tmpkernel = new RbfKernel((float)gamma);
                //  tmpkernel.ProblemElements = problem.Elements;
                rbfKernels.Add(tmpkernel);
            }



            double crossValidation = double.MinValue;
            double maxC = double.MinValue;
            RbfKernel bestKernelIndex = null;
            object lockObj = new object();


            List<Vector>[] foldsElements;
            List<float>[] foldsLabels;
            Validation.MakeFoldsSplit(problem, NrFolds, out foldsElements, out foldsLabels);


            //paralle search for best C and gamma, for each kernel try different C

            //foreach (var rbfKernel in rbfKernels)
            //{

            Parallel.ForEach(rbfKernels, (rbfKernel) =>
            {
                for (int j = 0; j < rangeC.Count; j++)
                {
                    //do cross validation
                    Stopwatch timer = Stopwatch.StartNew();

                    //double acc = Validation.CrossValidation(problem, rbfKernel,(float)rangeC[j], NrFolds);

                    double acc = Validation.CrossValidateOnFolds(problem.ElementsCount,
                                                                 foldsElements,
                                                                 foldsLabels, rbfKernel,
                                                                 (float)rangeC[j]);

                    lock (lockObj)
                    {
                        if (acc > crossValidation)
                        {
                            crossValidation = acc;
                            maxC = rangeC[j];
                            //maxG = rbfKernels[i].Gamma;
                            bestKernelIndex = rbfKernel;
                        }
                    }

                    Debug.WriteLine(
                        string.Format(
                            "CrossValidation time={0},C={1},gamma={2}->{3:0.#####}",
                            timer.Elapsed, rangeC[j], rbfKernel.Gamma, acc));

                }

            });

            C = (float)maxC;
            kernel = bestKernelIndex;


            Debug.WriteLine("\n");
            Debug.WriteLine("-------------- Grid Search summary ------------");
            Debug.WriteLine(string.Format("Max accuracy={0} c={1} gamma={2}  ", crossValidation, C, bestKernelIndex.Gamma));
        }
    }
}
