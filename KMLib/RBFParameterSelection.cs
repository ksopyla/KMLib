using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using System.Threading;
using KMLib.Kernels;

namespace KMLib
{
   public class RBFParameterSelection: ParameterSelection<Vector>
    {


        public override void SearchParams(Problem<Vector> problem, out float C, out IKernel<Vector> kernel)
        {
            //create range for penalty parameter C
            IList<double> rangeC = PowerRange(MinCPower, MaxCPower, PowerBase, PowerStep);

            //create range for gamma parameter in Rbf Kernel
            IList<double> rangeGamma = PowerRange(MinGammaPower, MaxGammaPower, PowerBase, PowerStep);

            //creates Rbf kernels with different gamma's
            IList<RbfKernel> rbfKernels = new List<RbfKernel>(rangeGamma.Count);
            foreach (var gamma in rangeGamma)
            {
                var tmpkernel = new RbfKernel((float)gamma);
                tmpkernel.ProblemElements = problem.Elements;
                rbfKernels.Add(tmpkernel);
            }



            double crossValidation = double.MinValue;
            double maxC = double.MinValue;
            int bestKernelIndex = 0;
            object lockObj = new object();

            //paralle search for best C and gamma, for each kernel try different C
            Parallel.For(0, rbfKernels.Count,
                (i) =>
                {
                    for (int j = 0; j < rangeC.Count; j++)
                    {
                        //do cross validation
                        double acc = Validation.CrossValidation(problem, rbfKernels[i], (float)rangeC[i], NrFolds);

                        lock (lockObj)
                        {
                            if (acc > crossValidation)
                            {
                                crossValidation = acc;
                                maxC = rangeC[i];
                                //maxG = rbfKernels[i].Gamma;
                                bestKernelIndex = i;
                            }
                        }

                        Debug.WriteLine(string.Format("[{0:0.000000},{1:0.000000}]={2:0.0000}", rbfKernels[i].Gamma, rangeC[i], acc));

                    }

                });

            C = (float)maxC;
            kernel = rbfKernels[bestKernelIndex];

            
            Debug.WriteLine("\n");
            Debug.WriteLine("-------------- Grid Search summary ------------");
            Debug.WriteLine(string.Format("Max accuracy={0} c={1} gamma={2}  ", crossValidation, C, rbfKernels[bestKernelIndex].Gamma));
        }
    }
}
