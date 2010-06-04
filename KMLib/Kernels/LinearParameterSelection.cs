using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using dnAnalytics.LinearAlgebra;

namespace KMLib.Kernels
{

    /// <summary>
    /// Class for finding best parameters for Linear kernel, in this case 
    /// <see cref="SearchParams"/> search only one parameter C
    /// </summary>
    public class LinearParameterSelection: ParameterSelection<SparseVector> 
    {
        public override void SearchParams(Problem<SparseVector> problem, 
            out float C, 
            out IKernel<SparseVector> kernel)
        {
         //   throw new NotImplementedException();
            Debug.WriteLine("Starting Parameter Selection for Linear kernel");

            //create range for penalty parameter C
            IList<double> rangeC = PowerRange(MinCPower, MaxCPower, PowerBase, PowerStep);

            double crossValidation = double.MinValue;
            double maxC = double.MinValue;
            LinearKernel bestKernel = null;
            object lockObj = new object();


            List<SparseVector>[] foldsElements;
            List<float>[] foldsLabels;
            Validation.MakeFoldsSplit(problem, NrFolds, out foldsElements, out foldsLabels);

            Parallel.ForEach(rangeC, paramC =>
            {
                //do cross validation
                    Stopwatch timer = Stopwatch.StartNew();

                    var tmpKernel = new LinearKernel();

                    double acc = Validation.CrossValidateOnFolds(problem.ElementsCount,
                                                                 foldsElements,
                                                                 foldsLabels, tmpKernel,
                                                                 (float)paramC);

                    lock (lockObj)
                    {
                        if (acc > crossValidation)
                        {
                            crossValidation = acc;
                            maxC = paramC;
                            bestKernel = tmpKernel;
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
