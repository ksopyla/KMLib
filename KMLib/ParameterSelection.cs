using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using dnAnalytics.LinearAlgebra;
using KMLib.Kernels;

namespace KMLib
{

    /// <summary>
    /// Contains static methods for parameter selection in diferent SVM models
    /// find penalty C and other parameters used in kernels
    /// </summary>
    public class ParameterSelection
    {
        public  bool ShowDebug = false;

        public  int NrFolds = 5;

        #region Power parameters
        /// <summary>
        /// min power of <see cref="PowerBase"/> for finding C parameter (default 2^-5)
        /// </summary>
        public  double MinCPower = -5; //0,03125
        /// <summary>
        /// max power of <see cref="PowerBase"/> for finding C parameter (default 2^15)
        /// </summary>
        public  double MaxCPower = 15; //32768
        /// <summary>
        /// step for changing power, (MinPower+step,  .... MaxCPower)
        /// </summary>
        public double PowerStep = 2;

        /// <summary>
        /// Default minimum power for the Gamma value (-15)
        /// </summary>
        public int MinGammaPower = -15; //0,000030517578125
        /// <summary>
        /// Default maximum power  for the Gamma Value (3)
        /// </summary>
        public int MaxGammaPower = 3; // 8

        /// <summary>
        /// Default power base
        /// </summary>
        public double PowerBase = 2;


        #endregion

        /// <summary>
        /// Create range of parameters value, (powerbase^minPower, powerBase^maxPower)
        /// </summary>
        /// <param name="minPower">Min Power</param>
        /// <param name="maxPower">Max power</param>
        /// <param name="powerBase">power Base</param>
        /// <param name="step">iteration step MinPower+step</param>
        /// <returns>list of parameters in specific range</returns>
        public static IList<double> PowerRange(double minPower, double maxPower, double powerBase, double step)
        {
            List<double> range = new List<double>();

            for (double d = minPower; d <= maxPower; d += step)
                range.Add(Math.Pow(powerBase, d));

            return range;
        }


        /// <summary>
        /// Perform Grid Search for parameter Selection for vector problem
        /// finds best penalty C and Gamma parameter in Rbf Kernel
        /// </summary>
        /// <param name="problem"></param>
        /// <param name="C"></param>
        /// <param name="Gamma"></param>
        public  void GridSearchForRbfKerel(
                        Problem<Vector> problem,
                        out float C, out double Gamma)
        {

            //create range for penalty parameter C
            IList<double> rangeC = PowerRange(MinCPower, MaxCPower, PowerBase, PowerStep);

            //create range for gamma parameter in Rbf Kernel
            IList<double> rangeGamma = PowerRange(MinGammaPower, MaxGammaPower, PowerBase, PowerStep);

            //creates Rbf kernels with different gamma's
            IList<RbfKernel> rbfKernels = new List<RbfKernel>(rangeGamma.Count);
            foreach (var gamma in rangeGamma)
            {
                var kernel = new RbfKernel((float)gamma);
                kernel.ProblemElements = problem.Elements;
                rbfKernels.Add(kernel);
            }



            double crossValidation = double.MinValue;
            double maxC = double.MinValue, maxG = double.MinValue;
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
                                maxG = rbfKernels[i].Gamma;
                            }
                        }
                        if (ShowDebug)
                        {
                            Console.WriteLine("[{0:0.000000},{1:0.000000}]={2:0.0000}", rbfKernels[i].Gamma, rangeC[i], acc);
                        }
                    }

                });

            C = (float)maxC;
            Gamma = maxG;

            if (ShowDebug)
            {
                Console.WriteLine();
                Console.WriteLine("-------------- Grid Search summary ------------");
                Console.WriteLine("Max accuracy={0} c={1} gamma={2}  ", crossValidation, C, Gamma);
            }
        }





    }
}
