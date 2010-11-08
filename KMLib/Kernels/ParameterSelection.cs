using System;
using System.Collections.Generic;

namespace KMLib.Kernels
{

    /// <summary>
    /// Contains static methods for parameter selection in diferent SVM models
    /// find penalty C and other parameters used in kernels
    /// </summary>
    public abstract class ParameterSelection<T>
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

       
        public abstract void SearchParams(Problem<T> problem,out float C,out IKernel<T> kernel);



    }
}


