using System;
using System.Text;

namespace KMLib
{
    public class Model<TProblemElement>
    {
        public int NumberOfClasses;

        public int FeaturesCount;

        /// <summary>
        /// all class labels
        /// </summary>
        public float[] Labels { get; set; }

        /// <summary>
        /// Support Elements, aka. support vectors
        /// </summary>
        public TProblemElement[] SupportElements;
        
        /// <summary>
        /// computed alpha's value for all trainning elements, these with alpha is non zero is support element
        /// </summary>
        public float[] Alpha;

        /// <summary>
        /// rho == b parameters in svm formulation
        /// </summary>
        public float Bias;
        
        /// <summary>
        /// <see cref="SupportElements"/> labels
        /// </summary>
        public float[] Y;
        
        /// <summary>
        /// <see cref=" SupportElements"/> indexes from original problem set
        /// </summary>
        public int[] SupportElementsIndexes;

        /// <summary>
        /// "W" vector in primal problem
        /// </summary>
        public double[] W;
        public float Obj;
        public int Iter;
        public TimeSpan ModelTime;
        public long ModelTimeMs;

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder(100);

            sb.AppendFormat("\nModel time={0} ms={1} ", ModelTime, ModelTimeMs);
            sb.AppendFormat("iter={0} \n", Iter);
            sb.AppendFormat("obj={0} ", Obj);
            sb.AppendFormat("rho={0} ", Bias);
            sb.AppendFormat("nSV={0} \n", SupportElements.Length);
            

            return sb.ToString();
        }






        
    }
}