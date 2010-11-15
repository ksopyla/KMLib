using System;
using System.Text;

namespace KMLib
{
    public class Model<TProblemElement>
    {
        public int NumberOfClasses;
        

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
        public float Rho;
        
        /// <summary>
        /// <see cref="SupportElements"/> labels
        /// </summary>
        ///public int[] Labels;
        
        /// <summary>
        /// <see cref=" SupportElements"/> indexes from original problem set
        /// </summary>
        public int[] SupportElementsIndexes;

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder(100);

            sb.AppendFormat("number of SV={0} \n", SupportElements.Length);
            sb.AppendFormat("number of alpha = {0} \n", Alpha.Length);
            sb.AppendFormat("number of alpha non zero ={0}", SupportElementsIndexes.Length);
            sb.AppendFormat("rho={0}", Rho);

            return sb.ToString();
        }

        

    }
}