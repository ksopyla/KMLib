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
        
        public float[] Alpha;
        public float Rho;
        public int[] Labels;
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