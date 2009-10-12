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

        /*
        /// <summary>
        /// Compute decision function for problemElement(Vectors)
        /// </summary>
        /// <param name="Model">Trained Model</param>
        /// <param name="problemElement">Problem Element </param>
        /// <returns></returns>
        public float DecisionValue(Model<TProblemElement> Model, TProblemElement problemElement)
        {

            float sum = 0;
            //sum( apha_i*y_i*K(x_i,problemElement)) + b
            //sum can by compute only on support vectors
            for (int i = 0; i < Model.SupportElements.Length; i++)
            {
                sum += Model.Alpha[i] * Model.Labels[i] * Kernel.Product(Model.SupportElements[i], problemElement);
            }

            return sum + Model.Rho;
        }

        public float DecisionValue(float[] alpha, float b, int k)
        {

            float sum = 0;
            //sum( apha_i*y_i*K(x_i,problemElement)) + b
            //sum can by compute only on support vectors
            for (int i = 0; i < Elements.Length; i++)
            {
                sum += alpha[i] * Labels[i] * Kernel.Product(Elements[i], Elements[k]);
            }

            return sum + b;

        }
         * */

    }
}