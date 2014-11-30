using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;

namespace KMLib.Transforms
{

    /// <summary>
    /// Normalize vectors according to lp - norm, p>=1
    /// </summary>
    public class LpNorm: IDataTransform<SparseVec>
    {

        double Power = 2;

        public LpNorm(double power)
        {
            Power = power;
        }

        public SparseVec Transform(SparseVec input)
        {

            float norm=(float)ComputeNorm(input);

            for (int i = 0; i < input.Count; i++)
            {
                input.Values[i] /= norm;
            }

            return input;
        }

        public SparseVec[] Transform(SparseVec[] input)
        {
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = Transform(input[i]);
            }
            return input;
            
        }

        double ComputeNorm(SparseVec vec)
        {
            double sum = 0;
            for (int i = 0; i < vec.Count; i++)
            {
                sum += Math.Pow(Math.Abs(vec.Values[i]), Power);

            }
            return Math.Pow(sum, 1 / Power);
        }
    }
}
