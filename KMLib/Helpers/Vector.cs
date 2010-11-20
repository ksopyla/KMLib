using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    internal class Vector
    {

        /// <summary>
        /// Max index
        /// </summary>
        public int Dim;

        /// <summary>
        /// number of non zero positions
        /// </summary>
        public int Count;

        public int[] Indices;

        public float[] Values;

        public Vector(int dim, ICollection<int> indexes, ICollection<float> vals)
        {
            Dim = dim;

            if (indexes.Count != vals.Count)
            {
                throw new ArgumentOutOfRangeException("collections have different sizes");
            }
            Indices = new int[indexes.Count];
            Values = new float[vals.Count];

            var enumerator1 = indexes.GetEnumerator();
            var enumerator2 = vals.GetEnumerator();

            int k = 0;
            while (enumerator1.MoveNext() && enumerator2.MoveNext())
            {
                Indices[k] = enumerator1.Current;
                Values[k] = enumerator2.Current;
                k++;
            }


        }


        /// <summary>
        /// compute vector dot product
        /// </summary>
        /// <param name="otherVector"></param>
        /// <returns></returns>
        public float DotProduct(Vector otherVector)
        {
           
            if (otherVector == null)
            {
                throw new ArgumentNullException("otherVector");
            }

            if (otherVector.Dim != Dim)
            {
                throw new ArgumentException("different dimensions", "otherVector");
            }

            float result = 0;
            

            if (Count < 1)
                return 0.0f;

            if (otherVector.Count < 1)
                return 0.0f;

            int i1 = 0;
            int i2 = 0;

            while (i1 < this.Count && i2 < otherVector.Count)
            {
                int index1 = Indices[i1];
                int index2 = otherVector.Indices[i2];

                if (index1 == index2)
                {
                    float mul = Values[i1] * otherVector.Values[i2];
                    result += mul;
                    i1++; i2++;
                }
                else if (index1 < index2)
                {
                    i1++;
                }
                else
                {
                    i2++;
                }
            }

            return result;

        }
    }
}
