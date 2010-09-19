using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    class Vector
    {

        public int Dim;

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
    }
}
