using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{

    public interface IVector<T>
    {

        
        float DotProuct();

        float DotProduct(T otherVec);
    }

    public class DeVec : IVector<DeVec>
    {

        public float DotProuct()
        {
            throw new NotImplementedException();
        }

        public float DotProduct(DeVec otherVec)
        {

            otherVec.DotProduct(this);
            throw new NotImplementedException();
        }
    }

    public class SpVec : IVector<SpVec>
    {

        public float DotProuct()
        {
            throw new NotImplementedException();
        }

        public float DotProduct(SpVec otherVec)
        {
            throw new NotImplementedException();
        }
    }


    public abstract class Vector
    {

        public int[] Indices { get; protected set; }

        public float[] Values { get; protected set; }

        /// <summary>
        /// Max index
        /// </summary>
        public int Dim;

        protected float SelfDotProd = float.NegativeInfinity;

        public  float DotProduct()
        {

            if (SelfDotProd < 0)
                throw new ArgumentOutOfRangeException("SelfDotProd is not computed");

            return SelfDotProd;

        }


    }


    public class SparseVec:Vector
    {

       

        /// <summary>
        /// number of non zero positions
        /// </summary>
        public int Count
        {
            get
            {
                return Indices.Length;
            }
        }

       

        public SparseVec(int dim, ICollection<int> indexes, ICollection<float> vals)
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
            int prevIdx = 0;
            while (enumerator1.MoveNext() && enumerator2.MoveNext())
            {
                if (prevIdx > enumerator1.Current)
                {
                    throw new ArgumentException("Indices should be in ascendig order");
                }
                Indices[k] = enumerator1.Current;
                Values[k] = enumerator2.Current;

                SelfDotProd += Values[k] * Values[k];
                k++;

                prevIdx = enumerator1.Current;
            }


        }

        public SparseVec(int dim, IList<KeyValuePair<int,float>> vec)
        {
            Dim = dim;

            
            Indices = new int[vec.Count];
            Values = new float[vec.Count];

            
            int prevIdx = 0;
            SelfDotProd = 0;
            for (int k = 0; k < vec.Count; k++)
            {
                var item = vec[k];

                if (prevIdx > item.Key)
                {
                    throw new ArgumentException("Indices should be in ascendig order");
                }
                Indices[k] = item.Key;
                Values[k] = item.Value;

                SelfDotProd += Values[k] * Values[k];
               
                prevIdx = item.Key;
            }


        }


        /// <summary>
        /// compute vector dot product
        /// </summary>
        /// <param name="otherVector"></param>
        /// <returns></returns>
        public float DotProduct(SparseVec otherVector)
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
