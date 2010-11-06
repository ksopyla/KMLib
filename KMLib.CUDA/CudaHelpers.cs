using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;

namespace KMLib.GPU
{

    /// <summary>
    /// some helper funtion to work with cuda
    /// </summary>
    public class CudaHelpers
    {
        public static void FillDenseVector(SparseVector mainVec,float[] fillVector)
        {
            Array.Clear(fillVector, 0, fillVector.Length);
            for (int j = 0; j < mainVec.mValueCount; j++)
            {
                int idx = mainVec.mIndices[j];
                float val = (float)mainVec.mValues[j];
                fillVector[idx] = val;
            }
        }

        /// <summary>
        /// Convert sparse vectors into
        /// </summary>
        /// <param name="vecVals"></param>
        /// <param name="vecIdx"></param>
        /// <param name="vecLenght"></param>
        /// <param name="problemElements"></param>
        public static void TransformToCSRFormat(out float[] vecVals, out int[] vecIdx, out int[] vecLenght,SparseVector[] problemElements)
        {
            //transform elements to specific array format -> CSR http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR_or_CRS.29
            int avgVectorLenght = problemElements[0].mValueCount;
            //list for all vectors values
            List<float> vecValsL = new List<float>(problemElements.Length * avgVectorLenght);

            //list for all vectors indexes
            List<int> vecIdxL = new List<int>(problemElements.Length * avgVectorLenght);

            //list of lenght of each vector, list of pointers
            List<int> vecLenghtL = new List<int>(problemElements.Length);

            //arrays for values, indexes and lenght

            int vecStartIdx = 0;
            for (int i = 0; i < problemElements.Length; i++)
            {
                var vec = problemElements[i];


                //!!!vector  not always has only one zero element at the end
                // mValues and mIndices have extra zero elements at the end, so 
                //after conversion we have to remove zeros from the end

                //coping and converting from double to float using Linq
                var converted = vec.mValues.Select(x => Convert.ToSingle(x)).Take(vec.mValueCount);
                vecValsL.AddRange(converted);

                vecIdxL.AddRange(vec.mIndices.Take(vec.mValueCount));


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vec.mValueCount;
            }

            //for last index
            vecLenghtL.Add(vecStartIdx);

            //convert list to arrays
            vecVals = vecValsL.ToArray();
            vecIdx = vecIdxL.ToArray();
            vecLenght = vecLenghtL.ToArray();

            //set list reference to null to free memeory
            vecIdxL = null;
            vecLenghtL = null;
            vecValsL = null;
        }


       
    }
}
