using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using System.Diagnostics;

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
           // Stopwatch timer = Stopwatch.StartNew();
            for (int i = 0; i < problemElements.Length; i++)
            {
                var vec = problemElements[i];


                //!!!vector  not always has only one zero element at the end
                // mValues and mIndices have extra zero elements at the end, so 
                //after conversion we have to remove zeros from the end

                //coping and converting from double to float using Linq
                //var converted = vec.mValues.Take(vec.mValueCount).Select(x => Convert.ToSingle(x));
                var converted = vec.mValues.Select(x => Convert.ToSingle(x)).Take(vec.mValueCount);
                //Array.ConstrainedCopy(vec.mValues, 0, vecVals, 0, vec.mValueCount);

                vecValsL.AddRange(converted);

                vecIdxL.AddRange(vec.mIndices.Take(vec.mValueCount));


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vec.mValueCount;
            }
          //  timer.Stop();


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

        /// <summary>
        ///  sets the values from one row of matrix, 
        ///  matrix is in sparse matrix in CSR format
        /// </summary>
        /// <param name="matVals">matrx values</param>
        /// <param name="matIdx">matrix indices</param>
        /// <param name="matRowLenght">matrix rows lenght</param>
        /// <param name="index">row index</param>
        ///<param name="bufferPtr">pointer to float dense vector</param>
        unsafe public static void InitBuffer(float[] matVals, int[] matIdx, int[] matRowLenght, int index, IntPtr bufferPtr)
        {

            unsafe
            {

                float* vecPtr = (float*)bufferPtr.ToPointer();

                for (int j = matRowLenght[index]; j < matRowLenght[index + 1]; j++)
                {
                    int idx = matIdx[j];
                    float val = matVals[j];
                    vecPtr[idx] = val;


                }

            }

        }

        /// <summary>
        ///  sets the value for one matrix row, 
        ///  matrix is in sparse matrix in CSR format
        /// </summary>
        /// <param name="matVals">matrx values</param>
        /// <param name="matIdx">matrix indices</param>
        /// <param name="matRowLenght">matrix rows lenght</param>
        /// <param name="index">row index</param>
        ///<param name="bufferPtr">pointer to float dense vector</param>
        unsafe public static void SetBufferIdx(int[] matIdx, int[] matRowLenght, int index, IntPtr bufferPtr, float value)
        {
            if (index < 0)
                return;
            unsafe
            {

                float* vecPtr = (float*)bufferPtr.ToPointer();

                for (int j = matRowLenght[index]; j < matRowLenght[index + 1]; j++)
                {
                    int idx = matIdx[j];
                    vecPtr[idx] = value;
                }

            }
        }


        internal static void InitBuffer(SparseVector sparseVector, IntPtr bufferPtr)
        {
            unsafe
            {

                float* vecPtr = (float*)bufferPtr.ToPointer();

                for (int j = 0; j < sparseVector.mValueCount; j++)
                {
                    int idx = sparseVector.mIndices[j];
                    float val = (float)sparseVector.mValues[j];
                    vecPtr[idx] = val;


                }

            }
        }

        internal static void SetBufferIdx(SparseVector sparseVector, IntPtr bufferPtr, float value)
        {
            unsafe
            {

                float* vecPtr = (float*)bufferPtr.ToPointer();
                for (int j = 0; j < sparseVector.mValueCount; j++)
                {
                    int idx = sparseVector.mIndices[j];
                    vecPtr[idx] = value;
                }

            }
        }
    }
}
