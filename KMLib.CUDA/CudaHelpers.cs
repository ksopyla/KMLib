using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;
using System.Diagnostics;
using KMLib.Helpers;
using GASS.CUDA.Types;
using GASS.CUDA;

namespace KMLib.GPU
{

    /// <summary>
    /// some helper funtion to work with cuda
    /// </summary>
    public class CudaHelpers
    {

        /// <summary>
        /// create dense vector based on sparese vector <see cref="mainVec"/>
        /// </summary>
        /// <param name="mainVec"></param>
        /// <param name="fillVector"></param>
        public static void FillDenseVector(SparseVec mainVec,float[] fillVector)
        {
            Array.Clear(fillVector, 0, fillVector.Length);
            for (int j = 0; j < mainVec.Count; j++)
            {
                int idx = mainVec.Indices[j];
                float val = (float)mainVec.Values[j];
                fillVector[idx] = val;
            }
        }

        /// <summary>
        /// Convert sparse vectors into CSR fromat (three array, one for values, one for indexes and one for vector pointers)
        /// </summary>
        /// <param name="vecVals"></param>
        /// <param name="vecIdx"></param>
        /// <param name="vecLenght"></param>
        /// <param name="problemElements"></param>
        public static void TransformToCSRFormat(out float[] vecVals, out int[] vecIdx, out int[] vecLenght,SparseVec[] problemElements)
        {
            //transform elements to specific array format -> CSR http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR_or_CRS.29
            int avgVectorLenght = problemElements[0].Count;
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
                //var converted = vec.mValues.Select(x => Convert.ToSingle(x)).Take(vec.mValueCount);
                //Array.ConstrainedCopy(vec.mValues, 0, vecVals, 0, vec.mValueCount);

                vecValsL.AddRange(vec.Values);

                vecIdxL.AddRange(vec.Indices);


                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += vec.Count;
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
        /// Convert sparse vectors into compact sparse column(CSC) fromat (three array, one for values, one for indexes and one for vector pointers)
        /// 
        /// 
        /// </summary>
        /// <param name="vecVals"></param>
        /// <param name="vecIdx"></param>
        /// <param name="vecLenght"></param>
        /// <param name="problemElements"></param>
        public static void TransformToCSCFormat(out float[] vecVals, out int[] vecIdx, out int[] vecLenght, SparseVec[] problemElements)
        {
            //transform elements to specific array format -> CSR http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR_or_CRS.29
            int avgVectorLenght = problemElements[0].Count;
            //list for all vectors values
            List<float> vecValsL = new List<float>(problemElements.Length * avgVectorLenght);

            //list for all vectors indexes
            List<int> vecIdxL = new List<int>(problemElements.Length * avgVectorLenght);


            int Dim = problemElements[0].Dim;
            //list of lenght of each vector, list of pointers
            List<int> vecLenghtL = new List<int>(Dim+1 );

            //arrays for values, indexes and lenght
            int[] elementsCheckedDims = new int[problemElements.Length];
            int vecStartIdx = 0;
            int curColSize = 0;
            
            for (int i = 1; i <= Dim; i++)
            {
                curColSize = 0;
                for (int k = 0; k < problemElements.Length; k++)
                {
                    var vec = problemElements[k];

                    int s=elementsCheckedDims[k];
                    
                    //find max index 
                    while (s < vec.Indices.Length && vec.Indices[s] < i)
                    {
                        s++;
                    }

                   // int val = Array.BinarySearch(vec.Indices, s, vec.Indices.Length, i);

                    elementsCheckedDims[k] = s;
                    if (s<vec.Indices.Length && vec.Indices[s] == i)
                    {
                        //insert in to vals and idx
                        vecValsL.Add(vec.Values[s]);
                        vecIdxL.Add(k);//k-th vector in k- column
                        curColSize++;

                    }

                }
                vecLenghtL.Add(vecStartIdx);
                vecStartIdx += curColSize;
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

        public static void TransformToCSCFormat2(out float[] vecVals, out int[] vecIdx, out int[] vecLenght, SparseVec[] problemElements)
        {
            //transform elements to specific array format -> CSR http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR_or_CRS.29
            int avgVectorLenght = problemElements[0].Count;
            int Dim = problemElements[0].Dim;

            List<int>[] colIdx = new List<int>[Dim+1];
            List<float>[] colVals = new List<float>[Dim+1];
            for (int i = 1; i <= Dim; i++)
            {
                colIdx[i] = new List<int>();
                colVals[i] = new List<float>();
            }
            
            //list for all vectors values
            List<float> vecValsL = new List<float>(problemElements.Length * avgVectorLenght);
            //list for all vectors indexes
            List<int> vecIdxL = new List<int>(problemElements.Length * avgVectorLenght);
            //list of lenght of each vector, list of pointers
            List<int> vecLenghtL = new List<int>(Dim + 1);
            
            for (int k = 0; k < problemElements.Length; k++)
            {
                var vec = problemElements[k];

                for (int s = 0; s < vec.Count; s++)
                {
                    int idx = vec.Indices[s];
                    float val = vec.Values[s];

                    colIdx[idx].Add(k);
                    colVals[idx].Add(val);
                }
            }


            int curColSize = 0;

            vecLenghtL.Add(curColSize);
            for (int i = 1; i <= Dim; i++)
            {
                curColSize+= colIdx[i].Count;
                vecLenghtL.Add(curColSize);
                vecIdxL.AddRange(colIdx[i]);
                vecValsL.AddRange(colVals[i]);

                colIdx[i] = null;
                colVals[i] = null;
            }

            
            
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
        /// Convert sparse vectors into ellpack-r format, all data has collumn majored ordering
        /// 
        /// </summary>
        /// <param name="vecVals"></param>
        /// <param name="vecCols"></param>
        /// <param name="rowLength"></param>
        /// <param name="problemElements"></param>
        public static void TransformToEllpackRFormat(out float[] vecVals,out int[] vecCols,out int[] rowLength, SparseVec[] problemElements)
        {
            int maxEl = 1;
            //for (int i = 0; i < problemElements.Length; i++)
            //{
            //    if(maxEl<problemElements[i].Count)
            //        maxEl=problemElements[i].Count;
            //}
            maxEl = (from m in problemElements
                     select m.Count).AsParallel().Max();

            double avgEl = (from t in problemElements
                            select t.Count).Average();

            int numRows = problemElements.Length;
            //2d array stored in 1d array
            vecVals = new float[numRows * maxEl];
            vecCols = new int[numRows * maxEl];
            //1d array
            rowLength = new int[numRows];

            for (int i = 0; i < numRows; i++)
            {
                var vec = problemElements[i];
                for (int j = 0; j < vec.Count; j++)
                {
                    vecVals[j * numRows + i] = vec.Values[j];
                    vecCols[j * numRows + i] = vec.Indices[j];
                }
                rowLength[i] = vec.Count;
            }

        }


        /// <summary>
        /// Convert sparse vector into slice ellpack format, all data has column majored ordering, with group of <see cref="threadPerRow"/> elements
        /// </summary>
        /// <param name="vecVals">vector values</param>
        /// <param name="vecCols"></param>
        /// <param name="sliceStart"></param>
        /// <param name="rowLenght"></param>
        /// <param name="problemElements"></param>
        /// <param name="threadsPerRow"></param>
        /// <param name="sliceSize"></param>
        public static void TransformToSlicedEllpack(out float[] vecVals, out int[] vecCols,out int[] sliceStart,out int[] rowLenght, SparseVec[] problemElements, int threadsPerRow, int sliceSize)
        {
            //int align = 128 *(int) Math.Ceiling((float)(sliceSize * threadsPerRow) / 128);

            
            int align = (int)Math.Ceiling(sliceSize * threadsPerRow / 64.0)*64;
            //int align = (int)Math.Ceiling(sliceSize * threadsPerRow / 2.0) * 2;
            int align2 = (int)Math.Ceiling(1.0 * sliceSize * threadsPerRow / 64) * 64;

            //Debug.Assert(align == align2);

            int numRows = problemElements.Length;
            int numSlices = (int)Math.Ceiling( (numRows+0.0)/ sliceSize);

            rowLenght = new int[numRows];

            sliceStart = new int[numSlices+1];
            //max non-zero in slice
            int[] sliceMax = new int[numSlices];

            int sliceNr = 0;
            //find max in slice
            for (int i = 0; i < numSlices; i++)
            {
                sliceMax[i] = -1;
                int idx = -1;
                for (int j = 0; j < sliceSize; j++)
                {
                    idx = j + i * sliceSize;
                    if (idx < numRows)
                    {
                        rowLenght[idx] = problemElements[idx].Count;
                        if (sliceMax[i] < rowLenght[idx])
                        {
                            sliceMax[i] = rowLenght[idx];
                        }
                    }
                }
                sliceStart[i + 1] = sliceStart[i] + (int)Math.Ceiling((sliceMax[i]+0.0) / threadsPerRow) * align;
                //var ttt = sliceStart[i] + sliceMax[i]*sliceSize ;

            }

            //
            int nnzEl = sliceStart[numSlices];
            vecCols = new int[nnzEl];
            vecVals = new float[nnzEl];

            sliceNr = 0;
            int rowInSlice = 0;
            //fill slice ellpack values and cols arrays
            for (int i = 0; i < numRows; i++)
            {
                //slice number in whole dataset
                sliceNr = i / sliceSize;
                //row number  in particular slice
                rowInSlice = i % sliceSize;
                var vec = problemElements[i];

                int threadNr = -1;
                float value = 0;
                int col = -1;

                int rowSlice = -1;// (int)Math.Ceiling((0.0 + vec.Count) / threadsPerRow);
                for (int k = 0; k < vec.Count; k++)
                {
                    threadNr = k % threadsPerRow;
                    rowSlice = k / threadsPerRow;
                    value = vec.Values[k];
                    col = vec.Indices[k];
                    //eg. if sliceSize=8, threadsPerRow=4, for first vector (i=0) with size 9
                    //computed idx should be= [0 1 2 3 , 32,33,34,35, 64]
                    int idx = sliceStart[sliceNr] + align * rowSlice + rowInSlice * threadsPerRow + threadNr;
                    
                    vecVals[idx]=value;
                    vecCols[idx] = col;
                }

            }
            
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
        ///  matrix is in sparse matrix CSR format
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


        internal static void InitBuffer(SparseVec sparseVector, IntPtr bufferPtr)
        {
            unsafe
            {

                float* vecPtr = (float*)bufferPtr.ToPointer();

                for (int j = 0; j < sparseVector.Count; j++)
                {
                    int idx = sparseVector.Indices[j];
                    float val = (float)sparseVector.Values[j];
                    vecPtr[idx] = val;


                }

            }
        }

        /// <summary>
        /// set the value on position which are the same as sparse vector indexes
        /// </summary>
        /// <param name="sparseVector"></param>
        /// <param name="bufferPtr"></param>
        /// <param name="value"></param>
        internal static void SetBufferIdx(SparseVec sparseVector, IntPtr bufferPtr, float value)
        {
            unsafe
            {

                float* vecPtr = (float*)bufferPtr.ToPointer();
                for (int j = 0; j < sparseVector.Count; j++)
                {
                    int idx = sparseVector.Indices[j];
                    vecPtr[idx] = value;
                }

            }
        }


        internal static void SetTextureMemory(CUDA cuda,CUmodule cuModule, ref CUtexref texture, string texName, float[] data, ref CUdeviceptr memPtr)
        {
            texture = cuda.GetModuleTexture(cuModule, texName);
            memPtr = cuda.CopyHostToDevice(data);
            cuda.SetTextureAddress(texture, memPtr, (uint)(sizeof(float) * data.Length));

        }
        internal static void SetTextureMemory(CUDA cuda, ref CUtexref texture, string texName, float[] data, ref CUdeviceptr memPtr)
        {
            texture = cuda.GetModuleTexture(texName);
            memPtr = cuda.CopyHostToDevice(data);
            cuda.SetTextureAddress(texture, memPtr, (uint)(sizeof(float) * data.Length));

        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="size">array size</param>
        /// <param name="maxBlocksPerGrid">maximum block size, good starting value is 64</param>
        /// <param name="maxThreadsPerBlock"></param>
        /// <param name="threads"></param>
        /// <param name="blocks"></param>
        internal static void GetNumThreadsAndBlocks(int size, int maxBlocksPerGrid, int maxThreadsPerBlock, ref int threads, ref int blocks)
        {

            //threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
            //blocks = (n + (threads * 2 - 1)) / (threads * 2);
            //blocks = MIN(maxBlocks, blocks);
            threads = (size < 2 * maxThreadsPerBlock) ? nextPow2((size + 1) / 2) : maxThreadsPerBlock;

            blocks = (size + (threads * 2 - 1)) / (threads * 2);
            blocks = Math.Min(blocks, maxBlocksPerGrid);
        }


        internal static int nextPow2(int x)
        {
            if (x < 0)
                throw new ArgumentException("x should be grateher than 0");

            --x;
            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;
            return ++x;
        }
    }
}
