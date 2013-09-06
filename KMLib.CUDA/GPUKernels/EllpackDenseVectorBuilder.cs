/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using GASS.CUDA.Types;
using GASS.CUDA;
using System.IO;
using System.Diagnostics;

namespace KMLib.GPU.GPUKernels
{

    /// <summary>
    /// This class is used by gpu kernels to build dense vector on GPU
    /// </summary>
    public class EllpackDenseVectorBuilder
    {
        CUdeviceptr vecPtr;
        CUdeviceptr valsPtr, idxPtr,vecLengthPtr;

        CUDA cuda;
        CUfunction cuFuncDense;
        CUmodule cuModule;

        string funcName = "makeDenseVectorEllpack";
        string cudaModuleName = "cudaSVMKernels.cubin";
        private int threadsPerBlock=128;
        
        private int mainVecIdxParamOffset;
        private uint mainVectorIdx;
        private uint nrRows;
        private uint vecDim;
        private int blocksPerGrid;


        public EllpackDenseVectorBuilder(CUDA cu,CUdeviceptr vector, CUdeviceptr vals,CUdeviceptr cols,CUdeviceptr length,int rows,int dim)
        {
            cuda = cu;
            vecPtr = vector;
            valsPtr = vals;
            idxPtr = cols;
            vecLengthPtr = length;
            nrRows = (uint)rows;
            vecDim = (uint)dim;

            blocksPerGrid = (int) Math.Ceiling( (vecDim + 0.0) / threadsPerBlock);

            var blocksPerGrid1 = (vecDim + threadsPerBlock - 1) / threadsPerBlock;

            Debug.Assert(blocksPerGrid == blocksPerGrid1);

        }


        public void Init()
        {
            InitCudaModule();

            SetFunctionParams();


        }

        private void SetFunctionParams()
        {
            cuda.SetFunctionBlockShape(cuFuncDense, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncDense, offset, valsPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncDense, offset, idxPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDense, offset, vecLengthPtr.Pointer);
            offset += IntPtr.Size;

            
            cuda.SetParameter(cuFuncDense, offset, vecPtr.Pointer);
            offset += IntPtr.Size;

            
            mainVecIdxParamOffset = offset;
            cuda.SetParameter(cuFuncDense, offset, (uint)mainVectorIdx);
            offset += sizeof(int);

            cuda.SetParameter(cuFuncDense, offset, (uint)nrRows);
            offset += sizeof(int);

            cuda.SetParameter(cuFuncDense, offset, (uint)vecDim);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncDense, (uint)offset);

        }

        private void InitCudaModule()
        {
            string modluePath = Path.Combine(Environment.CurrentDirectory, cudaModuleName);
            if (!File.Exists(modluePath))
                throw new ArgumentException("Failed access to cuda module" + modluePath);

            cuModule = cuda.LoadModule(modluePath);
            cuFuncDense = cuda.GetModuleFunction(funcName);
        }


        public void BuildDenseVector(int idx)
        {
            cuda.SetParameter(cuFuncDense, mainVecIdxParamOffset, idx);
            cuda.Launch(cuFuncDense, blocksPerGrid, 1);
            cuda.SynchronizeContext();

            //only for test
            //float[] result1 = new float[vecDim+1];
            //cuda.CopyDeviceToHost(vecPtr, result1);

        }


    }
}
