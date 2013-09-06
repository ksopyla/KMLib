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
using KMLib.Helpers;
using KMLib.Kernels;
using KMLib.SVMSolvers;
using GASS.CUDA;
using System.IO;
using GASS.CUDA.Types;
using System.Diagnostics;

namespace KMLib.GPU.Solvers
{

    /// <summary>
    /// SVM GPU L2 - solver
    ///  Solves:
    /// Min 0.5(\alpha^T Q \alpha) + p^T \alpha
    /// y^T \alpha = \delta
    /// y_i = +1 or -1
    /// 0 <= alpha_i <= Cp for y_i = 1
    /// 0 <= alpha_i <= Cn for y_i = -1
    /// solution will be put in \alpha, objective value will be put in obj
    /// </summary>
    /// <remarks>
    /// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918, implementation based on 
    ///  LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    ///  
    /// </remarks>
    public class GPUSmoFanSolver : Solver<SparseVec>, IDisposable
    {

        private CuVectorKernel gpuKernel;


        /// <summary>
        /// kernel diagonal
        /// </summary>
        private float[] QD;
        /// <summary>
        /// labels
        /// </summary>
        protected float[] y;
        /// <summary>
        /// gradient
        /// </summary>
        protected float[] G;		// gradient of objective function

        private float[] alpha;

        protected float EPS = 0.001f;

        protected const float INF = float.PositiveInfinity;



        private int problemSize;
        private CUDA cuda;
        private GASS.CUDA.Types.CUmodule cuModule;
        private string cudaModuleName = "gpuFanSmoSolver.cubin";
        private GASS.CUDA.Types.CUfunction cuFuncFindMaxI;
        private string funcFindMaxI = "FindMaxIdx";
        private GASS.CUDA.Types.CUfunction cuFuncFindMinJ;
        private string funcFindMinJ = "FindMinIdx";
        private GASS.CUDA.Types.CUfunction cuFuncFindStopping;
        private string funcFindStoping = "FindStoppingGradVal";
        private GASS.CUDA.Types.CUfunction cuFuncUpdateG;
        private string funcUpdateGFunc = "UpdateGrad";


        /// <summary>
        /// maximum reduction blocks, def=64
        /// </summary>
        private int maxReductionBlocks = 64;
        /// <summary>
        /// threads per block, def=128,
        /// </summary>
        /// <remarks>
        /// This value is connected with constant definded in gpuFanSmoSolver.cu BLOCK_SIZE
        /// they should be equal
        /// </remarks>
        private int threadsPerBlock = 128;

        /// <summary>
        /// number of blocks used for reduction, grid size
        /// </summary>
        private int reductionBlocks;
        /// <summary>
        /// numer of theread used for reduction
        /// </summary>
        private int reductionThreads;

        /// <summary>
        /// stores gradients value after GPU reduction, size is equal as reductionBlocks
        /// </summary>
        private float[] reduceVal;
        /// <summary>
        /// stores idx after GPU reduction, size is equal reductionBlocks
        /// </summary>
        private int[] reduceIdx;

        private CUdeviceptr alphaPtr;
        private GASS.CUDA.Types.CUdeviceptr gradPtr;
        private GASS.CUDA.Types.CUdeviceptr yPtr;
        private GASS.CUDA.Types.CUdeviceptr kernelDiagPtr;
        private GASS.CUDA.Types.CUdeviceptr kiPtr;
        private GASS.CUDA.Types.CUdeviceptr kjPtr;
        private GASS.CUDA.Types.CUdeviceptr valRedPtr;
        private GASS.CUDA.Types.CUdeviceptr idxRedPtr;
        private CUdeviceptr constCPtr;
        private CUcontext cuCtx;

        #region cuda function param offsets
        
        private int GMaxParamOffsetInMinJ;
        private int QD_iParamOffsetInMinJ;
        private int yiParamOffsetInMinJ;
        private object diff_i;

        private int diff_j_ParamOffsetInUpgGrad;
        private int diff_i_ParamOffsetInUpgGrad;
        #endregion
        /// <summary>
        /// number of thhreads for updateing gradient
        /// </summary>
        private int updGThreadsPerBlock;
        private int updGBlocksPerGrid;
        private int iter;
        private int MaxIter=3000000;


        public GPUSmoFanSolver(Problem<SparseVec> problem, IKernel<SparseVec> kernel, float C)
            : base(problem, kernel, C)
        {
            this.C = C;
            problemSize = problem.ElementsCount;

            y = problem.Y;
            QD = kernel.DiagonalDotCache;
            alpha = new float[problemSize];
            G = new float[problemSize];
            gpuKernel = (CuVectorKernel)kernel;
        }


        /// <summary>
        /// Computes model by solving optimization problem
        /// </summary>
        /// <returns>Model</returns>
        public override Model<SparseVec> ComputeModel()
        {
            Stopwatch timer = Stopwatch.StartNew();

            int problemSize = problem.ElementsCount;
            //float[] alphaResult = new float[problem.ElementsCount];

            SolutionInfo si = new SolutionInfo();
            Solve(problem.Y, si);

            timer.Stop();
            Model<SparseVec> model = new Model<SparseVec>();
            model.NumberOfClasses = 2;
            model.Alpha = alpha;//alphaResult;
            model.Bias = si.rho;
            model.Iter = si.iter;
            model.Obj = si.obj;
            model.ModelTime = timer.Elapsed;
            model.ModelTimeMs = timer.ElapsedMilliseconds;

            //------------------
            List<SparseVec> supportElements = new List<SparseVec>(alpha.Length);
            List<int> suporrtIndexes = new List<int>(alpha.Length);
            List<float> supportLabels = new List<float>(alpha.Length);
            for (int j = 0; j < alpha.Length; j++)
            {
                if (Math.Abs(alpha[j]) > 0)
                {
                    supportElements.Add(problem.Elements[j]);
                    suporrtIndexes.Add(j);
                    supportLabels.Add(problem.Y[j]);
                }

            }
            model.SupportElements = supportElements.ToArray();
            model.SupportElementsIndexes = suporrtIndexes.ToArray();
            model.Y = supportLabels.ToArray();



            return model;
        }

        /// <summary>
        /// Solves the optimization problem
        /// </summary>
        /// <param name="minusOnes"></param>
        /// <param name="y_"></param>
        /// <param name="alpha_"></param>
        /// <param name="si"></param>
        /// <param name="shrinking"></param>
        private void Solve(float[] y, SolutionInfo si)
        {

            // initialize gradient
            {   
                int i;
                for (i = 0; i < problemSize; i++)
                {
                    G[i] = -1;
                    alpha[i] = 0;
                }
            }


            InitCudaModule();

            SetCudaData();
            // optimization step
            float GMaxI=0;
            float GMaxJ=0;
            iter = 1;


            while (iter<MaxIter)
            {

                //Find i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha)
                Tuple<int, float> maxPair = FindMaxPair();
                int i = maxPair.Item1;
                GMaxI = maxPair.Item2;
                
                if (iter % 250 == 0)
                {
                    GMaxJ = FindStoppingGradVal();

                    if (GMaxI - GMaxJ < EPS)
                        break;
                }

                //Compute i-th kernel collumn, set the specific memory region on GPU, 
                ComputeKernel(i, kiPtr);


                // j: mimimizes the decrease of obj value
                //    (if quadratic coefficeint <= 0, replace it with tau)
                //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
                int j = FindMinPair(i, GMaxI);




                //Compute j-th kernel collumn
                ComputeKernel(j, kjPtr);


                float old_alpha_i = alpha[i];
                float old_alpha_j = alpha[j];

                //update alpha - serial code, one iteration
                UpdateAlpha(i, j);
                // update gradient G
                float delta_alpha_i = alpha[i] - old_alpha_i;
                float delta_alpha_j = alpha[j] - old_alpha_j;


                UpdateGrad(i, j, delta_alpha_i, delta_alpha_j);

                //czy to potrzebne???
                // update alpha_status and G_bar,
                iter++;
            }//end while

            
            cuda.CopyDeviceToHost(gradPtr, G);

            cuda.SynchronizeContext();
            // calculate rho
            si.rho = calculate_rho();

            //because we start from 1 not from 0;
            si.iter = iter - 1;
            // calculate objective value
            {
                float v = 0;
                int i;
                for (i = 0; i < problemSize; i++)
                    v += alpha[i] * (G[i] - 1);

                si.obj = v / 2;
            }


            si.upper_bound_p = C;
            si.upper_bound_n = C;
        }



        private void UpdateAlpha(int i, int j)
        {
            float Qii = QD[i];
            float Qjj = QD[j];
            float yi = y[i];
            float yj = y[j];
            float Qij = 0;

            //todo: remove it only for testing
            //float[] a = new float[alpha.Length];
            //cuda.CopyDeviceToHost(kiPtr, a);

            //copy Qij form device to host
            float[] temp = new float[] { 0.0f };
            cuda.CopyDeviceToHost(kiPtr + sizeof(float) * j, temp);
            Qij = temp[0];

            //copy Grad[i] from device to host
            float[] gradI = new float[] { 0.0f };
            cuda.CopyDeviceToHost(gradPtr + sizeof(float) * i, gradI);
            G[i] = gradI[0];
            //copy Grad[j] from device to host
            float[] gradJ = new float[] { 0.0f };
            cuda.CopyDeviceToHost(gradPtr + sizeof(float) * j, gradJ);
            G[j] = gradJ[0];


            cuda.SynchronizeContext();

            ////copy alpha[i]
            //float[] aI = new float[] { 0.0f };
            //cuda.CopyDeviceToHost(alphaPtr + sizeof(float) * i, aI);
            //alpha[i] = aI[0];

            //float[] aJ = new float[] { 0.0f };
            //cuda.CopyDeviceToHost(alphaPtr + sizeof(float) * j, aJ);
            //alpha[j] = aJ[0];



            float quad_coef = Qii + Qjj - 2 * yi * yj * Qij;
            if (quad_coef <= 0)
                quad_coef = 1e-12f;

            float delta = 0;
            float diff = 0;
            float sum = 0;

            if (y[i] != y[j])
            {
                 delta = (-G[i] - G[j]) / quad_coef;
                 diff = alpha[i] - alpha[j];
                alpha[i] += delta;
                alpha[j] += delta;

                if (diff > 0)
                {
                    if (alpha[j] < 0)
                    {
                        alpha[j] = 0;
                        alpha[i] = diff;
                    }
                }
                else
                {
                    if (alpha[i] < 0)
                    {
                        alpha[i] = 0;
                        alpha[j] = -diff;
                    }
                }
                if (diff > 0)
                {
                    if (alpha[i] > C)
                    {
                        alpha[i] = C;
                        alpha[j] = C - diff;
                    }
                }
                else
                {
                    if (alpha[j] > C)
                    {
                        alpha[j] = C;
                        alpha[i] = C + diff;
                    }
                }
            }
            else
            {
                delta = (G[i] - G[j]) / quad_coef;
                sum = alpha[i] + alpha[j];
                alpha[i] -= delta;
                alpha[j] += delta;

                if (sum > C)
                {
                    if (alpha[i] > C)
                    {
                        alpha[i] = C;
                        alpha[j] = sum - C;
                    }
                }
                else
                {
                    if (alpha[j] < 0)
                    {
                        alpha[j] = 0;
                        alpha[i] = sum;
                    }
                }
                if (sum > C)
                {
                    if (alpha[j] > C)
                    {
                        alpha[j] = C;
                        alpha[i] = sum - C;
                    }
                }
                else
                {
                    if (alpha[i] < 0)
                    {
                        alpha[i] = 0;
                        alpha[j] = sum;
                    }
                }
            }

            //set alpha on device
            cuda.CopyHostToDevice(alphaPtr + sizeof(float) * i, new float[] { alpha[i] });
            cuda.CopyHostToDevice(alphaPtr + sizeof(float) * j, new float[] { alpha[j] });

            //todo: remove it
            //cuda.SynchronizeContext();

            //todo: remove it, only for debuging
            //cuda.CopyDeviceToHost(alphaPtr, a);

        }

        /// <summary>
        /// Updates gradients after alpha changes
        ///  for (int k = 0; k < active_size; k++)
        ///  {
        ///    G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
        ///   }
        /// </summary>
        /// <param name="i"></param>
        /// <param name="j"></param>
        /// <param name="delta_alpha_i"></param>
        /// <param name="delta_alpha_j"></param>
        private void UpdateGrad(int i, int j, float delta_alpha_i, float delta_alpha_j)
        {
            //float[] t = new float[G.Length];
            //float[] t1 = new float[G.Length];
            //cuda.CopyDeviceToHost(gradPtr, t);
            
            //var KI = Enumerable.Repeat(1.0f, G.Length).ToArray();
            //var KJ = Enumerable.Repeat(1.0f, G.Length).ToArray();
            //cuda.CopyHostToDevice(kiPtr, KI);
            //cuda.CopyHostToDevice(kjPtr, KJ);
            //delta_alpha_i = 0.1f;
            //delta_alpha_j = 0.2f;

            cuda.SetParameter(cuFuncUpdateG,diff_i_ParamOffsetInUpgGrad,delta_alpha_i);
            cuda.SetParameter(cuFuncUpdateG,diff_j_ParamOffsetInUpgGrad,delta_alpha_j);
            cuda.Launch(cuFuncUpdateG, updGBlocksPerGrid, 1);
            cuda.SynchronizeContext();

            
            //cuda.CopyDeviceToHost(gradPtr, t1);
            
        }

      


        private Tuple<int, float> FindMaxPair()
        {
            cuda.Launch(cuFuncFindMaxI, reductionBlocks, 1);

            cuda.CopyDeviceToHost(valRedPtr, reduceVal);
            cuda.CopyDeviceToHost(idxRedPtr, reduceIdx);
            cuda.SynchronizeContext();

            float max = float.NegativeInfinity;
            int idx = -1;
            for (int i = 0; i < reductionBlocks; i++)
            {
                if (max < reduceVal[i])
                {
                    max = reduceVal[i];
                    idx = reduceIdx[i];
                }
            }

            return new Tuple<int, float>(idx, max);

        }

        private int FindMinPair(int i, float GmaxI)
        {
            //float[] tempData = new float[y.Length];
            //cuda.CopyDeviceToHost(yPtr, tempData);
            //cuda.CopyDeviceToHost(alphaPtr, tempData);
            //cuda.CopyDeviceToHost(gradPtr, tempData);
            //cuda.CopyDeviceToHost(kiPtr, tempData);
            //cuda.CopyDeviceToHost(kernelDiagPtr, tempData);


            cuda.SetParameter(cuFuncFindMinJ, GMaxParamOffsetInMinJ, GmaxI);
            cuda.SetParameter(cuFuncFindMinJ, QD_iParamOffsetInMinJ, QD[i]);
            cuda.SetParameter(cuFuncFindMinJ, yiParamOffsetInMinJ, y[i]);

            cuda.Launch(cuFuncFindMinJ, reductionBlocks, 1);
            cuda.CopyDeviceToHost(valRedPtr, reduceVal);
            cuda.CopyDeviceToHost(idxRedPtr, reduceIdx);
            cuda.SynchronizeContext();

            float max = float.NegativeInfinity;
            int idx = -1;
            for (int k = 0; k < reduceVal.Length; k++)
            {
                if (max < reduceVal[k])
                {
                    max = reduceVal[k];
                    idx = reduceIdx[k];
                }
            }

            return idx;
        }


        /// <summary>
        /// compute kernel and sets memory pointed by ptr (onec for i-th, once for j-th kernel column)
        /// </summary>
        /// <param name="i"></param>
        /// <param name="ptr"></param>
        private void ComputeKernel(int i, CUdeviceptr ptr)
        {
            gpuKernel.AllProductsGPU(i, ptr);

        }

        private float FindStoppingGradVal()
        {
            cuda.Launch(cuFuncFindStopping, reductionBlocks, 1);
            cuda.CopyDeviceToHost(valRedPtr, reduceVal);
            cuda.SynchronizeContext();
            
            float min = float.PositiveInfinity;
            for (int k = 0; k < reductionBlocks; k++)
            {
                if (min > reduceVal[k])
                {
                    min = reduceVal[k];
                }
            }

            return min;

        }

        private void InitCudaModule()
        {
            cuda = gpuKernel.cuda;

            //cuda = new CUDA(0, true);
            //cuCtx = cuda.CreateContext(0, CUCtxFlags.MapHost);
            //cuda.SetCurrentContext(cuCtx);

            string modluePath = Path.Combine(Environment.CurrentDirectory, cudaModuleName);
            if (!File.Exists(modluePath))
                throw new ArgumentException("Failed access to cuda module" + modluePath);

            cuModule = cuda.LoadModule(modluePath);
            cuFuncFindMaxI = cuda.GetModuleFunction(funcFindMaxI);
            cuFuncFindMinJ = cuda.GetModuleFunction(funcFindMinJ);
            cuFuncFindStopping = cuda.GetModuleFunction(funcFindStoping);
            cuFuncUpdateG = cuda.GetModuleFunction(funcUpdateGFunc);


        }


        private void SetCudaData()
        {

            

            CudaHelpers.GetNumThreadsAndBlocks(problemSize, maxReductionBlocks, threadsPerBlock, ref reductionThreads, ref reductionBlocks);

            alphaPtr = cuda.CopyHostToDevice(alpha);
            gradPtr = cuda.CopyHostToDevice(G);
            yPtr = cuda.CopyHostToDevice(y);
            kernelDiagPtr = cuda.CopyHostToDevice(QD);

            //kernel columns i,j is simpler to copy array of zeros 
            kiPtr = cuda.CopyHostToDevice(alpha);
            kjPtr = cuda.CopyHostToDevice(alpha);

            //todo:remove it
            int redSize = reductionThreads; //reductionBlocks
            reduceVal = new float[redSize];
            reduceIdx = new int[redSize];

            
            valRedPtr = cuda.CopyHostToDevice(reduceVal);
            idxRedPtr = cuda.CopyHostToDevice(reduceIdx);


            constCPtr = cuda.GetModuleGlobal(cuModule, "C");
            float[] cData = new float[] { C };
            cuda.CopyHostToDevice(constCPtr, cData);

            SetCudaParams();

        }

        private void SetCudaParams()
        {

            #region Set cuda function parmeters for finding Max Idx

            cuda.SetFunctionBlockShape(cuFuncFindMaxI, reductionThreads, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncFindMaxI, offset, yPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, gradPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, idxRedPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, valRedPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxI, offset, (uint)problemSize);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncFindMaxI, (uint)offset);
            #endregion



            #region Set cuda function parmeters for computing Min Idx

            cuda.SetFunctionBlockShape(cuFuncFindMinJ, reductionThreads, 1, 1);

            offset = 0;
            cuda.SetParameter(cuFuncFindMinJ, offset, yPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMinJ, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMinJ, offset, gradPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMinJ, offset, kiPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMinJ, offset, kernelDiagPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMinJ, offset, idxRedPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMinJ, offset, valRedPtr.Pointer);
            offset += IntPtr.Size;

            GMaxParamOffsetInMinJ = offset;
            cuda.SetParameter(cuFuncFindMinJ, offset, 0.0f);
            offset += sizeof(float);
            QD_iParamOffsetInMinJ = offset;
            cuda.SetParameter(cuFuncFindMinJ, offset, 0.0f);
            offset += sizeof(float);
            yiParamOffsetInMinJ = offset;
            cuda.SetParameter(cuFuncFindMinJ, offset, 0.0f);
            offset += sizeof(float);

            cuda.SetParameter(cuFuncFindMinJ, offset, (uint)problemSize);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncFindMinJ, (uint)offset);
            #endregion

            #region Set cuda function parmeters for computing Min value for stopping

            cuda.SetFunctionBlockShape(cuFuncFindStopping, reductionThreads, 1, 1);

            offset = 0;
            cuda.SetParameter(cuFuncFindStopping, offset, yPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindStopping, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindStopping, offset, gradPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindStopping, offset, valRedPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindStopping, offset, (uint)problemSize);
            offset += sizeof(int);
            cuda.SetParameterSize(cuFuncFindStopping, (uint)offset);
            #endregion

            #region set cuda function parma for gradient updateing

            updGThreadsPerBlock = 64;
            //4 operaation for threads,
            updGBlocksPerGrid = (problemSize + 4 * updGThreadsPerBlock - 1) / (4 * updGThreadsPerBlock);

            cuda.SetFunctionBlockShape(cuFuncUpdateG, updGThreadsPerBlock, 1, 1);

            offset = 0;
            cuda.SetParameter(cuFuncUpdateG, offset, kiPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncUpdateG, offset, kjPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncUpdateG, offset, gradPtr.Pointer);
            offset += IntPtr.Size;
            diff_i_ParamOffsetInUpgGrad = offset;
            cuda.SetParameter(cuFuncUpdateG, offset, 0.0f);
            offset += sizeof(float);
            diff_j_ParamOffsetInUpgGrad = offset;
            cuda.SetParameter(cuFuncUpdateG, offset, 0.0f);
            offset += sizeof(float);
            cuda.SetParameter(cuFuncUpdateG, offset, (uint)problemSize);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncUpdateG, (uint)offset);
            #endregion
        }

        float calculate_rho()
        {
            float r;
            int nr_free = 0;
            float ub = INF, lb = -INF, sum_free = 0;
            for (int i = 0; i < problemSize; i++)
            {
                float yG = y[i] * G[i];

                if (alpha[i]==0)
                {
                    if (y[i] > 0)
                        ub = Math.Min(ub, yG);
                    else
                        lb = Math.Max(lb, yG);
                }
                else if (alpha[i]==C)
                {
                    if (y[i] < 0)
                        ub = Math.Min(ub, yG);
                    else
                        lb = Math.Max(lb, yG);
                }
                else
                {
                    ++nr_free;
                    sum_free += yG;
                }
            }

            if (nr_free > 0)
                r = sum_free / nr_free;
            else
                r = (ub + lb) / 2;

            return r;
        }


        public void Dispose()
        {

            if (cuda != null)
            {
                cuda.Free(yPtr);
                cuda.Free(alphaPtr);
                cuda.Free(gradPtr);
                cuda.Free(kiPtr);
                cuda.Free(kjPtr);
                cuda.Free(kernelDiagPtr);

                cuda.Free(valRedPtr);
                cuda.Free(idxRedPtr);

                cuda.UnloadModule(cuModule);

            }
        }
    }
}