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
using KMLib.GPU.GPUKernels.Col2;

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
    ///  
    /// </remarks>
    public class GPUSmoFOSolver : Solver<SparseVec>, IDisposable
    {

        private CuVectorKernelCol2 gpuKernel;


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
        private string cudaModuleName = "gpuFOSmoSolver.cubin";
        
        private GASS.CUDA.Types.CUfunction cuFuncFindMaxIMinJ;
        private string funcFindMaxIMinJ = "FindMaxI_MinJ";
        
        
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
        private CUdeviceptr constBPtr;
        private CUdeviceptr constAPtr;
        private uint alignSize;
        private float[] B;
        private float[] A;


        public GPUSmoFOSolver(Problem<SparseVec> problem, IKernel<SparseVec> kernel, float C)
            : base(problem, kernel, C)
        {
            this.C = C;
            problemSize = problem.ElementsCount;

            y = problem.Y;
            QD = kernel.DiagonalDotCache;
            alpha = new float[problemSize];
            G = new float[problemSize];
            gpuKernel = (CuVectorKernelCol2)kernel;
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

                //Find i: Maximizes -y_i * grad(f)_i, i in I_up(\alpha) and j: minimizes -y_j * grad(f)_j, j in I_down(\alpha) 
                //maximal voilating pair approche
                Tuple<int, float> maxIPair;
                Tuple<int, float> minJPair;
                FindMaxIMinJPair(out maxIPair,out minJPair);
                                
                int i = maxIPair.Item1;
                GMaxI = maxIPair.Item2;

                int j = minJPair.Item1;
                GMaxJ = minJPair.Item2;
                
                 if (GMaxI + GMaxJ < EPS)
                        break;
                

                //Compute i-th kernel collumn, set the specific memory region on GPU, 
                ComputeKernel(i, kiPtr,j,kjPtr);


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

      


        private void FindMaxIMinJPair(out Tuple<int, float> maxIPair, out Tuple<int, float> minJPair)
        {
            cuda.Launch(cuFuncFindMaxIMinJ, reductionBlocks, 1);


            //in this tables first reductionBlocks elemetnts are connected with index 'i'
            //and the other reductionBlocks elements are connected with index 'j'
            cuda.CopyDeviceToHost(valRedPtr, reduceVal);
            cuda.CopyDeviceToHost(idxRedPtr, reduceIdx);
            cuda.SynchronizeContext();

            float maxI = float.NegativeInfinity;
            int idxI = -1;

            //max of minus grad
            float minJ = float.NegativeInfinity;
            int idxJ = -1;
            
            for (int i = 0; i < reductionBlocks; i++)
            {
                if (maxI < reduceVal[i])
                {
                    maxI = reduceVal[i];
                    idxI = reduceIdx[i];
                }

                if (minJ < reduceVal[i + reductionThreads])
                {
                    minJ = reduceVal[i + reductionThreads];
                    idxJ = reduceIdx[i + reductionThreads];
                }
            }

            
            //reduce elements which are above aligned index
            for (uint t = alignSize; t < problemSize; t++)
            {
                int yt =(int) y[t];
                float tempMax = yt * alpha[t] < B[yt + 1] ? -G[t] * yt : float.NegativeInfinity;

                if (maxI < tempMax)
                {
                    maxI = tempMax;
                    idxI = (int)t;
                }


                float tempMin = yt * alpha[t] > A[yt + 1] ? G[t] * yt : float.NegativeInfinity;

                if (minJ<tempMin)
                {
                    minJ = tempMin;
                    idxJ = (int)t;
                }



            }

            maxIPair= new Tuple<int, float>(idxI, maxI);
            minJPair = new Tuple<int, float>(idxJ, minJ);

        }

       

        /// <summary>
        /// compute kernel and sets memory pointed by ptr (onec for i-th, once for j-th kernel column)
        /// </summary>
        /// <param name="i"></param>
        /// <param name="QiPtr"></param>
        private void ComputeKernel(int i, CUdeviceptr QiPtr,int j, CUdeviceptr QjPtr)
        {

            //this cuda kernel computes 2 kernel colums as one long block of memory 
            //first bytes stores the i-kernel kolumn elements, last stores j-kernel column elements
            //so as a pointer to result is passed QiPtr and QjPtr is computed accordingly
            //QjPtr = QiPtr + sizeof(float) * problemElements.Length;
            gpuKernel.AllProductsGPU(i,j,QiPtr);

            QjPtr = QiPtr + sizeof(float) * problemSize;

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
            cuFuncFindMaxIMinJ = cuda.GetModuleFunction(funcFindMaxIMinJ);
           
            cuFuncUpdateG = cuda.GetModuleFunction(funcUpdateGFunc);


        }


        private void SetCudaData()
        {

            

            CudaHelpers.GetNumThreadsAndBlocks(problemSize, maxReductionBlocks, threadsPerBlock, ref reductionThreads, ref reductionBlocks);

            alphaPtr = cuda.CopyHostToDevice(alpha);
            gradPtr = cuda.CopyHostToDevice(G);
            yPtr = cuda.CopyHostToDevice(y);
           

            //kernel columns i,j is simpler to copy array of zeros 

            uint memSize = (uint)(sizeof(float) * problemSize * 2);

            kiPtr = cuda.Allocate(memSize);
            kjPtr = kiPtr + sizeof(float) * problemSize;

            //todo:remove it
            int redSize = reductionThreads; //reductionBlocks
            reduceVal = new float[redSize*2];
            reduceIdx = new int[redSize*2];

            
            valRedPtr = cuda.CopyHostToDevice(reduceVal);
            idxRedPtr = cuda.CopyHostToDevice(reduceIdx);


            constCPtr = cuda.GetModuleGlobal(cuModule, "C");
            float[] cData = new float[] { C };
            cuda.CopyHostToDevice(constCPtr, cData);

            constBPtr = cuda.GetModuleGlobal(cuModule, "B");
            B = new float[] {0,0, C };
            cuda.CopyHostToDevice(constBPtr, B);

            constAPtr = cuda.GetModuleGlobal(cuModule, "A");
            A = new float[] { -C, 0, 0 };
            cuda.CopyHostToDevice(constAPtr, A);


            SetCudaParams();

        }

        private void SetCudaParams()
        {

            #region Set cuda function parmeters for finding Max Idx

            cuda.SetFunctionBlockShape(cuFuncFindMaxIMinJ, reductionThreads, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncFindMaxIMinJ, offset, yPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxIMinJ, offset, alphaPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxIMinJ, offset, gradPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxIMinJ, offset, idxRedPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncFindMaxIMinJ, offset, valRedPtr.Pointer);
            offset += IntPtr.Size;

            alignSize = (uint) ((problemSize / (reductionThreads * 2)) * reductionThreads * 2);
            cuda.SetParameter(cuFuncFindMaxIMinJ, offset, alignSize);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncFindMaxIMinJ, (uint)offset);
            #endregion



            #region set cuda function parma for gradient updating

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
                
                //kjPtr is only offset of kiPtr, so if we free kiPtr its not necessary to free kjPtr, because it have been already free
                //cuda.Free(kjPtr);
              


                cuda.Free(valRedPtr);
                cuda.Free(idxRedPtr);

                cuda.UnloadModule(cuModule);

            }
        }
    }
}