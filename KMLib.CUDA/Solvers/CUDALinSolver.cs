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
using KMLib.SVMSolvers;
using System.Diagnostics;
using GASS.CUDA;
using GASS.CUDA.Types;
using System.IO;
using System.Runtime.InteropServices;
using System.Globalization;

namespace KMLib.GPU.Solvers
{
    /// <summary>
    /// Solver for linear SVM based on LIBLINEAR package
    /// http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    /// Paper: "A Dual Coordinate Descent Method for Large-scale Linear SVM" Hsieh et al., ICML 2008
    /// 
    /// modified version for computing on CUDA devices
    /// author: Krzysztof Sopyła (krzysztofsopyla@gmail.com)
    /// </summary>
    public class CUDALinSolver : LinearSolver
    {

        #region cuda names
        /// <summary>
        /// cuda module name
        /// </summary>
        protected const string cudaModuleName = "linSVMSolver.cubin";


        /// <summary>
        /// cuda texture name for main vector
        /// </summary>
        protected const string cudaMainVecTexRefName = "mainVectorTexRef";
        /// <summary>
        /// cuda texture name for labels
        /// </summary>
        protected const string cudaLabelsTexRefName = "labelsTexRef";

        /// <summary>
        /// cuda function name for computing product between W vector and all vector elements
        /// </summary>
        protected string cudaProductKernelName = "ComputeDotProd";

        /// <summary>
        /// cuda function name for computing step
        /// </summary>
        protected string cudaSolveL2SVM = "lin_l2r_l2_svc_solver_with_gradient";

        /// <summary>
        /// cuda function name for updating W vector
        /// </summary>
        protected string cudaUpdateW = "update_W";


        #endregion

        #region cuda types

        /// <summary>
        /// Cuda .net class for cuda opeation
        /// </summary>
        protected CUDA cuda;


        /// <summary>
        /// cuda loaded module
        /// </summary>
        protected CUmodule cuModule;

        /// <summary>
        /// cuda kernel function for computing dot product beetween W and all elements
        /// </summary>
        protected CUfunction cuFuncDotProd;

        /// <summary>
        /// cuda kernel function for computing steps
        /// </summary>
        protected CUfunction cuFuncSolver;


        /// <summary>
        /// cuda kernel function for updating vector W
        /// </summary>
        protected CUfunction cuFuncUpdateW;

        /// <summary>
        /// Cuda device pointer to vectors values in CSR matrix  format
        /// </summary>                                  
        protected CUdeviceptr valsCSRPtr;
        /// <summary>                                   
        /// cuda devie pointer to vectors indexes in CSR matrix format
        /// </summary>                                    
        protected CUdeviceptr idxCSRPtr;
        /// <summary>                                     
        /// cuda device pointer to vectors lenght in CSR matrix format
        /// </summary>
        protected CUdeviceptr vecLenghtCSRPtr;


        /// <summary>
        /// Cuda device pointer to vectors values in CSC matrix format
        /// </summary>                                   
        protected CUdeviceptr valsCSCPtr;
        /// <summary>                                    
        /// cuda devie pointer to vectors indexes in CSC matrix format
        /// </summary>                                  
        protected CUdeviceptr idxCSCPtr;
        /// <summary>                                   
        /// cuda device pointer to vectors lenght in CSC matrix format
        /// </summary>
        protected CUdeviceptr vecLenghtCSCPtr;


        /// <summary>
        /// cuda device pointer to diagonal cache QD
        /// </summary>
        private CUdeviceptr qdPtr;


        /// <summary>
        /// cuda device pointer for output
        /// </summary>
        protected CUdeviceptr gradPtr;

        /// <summary>
        /// cuda device pointer for output deltas
        /// </summary>
        protected CUdeviceptr deltasPtr;


        /// <summary>
        /// cuda reference to texture for deltas, 
        /// </summary>
        protected CUtexref cuDeltasTexRef;

        /// <summary>
        /// cuda device pointer for alpha array's
        /// </summary>
        private CUdeviceptr alphaPtr;


        /// <summary>
        /// cuda device pointer for diag, needed for coping to constant array on device
        /// </summary>
        private CUdeviceptr diagPtr;

        /// <summary>
        /// cuda reference to texture for main problem element(vector), 
        /// </summary>
        protected CUtexref cuMainVecTexRef;

        protected CUdeviceptr mainVecPtr;

        /// <summary>
        /// cuda refeerenc to texture for labels
        /// </summary>
        protected CUtexref cuLabelsTexRef;

        /// <summary>
        /// cuda pointer to labels, neded for coping to texture
        /// </summary>
        protected CUdeviceptr labelsPtr;


        #endregion


        /// <summary>
        /// vector "W" for computing kernel product with other vectors
        /// </summary>
        /// <remarks>all the time this vector will be modified and copied to cuda array</remarks>
        protected float[] mainVector;



        /// <summary>
        /// native pointer to memory region, used for computing  dot product and gradiens in solver
        /// </summary>
        protected IntPtr gradIntPtr;

        /// <summary>
        /// native pointer to memory region, used for computing steps in solver
        /// </summary>
        protected IntPtr deltasIntPtr;
        private float[] alpha;
        private float[] deltas;
        private float[] QD;



        protected int threadsPerBlock = CUDAConfig.XBlockSize;

        /// <summary>
        /// indicates how many blocks pre grid we create for cuda kernel launch
        /// </summary>
        protected int blocksPerGrid = -1;
        private float[] diag;
        private float stepScaling;


        /// <summary>
        /// Construct linear solver
        /// </summary>
        /// <param name="problem">trainning problem</param>
        /// <param name="C">penalty parameter</param>
        public CUDALinSolver(Problem<SparseVec> problem, float C)
            : base(problem, C)
        {
        }

        public CUDALinSolver(Problem<SparseVec> problem, float C, int[] weightedLabels, double[] weights)
            : base(problem, C, weightedLabels, weights)
        {
        }

        public override Model<SparseVec> ComputeModel()
        {

            int j;
            int l = problem.ElementsCount; //prob.l;
            int n = problem.FeaturesCount;// prob.n;
            int w_size = n; // prob.n;
            Model<SparseVec> model = new Model<SparseVec>();
            model.FeaturesCount = n;

            if (bias >= 0)
            {
                //Add to each feature vector last feature ==1;
                model.FeaturesCount = n - 1;
            }
            else
                model.FeaturesCount = n;

            model.Bias = bias;

            int[] perm = new int[l];
            // group training data of the same class
            //GroupClassesReturn rv = groupClasses(prob, perm);
            int nr_class = 0;
            int[] label;// = new int[l];// = rv.label;
            int[] start;// = rv.start;
            int[] count;// = rv.count;

            groupClasses(problem, out nr_class, out label, out start, out count, perm);

            model.NumberOfClasses = nr_class;


            model.Labels = new float[nr_class];
            for (int i = 0; i < nr_class; i++)
                model.Labels[i] = (float)label[i];

            // calculate weighted C
            double[] weighted_C = new double[nr_class];
            for (int i = 0; i < nr_class; i++)
            {
                weighted_C[i] = C;
            }


            SetClassWeights(nr_class, label, weighted_C);

            // constructing the subproblem
            //permutated vectors
            SparseVec[] permVec = new SparseVec[problem.ElementsCount];
            Debug.Assert(l == problem.ElementsCount);
            for (int i = 0; i < l; i++)
            {
                permVec[i] = problem.Elements[perm[i]];
            }


            Problem<SparseVec> sub_prob = new Problem<SparseVec>();
            sub_prob.ElementsCount = l;
            sub_prob.FeaturesCount = n;
            //we set labels below
            sub_prob.Y = new float[sub_prob.ElementsCount];
            sub_prob.Elements = permVec;

            //Initailize CUDA driver and load module
            InitCudaModule();

            if (nr_class == 2)
            {

                float[] w = new float[w_size];
                //for (int z = 0; z < w.Length; z++)
                //{
                //    w[z] = 1.0f;
                //}


                int e0 = start[0] + count[0];
                int k = 0;
                for (; k < e0; k++)
                    sub_prob.Y[k] = +1;
                for (; k < sub_prob.ElementsCount; k++)
                    sub_prob.Y[k] = -1;

                //copy all needed data to CUDA device
                SetCudaData(sub_prob);

                //Fill data on CUDA
                FillDataOnCuda(sub_prob, w, weighted_C[0], weighted_C[1]);

                solve_l2r_l2_svc_cuda(sub_prob, w, epsilon, weighted_C[0], weighted_C[1]);
                //solve_l2r_l1l2_svc(model.W, epsilon, weighted_C[0], weighted_C[1], solverType);

                model.W = new double[w_size];
                for (int s = 0; s < w.Length; s++)
                {
                    model.W[s] = w[s];
                }
            }
            else
            {
                model.W = new double[w_size * nr_class];
                float[] w = new float[w_size];

                SetCudaData(sub_prob);

                ///one against many
                for (int i = 0; i < nr_class; i++)
                {
                    int si = start[i];
                    int ei = si + count[i];

                    int k = 0;
                    for (; k < si; k++)
                        sub_prob.Y[k] = -1;
                    for (; k < ei; k++)
                        sub_prob.Y[k] = +1;
                    for (; k < sub_prob.ElementsCount; k++)
                        sub_prob.Y[k] = -1;

                    FillDataOnCuda(sub_prob, w, weighted_C[i], C);
                    //train_one(sub_prob, param, w, weighted_C[i], param.C);
                    solve_l2r_l2_svc_cuda(sub_prob, w, epsilon, weighted_C[i], C);

                    for (j = 0; j < n; j++)
                        model.W[j * nr_class + i] = w[j];
                }
            }

            DisposeCuda();

            return model;
        }


        /// <summary>
        /// fill data which can be changed like
        /// Y - labels
        /// diag -
        /// </summary>
        /// <param name="sub_prob"></param>
        /// <param name="w"></param>
        /// <param name="Cn"></param>
        /// <param name="Cp"></param>
        private void FillDataOnCuda(Problem<SparseVec> sub_prob, float[] w, double Cn, double Cp)
        {
            cuda.CopyHostToDevice(labelsPtr, sub_prob.Y);

            diag = new float[] { (float)(0.5 / Cn), 0, (float)(0.5 / Cp) };

            cuda.CopyHostToDevice(diagPtr, diag);

            cuda.CopyHostToDevice(mainVecPtr, w);
        }

        private void SetCudaData(Problem<SparseVec> sub_prob)
        {

            int vecDim = sub_prob.Elements[0].Dim;

            /* 
             * copy vectors to CUDA device
             */
            float[] vecVals;
            int[] vecIdx;
            int[] vecLenght;
            CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght, sub_prob.Elements);
            valsCSRPtr = cuda.CopyHostToDevice(vecVals);
            idxCSRPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtCSRPtr = cuda.CopyHostToDevice(vecLenght);


            CudaHelpers.TransformToCSCFormat(out vecVals, out vecIdx, out vecLenght, sub_prob.Elements);
            valsCSCPtr = cuda.CopyHostToDevice(vecVals);
            idxCSCPtr = cuda.CopyHostToDevice(vecIdx);
            vecLenghtCSCPtr = cuda.CopyHostToDevice(vecLenght);



            /* 
             * allocate memory for gradient
             */
            uint memSize = (uint)(sub_prob.ElementsCount * sizeof(float));
            //allocate mapped memory for our results (dot product beetween vector W and all elements)
            gradIntPtr = cuda.HostAllocate(memSize, CUDADriver.CU_MEMHOSTALLOC_DEVICEMAP);
            gradPtr = cuda.GetHostDevicePointer(gradIntPtr, 0);

            //allocate memory for main vector, size of this vector is the same as dimenson, so many 
            //indexes will be zero, but cuda computation is faster
            mainVector = new float[vecDim];

           
            //move W wector
            //CudaHelpers.FillDenseVector(problemElements[0], mainVector);
            CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuMainVecTexRef, cudaMainVecTexRefName, mainVector, ref mainVecPtr);


            //set texture memory for labels
            CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuLabelsTexRef, cudaLabelsTexRefName, sub_prob.Y, ref labelsPtr);


            /*
             * data for cuda solver
            */

            //normaly for L2 solver QDii= xii*xii+Diag_i
            //where Diag_i = 0.5/Cp if yi=1
            //      Diag_i = 0.5/Cn if yi=-1
            //but we will add this on GPU
            QD = new float[sub_prob.ElementsCount];
            alpha = new float[sub_prob.ElementsCount];
            deltas = new float[sub_prob.ElementsCount];
            float[] diag = new float[3];
            for (int i = 0; i < sub_prob.ElementsCount; i++)
            {
                QD[i] = sub_prob.Elements[i].DotProduct();
                alpha[i] = 0f;
                deltas[i] = 0;
            }

            qdPtr = cuda.CopyHostToDevice(QD);

            alphaPtr = cuda.Allocate(alpha);


            //deltasPtr = cuda.Allocate(deltas);
            CudaHelpers.SetTextureMemory(cuda,cuModule,ref cuDeltasTexRef, "deltasTexRef", deltas, ref deltasPtr);

            diagPtr = cuda.GetModuleGlobal(cuModule, "diag_shift");
            //set this in fill function
            //cuda.CopyHostToDevice(diagPtr, diag);

            //CUdeviceptr dimPtr = cuda.GetModuleGlobal(cuModule, "Dim");
            ////todo: check if it ok
            ////cuda.Memset(dimPtr,(uint) vecDim, 1);
            //int[] dimArr = new int[] { vecDim };
            //cuda.CopyHostToDevice(dimPtr,dimArr);

            //CUDARuntime.cudaMemcpyToSymbol("Dim", dimPtr, 1, 0, cudaMemcpyKind.cudaMemcpyHostToDevice);
            //CUDARuntime.cudaMemcpyToSymbol("Dim", ,1,0, cudaMemcpyKind.cudaMemcpyHostToDevice);

            CUdeviceptr deltaScalingPtr = cuda.GetModuleGlobal(cuModule, "stepScaling");

            //two ways of computing scaling param, should be the same, but it depends on rounding.
            //stepScaling = (float)(1.0 / Math.Sqrt(sub_prob.ElementsCount));

            stepScaling = 0.0002f;// (float)(1.0 / sub_prob.ElementsCount);

            //set scaling constant
            float[] scArr = new float[] { stepScaling };
            cuda.CopyHostToDevice(deltaScalingPtr, scArr);
            //cuda.Memset(deltaScalingPtr, (uint) scaling,sizeof(float));

            //cuda.CopyHostToDevice(dimPtr, problem.Elements[0].Dim);

            SetCudaParameters(sub_prob);
        }


        /// <summary>
        /// Sets parameters needeb by cuda kernels
        /// </summary>
        /// <param name="sub_prob"></param>
        private void SetCudaParameters(Problem<SparseVec> sub_prob)
        {
            /* 
             * Set Cuda functions parameters 
             */


            /*
             *  Set cuda function parmeters for computing Dot product
             */
            #region Set cuda function parmeters for computing Dot product

            cuda.SetFunctionBlockShape(cuFuncDotProd, threadsPerBlock, 1, 1);

            int offset = 0;
            cuda.SetParameter(cuFuncDotProd, offset, valsCSRPtr.Pointer);
            offset += IntPtr.Size;
            cuda.SetParameter(cuFuncDotProd, offset, idxCSRPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, vecLenghtCSRPtr.Pointer);
            offset += IntPtr.Size;


            cuda.SetParameter(cuFuncDotProd, offset, gradPtr.Pointer);
            offset += IntPtr.Size;

            cuda.SetParameter(cuFuncDotProd, offset, (uint)sub_prob.ElementsCount);
            offset += sizeof(int);

            cuda.SetParameterSize(cuFuncDotProd, (uint)offset);
            #endregion

            /*
             *  Set Cuda function parameters for computing deltas
             */
            #region Set Cuda function parameters for computing deltas

            //todo: is threads per block for solver corect?
            cuda.SetFunctionBlockShape(cuFuncSolver, threadsPerBlock, 1, 1);
            int offset2 = 0;
            cuda.SetParameter(cuFuncSolver, offset2, qdPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameter(cuFuncSolver, offset2, alphaPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameter(cuFuncSolver, offset2, gradPtr.Pointer);
            offset2 += IntPtr.Size;
            cuda.SetParameter(cuFuncSolver, offset2, deltasPtr.Pointer);
            offset2 += IntPtr.Size;

            cuda.SetParameter(cuFuncSolver, offset2, (uint)sub_prob.ElementsCount);
            offset2 += sizeof(int);

            cuda.SetParameterSize(cuFuncSolver, (uint)offset2);

            #endregion

            /*
             * Set cuda function parameters for updating W vector
             */
            #region Set cuda function parameters for updating W vector

            //todo: is threads per block for updates W corect?
            cuda.SetFunctionBlockShape(cuFuncUpdateW, threadsPerBlock, 1, 1);

            int offset3 = 0;
            cuda.SetParameter(cuFuncUpdateW, offset3, valsCSCPtr.Pointer);
            offset3 += IntPtr.Size;
            cuda.SetParameter(cuFuncUpdateW, offset3, idxCSCPtr.Pointer);
            offset3 += IntPtr.Size;

            cuda.SetParameter(cuFuncUpdateW, offset3, vecLenghtCSCPtr.Pointer);
            offset3 += IntPtr.Size;


            cuda.SetParameter(cuFuncUpdateW, offset3, mainVecPtr.Pointer);
            offset3 += IntPtr.Size;

            //cuda.SetParameter(cuFuncUpdateW, offset3, (uint)(sub_prob.ElementsCount+50) );//[0].Dim-40) );
            cuda.SetParameter(cuFuncUpdateW, offset3, (uint)sub_prob.Elements[0].Dim);
            offset3 += sizeof(int);

            cuda.SetParameterSize(cuFuncUpdateW, (uint)offset3);

            #endregion

        }






        /// <summary>
        /// Dispose all object used by CUDA
        /// </summary>
        private void DisposeCuda()
        {
            if (cuda != null)
            {
                //free all resources
                cuda.Free(valsCSRPtr);
                cuda.Free(valsCSCPtr);
                valsCSRPtr.Pointer =IntPtr.Zero;
                valsCSCPtr.Pointer =IntPtr.Zero;

                cuda.Free(idxCSRPtr);
                cuda.Free(idxCSCPtr);
                idxCSRPtr.Pointer =IntPtr.Zero;
                idxCSCPtr.Pointer =IntPtr.Zero;

                cuda.Free(vecLenghtCSRPtr);
                cuda.Free(vecLenghtCSCPtr);
                vecLenghtCSRPtr.Pointer =IntPtr.Zero;
                vecLenghtCSCPtr.Pointer =IntPtr.Zero;



                cuda.Free(qdPtr);
                qdPtr.Pointer =IntPtr.Zero;
                //  cuda.Free(diagPtr);
                diagPtr.Pointer =IntPtr.Zero;
                cuda.Free(alphaPtr);
                alphaPtr.Pointer =IntPtr.Zero;
                cuda.Free(gradPtr);
                gradPtr.Pointer =IntPtr.Zero;

                cuda.Free(deltasPtr);
                deltasPtr.Pointer =IntPtr.Zero;
                cuda.DestroyTexture(cuDeltasTexRef);

                cuda.Free(labelsPtr);
                labelsPtr.Pointer =IntPtr.Zero;
                cuda.DestroyTexture(cuLabelsTexRef);

                cuda.Free(mainVecPtr);
                mainVecPtr.Pointer =IntPtr.Zero;

                cuda.DestroyTexture(cuMainVecTexRef);

                cuda.UnloadModule(cuModule);
                cuda.Dispose();
                cuda = null;
            }

        }

        private void solve_l2r_l2_svc_cuda(Problem<SparseVec> sub_prob, float[] w, double epsilon, double Cp, double Cn)
        {
            
           
            //blocks per Grid for compuing dot prod
            int bpgDotProd = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;
            //blocks per Grid for solver kernel
            int bpgSolver = (sub_prob.Elements.Length + threadsPerBlock - 1) / threadsPerBlock;
            //blocks per Grid for update_W kernel
            int bpgUpdateW = (sub_prob.Elements[0].Dim + threadsPerBlock - 1) / threadsPerBlock;

            double obj = Double.PositiveInfinity;
            int maxIter = 2000;


            float[] deltasCu = new float[sub_prob.ElementsCount];
            float[]alphaCu = new float[sub_prob.ElementsCount];

            float[] alpha_i = new float[sub_prob.ElementsCount];
            float[] w1 = new float[sub_prob.Elements[0].Dim];
            float[] w2 = new float[sub_prob.Elements[0].Dim];

            for (int i = 0; i < w.Length; i++)
            {
                w1[i]=w2[i] = w[i];
            }

            //inverted hessian for toy_2d_3 problem
            //float[,] invHessian = new float[3, 3]{  
            //    {5.05f, -0.96f, -3.5f},
            //    {-0.96f, 0.9f, 1},
            //    {-3.59f , 1f, 2.8f}
            //};

            int iter = 0;

            while (iter<maxIter)
            {

                //computes dot product between W and all elements

                cuda.Launch(cuFuncDotProd, bpgDotProd, 1);

                cuda.SynchronizeContext();
                float[] grad = new float[sub_prob.ElementsCount];
                Marshal.Copy(gradIntPtr, grad, 0, grad.Length);

                #region test host code
                
                //float[] dots = new float[sub_prob.ElementsCount];
                //float[] dots1 = new float[sub_prob.ElementsCount];
                //for (int i = 0; i < dots.Length; i++)
                //{

                //    var element = sub_prob.Elements[i];
                //    for (int k = 0; k < element.Count; k++)
                //    {
                //        dots[i] += w[element.Indices[k] - 1] * element.Values[k];
                //        dots1[i] += w1[element.Indices[k] - 1] * element.Values[k];
                //    }
                //    dots[i] *= sub_prob.Y[i];
                //    dots1[i] *= sub_prob.Y[i];

                //}

                //float[] grad_i = new float[sub_prob.ElementsCount];
                //for (int i = 0; i < grad_i.Length; i++)
                //{

                //    float dot = 0;
                    
                //    var vec_i = sub_prob.Elements[i];
                //    sbyte y_i = (sbyte)sub_prob.Y[i];
                //    for (int j = 0; j < sub_prob.ElementsCount; j++)
                //    {
                //        var vec_j = sub_prob.Elements[j];
                //        sbyte y_j = (sbyte)sub_prob.Y[j];
                //        float part_dot = 0;
                //        for (int k = 0; k < vec_i.Dim; k++)
                //        {
                //            part_dot += vec_i.Values[k] * vec_j.Values[k];
                //        }
                //        if (i == j)
                //        {
                //            part_dot += diag[y_i + 1] ;
                //        }

                //        part_dot = part_dot * y_i * y_j;
                //        part_dot *= alpha_i[j];
                //        dot += part_dot;
                       
                //    }
                //    grad_i[i] = dot - 1;

                //}

                #endregion

                cuda.Launch(cuFuncSolver, bpgSolver, 1);

                cuda.SynchronizeContext();
                float[] grad2 = new float[sub_prob.ElementsCount];
                Marshal.Copy(gradIntPtr, grad2, 0, grad2.Length);

                //float[] grad3 = new float[sub_prob.ElementsCount];
               
                //float[] projGrad = new float[sub_prob.ElementsCount];
                //float[] projGrad_i = new float[sub_prob.ElementsCount];
                //for (int i = 0; i < grad3.Length; i++)
                //{
                //    sbyte y_i = (sbyte)sub_prob.Y[i];
                //    grad3[i] = dots1[i] - 1 + alpha[i] * diag[y_i + 1];

                //    if (alpha[i] == 0)
                //    {
                //        projGrad[i] = Math.Min(0, grad3[i]);
                //       // projGrad_i[i] = Math.Min(0, grad_i[i]);
                //    }
                //    else
                //    {
                //        projGrad[i] = grad3[i];
                //       // projGrad_i[i] = grad_i[i];
                //    }

                //}

                

                cuda.CopyDeviceToHost(deltasPtr, deltasCu);
                cuda.CopyDeviceToHost(alphaPtr, alphaCu);


                cuda.Launch(cuFuncUpdateW, bpgUpdateW, 1);
                
                cuda.SynchronizeContext();

                cuda.CopyDeviceToHost(mainVecPtr, w);
               
                //take grad and check stop condition
                //Marshal.Copy(gradIntPtr, , 0, results.Length);

                
                //compute w1
                
                //double su = 0;
                //float[] wAll = new float[sub_prob.Elements[0].Dim];
                //for (int p = 0; p < sub_prob.ElementsCount; p++)
                //{
                //    sbyte y_i = (sbyte)sub_prob.Y[p];
                //    float old_alpha = alpha[p];

                //    float alphaStep = 0;

                //    //for (int k = 0; k < alpha_i.Length; k++)
                //    //{
                //    //    alphaStep += invHessian[p,k] * projGrad_i[k];
                //    //}


                //    alpha[p] = Math.Max(alpha[p] -stepScaling* projGrad[p] / (QD[p] + diag[y_i + 1]), 0);

                //    alpha_i[p] = Math.Max(alpha_i[p] - stepScaling* projGrad_i[p] / (QD[p] + diag[y_i + 1]), 0);
                //   // alpha_i[p] = Math.Max(alpha_i[p] - 0.01f* projGrad_i[p] , 0);
                //    //alpha_i[p] = Math.Max(alpha_i[p] - alphaStep, 0);

                //    float d = deltasCu[p];
                //    float d2 =(alpha[p] - old_alpha) * y_i;
                //    var spVec = sub_prob.Elements[p];
                //    for (int k = 0; k < spVec.Count; k++)
                //    {
                //        w1[spVec.Indices[k] - 1] += d2 * spVec.Values[k];
                //        w2[spVec.Indices[k] - 1] += d2 * spVec.Values[k];
                //        wAll[spVec.Indices[k] - 1] += alpha[p]*y_i * spVec.Values[k];
                //    }
                //    su += y_i * alpha[p];
                // }

#if DEBUG
                obj = ComputeObj(w, alphaCu, sub_prob, diag);

               
               
              Debug.WriteLine(obj.ToString(CultureInfo.GetCultureInfo("pl-PL").NumberFormat));

#endif

              float minPG = float.PositiveInfinity;
              float maxPG = float.NegativeInfinity;
              for (int i = 0; i < grad2.Length; i++)
              {
                  minPG = Math.Min(minPG, grad2[i]);
                  maxPG = Math.Max(maxPG, grad2[i]);
              }
              if (maxPG < 0)
                  maxPG = float.NegativeInfinity;
              if (minPG > 0)
                  minPG = float.PositiveInfinity;

              if (Math.Abs( maxPG - minPG) <= epsilon)
                  break;

                iter++;
            }

            cuda.SynchronizeContext();
            //copy resulsts form device to host
            cuda.CopyDeviceToHost(mainVecPtr, w);
            cuda.CopyDeviceToHost(alphaPtr, alpha);



            ComputeObj(w, alpha, sub_prob, diag);
            //int l = sub_prob.ElementsCount;// prob.l;
            //int w_size = sub_prob.FeaturesCount;// prob.n;
            //double v = 0;
            //int nSV = 0;
            //for (int i = 0; i < w_size; i++)
            //    v += w[i] * w[i];
            //for (int i = 0; i < l; i++)
            //{
            //    sbyte y_i =(sbyte) sub_prob.Y[i];
            //    v += alpha[i] * (alpha[i] * diag[y_i+1] - 2);
            //    if (alpha[i] > 0) ++nSV;
            //}


            //Debug.WriteLine("Objective value = {0}", v / 2);
            //Debug.WriteLine("nSV = {0}", nSV);



        }

        //private double ComputeObj(float[] w, float[] alpha, Problem<SparseVec> sub_prob, float[] diag)
        //{
        //    double v = 0, v1=0;
        //    int nSV = 0;
        //    for (int i = 0; i < w.Length; i++)
        //    {
        //        v += w[i] * w[i];
        //        v1 += 0.5*w[i] * w[i];
        //    }
        //    for (int i = 0; i < alpha.Length; i++)
        //    {
        //        sbyte y_i = (sbyte)sub_prob.Y[i];

        //        //original line
        //        //v += alpha[i] * (alpha[i] * diag[GETI(y_i, i)] - 2);
        //        v += alpha[i] * (alpha[i] * diag[y_i + 1] - 2);
        //        v1 += 0.5* alpha[i] * (alpha[i] * diag[y_i + 1] - 2);
        //        if (alpha[i] > 0) ++nSV;
        //    }

        //    v = v / 2;
        //  //  Debug.WriteLine("Objective value = {0}", v);
        //  //  Debug.WriteLine("nSV = {0}", nSV);

        //    return v;
        //}



        protected void InitCudaModule()
        {
            cuda = new CUDA(0, true);
            cuModule = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, cudaModuleName));
            cuFuncDotProd = cuda.GetModuleFunction(cudaProductKernelName);
            cuFuncSolver = cuda.GetModuleFunction(cudaSolveL2SVM);
            cuFuncUpdateW = cuda.GetModuleFunction(cudaUpdateW);
        }

        //protected void SetTextureMemory(ref CUtexref texture, string texName, float[] data, ref CUdeviceptr memPtr)
        //{
        //    texture = cuda.GetModuleTexture(cuModule, texName);
        //    memPtr = cuda.CopyHostToDevice(data);
        //    cuda.SetTextureAddress(texture, memPtr, (uint)(sizeof(float) * data.Length));

        //}

    }
}
