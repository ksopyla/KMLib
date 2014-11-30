using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using dnAnalytics.LinearAlgebra;
using KMLib;
using KMLib.Helpers;
using KMLib.Kernels;
using KMLib.GPU;
using System.Diagnostics;
//using dnaLA = dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;
using KMLib.SVMSolvers;
using System.IO;
using KMLib.GPU.Solvers;
using KMLib.Transforms;
using KMLib.GPU.GPUKernels.Col2;
using KMLib.GPU.GPUEvaluators;


namespace KMLibUsageApp
{
    internal class Program
    {
        private static float C = 4f;
        static float gamma = 0.5f;
        private static int folds = 5;

        static IDataTransform<SparseVec> dataTransform = new NullTransform();

        private static void Main(string[] args)
        {
            if (args.Length < 1)
                throw new ArgumentException("to liitle arguments");
            string dataFolder = args[0];
            //dataFolder = @"./Data";

            Debug.Listeners.Add(new ConsoleTraceListener());

                        
            IList<Tuple<string, string, int>> dataSetsToTest = CreateDataSetList(dataFolder);

            string trainningFile;
            string testFile;
            int numberOfFeatures;
            ChooseDataSet(dataFolder, out trainningFile, out testFile, out numberOfFeatures);

                  
            SVMClassifyLowLevel(trainningFile, testFile, numberOfFeatures, C);

            Console.WriteLine("Press any button");
            Console.ReadKey();

        }


        private static IList<Tuple<string, string, int>> CreateDataSetList(string dataFolder)
        {
            List<Tuple<string, string, int>> dataSets = new List<Tuple<string, string, int>>(8);



            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/a1a.train",
            //    dataFolder + "/a1a.train",
            //    123));



            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/mnist.scale20k",
            //    dataFolder + "/mnist.scale1k.t",
            //    784));



            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/w8a",
                dataFolder + "/w8a.t",
                300));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/a9a",
                dataFolder + "/a9a.t",
                123));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/news20.binary",
                dataFolder + "/news20.binary",
                1335191));


            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/real-sim",
                dataFolder + "/real-sim",
                20958));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/rcv1_test.binary",
                dataFolder + "/rcv1_train.binary",
                47236));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/mnist.scale",
                dataFolder + "/mnist.scale.t",
                784));



            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/dominionstats.01scale_r2.train",
                dataFolder + "/dominionstats.01scale_r2.test",
                596));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/tweet.train",
                dataFolder + "/tweet.test",
                52242));


            dataSets.Add(new Tuple<string, string, int>(
               dataFolder + "/webspam_wc_normalized_unigram.svm",
               dataFolder + "/webspam_wc_normalized_unigram.svm",
               254));


            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/kytea-msr_first_500k.train",
                dataFolder + "/kytea-msr.test",
                8683737));


            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/kytea-msr_first_1M.train",
                dataFolder + "/kytea-msr.test",
                8683737));

            //dataSets.Add(new Tuple<string, string, int>(
            //   dataFolder + "/url_combined_1.5M.train",
            //   dataFolder + "/url_combined_800k.test",
            //   3231961));

            //dataSets.Add(new Tuple<string, string, int>(
            //   dataFolder + "/kdda_2M",
            //   dataFolder + "/kdda.t",
            //   20216830));

            return dataSets;
        }


        /// <summary>
        /// Contains hard coded paths to datasets
        /// </summary>
        /// <param name="dataFolder"></param>
        /// <param name="trainningFile"></param>
        /// <param name="testFile"></param>
        /// <param name="numberOfFeatures"></param>
        private static void ChooseDataSet(string dataFolder, out string trainningFile, out string testFile, out int numberOfFeatures)
        {
            #region grant

            //trainningFile = dataFolder + "/zegarki_meskie_damskie_sift_kmeans_5_50_headers.svm.libsvm";
            //testFile = dataFolder + "/zegarki_meskie_damskie_sift_kmeans_5_50_headers.svm.libsvm";
            //numberOfFeatures = 5;

            //trainningFile = dataFolder + "/SVMzegarki_md_kmenas_w1000_i920.txt";
            //testFile = dataFolder + "/SVMzegarki_md_kmenas_w1000_i920.txt";
            //numberOfFeatures = 1000;
            #endregion

            #region toy samples
            //testFile = trainningFile = dataFolder + "/a1a.small.train";
            ////in a1a problem max index is 123
            //numberOfFeatures = 123;

            //trainningFile = dataFolder + "/toy_2d.train";
            // trainningFile = dataFolder + "/toy_2d_16.train";
            // trainningFile = dataFolder + "/toy_2d_3.train";
            // testFile = dataFolder + "/toy_2d.test";
            // numberOfFeatures = 2;

            //testFile = trainningFile = dataFolder + "/toy_8ins_6d.txt";
            //numberOfFeatures = 6;

            //trainningFile = dataFolder + "/toy_10d_10.train";
            //testFile = dataFolder + "/toy_10d_10.train";
            //numberOfFeatures = 10;
            #endregion


            //trainningFile = dataFolder + "/a1a.train";
            ////testFile = dataFolder + "/a1a.test";
            //////testFile = dataFolder + "/a1a.train";
            //testFile = dataFolder + "/a1a.train";
            ////in a1a problem max index is 123
            //numberOfFeatures = 123;


            //trainningFile = dataFolder + "/a9a";
            //testFile = dataFolder + "/a9a.t";
            ////testfile = datafolder + "/a9a";
            //numberOfFeatures = 123;

            //trainningFile = dataFolder + "/a9a_128.train";
            //testFile = dataFolder + "/a9a.t";
            ////testFile = dataFolder + "/a9a";
            //numberOfFeatures = 123;

            trainningFile = dataFolder + "/w8a";
            testFile = dataFolder + "/w8a.t";
            numberOfFeatures = 300;

            ////trainningFile = dataFolder + "/rcv1_train.binary";
            ////testFile = dataFolder + "/rcv1_test.binary";
            //trainningFile = dataFolder + "/rcv1_test.binary";
            //testFile = dataFolder + "/rcv1_train.binary";
            //numberOfFeatures = 47236;

            //trainningFile = dataFolder + "/news20.binary";
            //testFile = dataFolder + "/news20.binary";
            //numberOfFeatures = 1335191;

            ////trainningFile = dataFolder + "/mnist.scale";
            //trainningFile = dataFolder + "/mnist.scale20k";
            //testFile = dataFolder + "/mnist.scale1k.t";
            //////testFile = dataFolder + "/mnist.scale.t";
            //numberOfFeatures = 784;

            //trainningFile = dataFolder + "/real-sim_small_3K";
            //string trainningFile = dataFolder + "/real-sim_med_6K";
            //string trainningFile = dataFolder + "/real-sim_med_10K";
            //trainningFile = dataFolder + "/real-sim";
            //testFile = dataFolder + "/real-sim";
            //numberOfFeatures = 20958;
           
            
            //dominostats #train	193657  #test	82996 dim	596
            //trainningFile = dataFolder + "/dominionstats.train"; //#2626 inst
            //trainningFile = dataFolder + "/dominionstats.01scale_r2.train"; //#2626 inst
            ////testFile = dataFolder + "/dominionstats.01scale_r2.test"; //#2626 inst
            //testFile = dataFolder + "/dominionstats.test"; //#1125
            //numberOfFeatures = 596;
            

            //http://mlcomp.org/datasets/469
            ////#train	177862 test	76227 dim	52242, min error=0.046 liblinear
            //trainningFile = dataFolder + "/tweet.train"; //#2626 inst
            //testFile = dataFolder + "/tweet.test"; //#1125
            //numberOfFeatures = 52242;


            //http://mlcomp.org/datasets/513
            ////#train	3963546 test	180370, train dim=8446390, test dim=8683737 
            //trainningFile = dataFolder + "/kytea-msr.train";
            //trainningFile = dataFolder + "/kytea-msr_first_500k.train";
            //trainningFile = dataFolder + "/kytea-msr_first_1M.train";
            //testFile = dataFolder + "/kytea-msr.test";
            //numberOfFeatures = 8683737;


            //trainningFile = dataFolder + "/webspam_wc_normalized_unigram.svm";
            //testFile = dataFolder + "/webspam_wc_normalized_unigram.svm";
            //numberOfFeatures = 254;


            //trainningFile = dataFolder + "/url_combined";
            //testFile = dataFolder + "/url_combined";
            //numberOfFeatures = 3231961;

            //trainningFile = dataFolder + "/url_combined_1.5M.train";
            //testFile = dataFolder + "/url_combined_800k.test";
            //numberOfFeatures = 3231961;


            //trainningFile = dataFolder + "/kdda";
            //testFile = dataFolder + "/kdda.t";
            //numberOfFeatures = 20216830;

            //trainningFile = dataFolder + "/kdda_4M";
            //testFile = dataFolder + "/kdda.t";
            //numberOfFeatures = 20216830;

            //trainningFile = dataFolder + "/kdda_2M";
            //testFile = dataFolder + "/kdda.t";
            //numberOfFeatures = 20216830;

        }


        /// <summary>
        /// Train and test SVM, using low level api, construct kernels, solver and evaluator by hand
        /// </summary>
        private static void SVMClassifyLowLevel(string trainningFile,
            string testFile,
            int numberOfFeatures,
            float paramC)
        {

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();

            Console.WriteLine("read vectors");
            //Problem<SparseVec> test1 = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);
            //Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);
            
            Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);
            train.Elements = dataTransform.Transform(train.Elements);

            Console.WriteLine("end read vectors");

            Model<SparseVec> model;
            
            //Evaluator<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            Evaluator<SparseVec> evaluator = new DualEvaluator<SparseVec>();

            //CSR
            //Evaluator<SparseVec> evaluator = new CuRBFCSREvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new CuNChi2CSREvaluator();
            //Evaluator<SparseVec> evaluator = new CuExpChiCSREvaluator(gamma);
            
            //ELL-ILP
            //Evaluator<SparseVec> evaluator = new CuRBFEllILPEvaluator(gamma);
            
            //ELLPACK
            //Evaluator<SparseVec> evaluator = new CuRBFEllpackEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new CuNChi2EllpackEvaluator();
            //Evaluator<SparseVec> evaluator = new CuExpChiEllpackEvaluator(gamma);
            
            //ERTILP
            //Evaluator<SparseVec> evaluator = new CuRBFERTILPEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new CuNChi2ERTILPEvaluator();
            //Evaluator<SparseVec> evaluator = new CuExpChiERTILPEvaluator(gamma);
            
            //SLIECED ELLPACK
            //Evaluator<SparseVec> evaluator = new CuRBFSlEllEvaluator(gamma);
            
            //SERTILP
            //Evaluator<SparseVec> evaluator = new CuRBFSERTILPEvaluator(gamma);


            #region Cuda kernels

            //IKernel<SparseVec> kernel = new CuLinearKernel();
            
            //********** RBF kernels **************//
            //IKernel<SparseVec> kernel = new CuRBFCSRKernel(gamma );
            //IKernel<SparseVec> kernel = new CuRBFEllpackKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFEllILPKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFERTILPKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFSlEllKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFSERTILPKernel(gamma);

            //IKernel<SparseVec> kernel = new CuRBFEllILPKernelCol2(gamma);

            //********* nChi2 Kernels *******************//
            //IKernel<SparseVec> kernel = new CuNChi2CSRKernel();
            //IKernel<SparseVec> kernel = new CuNChi2ERTILPKernel();
            //IKernel<SparseVec> kernel = new CuNChi2SERTILPKernel();

            //IKernel<SparseVec> kernel = new CuNChi2EllKernel();
            //IKernel<SparseVec> kernel = new CuNChi2SlEllKernel();

            /************ chi kernels ***********/
            //IKernel<SparseVec> kernel = new CuChi2EllKernel();


            //********** ExpChi2 Kernels ********************//
            
            IKernel<SparseVec> kernel = new CuExpChiCSRKernel(gamma);
            
            //IKernel<SparseVec> kernel = new CuExpChiERTILPKernel(gamma);
            //IKernel<SparseVec> kernel = new CuExpChiSERTILPKernel(gamma);

            //IKernel<SparseVec> kernel = new CuExpChiEllKernel(gamma);
            //IKernel<SparseVec> kernel = new CuExpChiSlEllKernel(gamma);


            #endregion
            //IKernel<SparseVec> kernel = new RbfKernel(gamma);
            //IKernel<SparseVec> kernel = new LinearKernel();
            //IKernel<SparseVec> kernel = new ExpChiSquareKernel(gamma);


            Console.WriteLine("kernel init");
            kernel.ProblemElements = train.Elements;
            kernel.Y = train.Y;
            kernel.Init();

            //var Solver = new ParallelSmoFanSolver<SparseVec>(train, kernel, C);
            //this solver works a bit faster and use less memory
            var Solver = new ParallelSmoFanSolver2<SparseVec>(train, kernel, C);
            //var Solver = new SmoFanSolver<SparseVec>(train, kernel, C);
            //var Solver = new SmoRandomSolver<SparseVec>(train, kernel, C);
            
            //var Solver = new SmoFirstOrderSolver<SparseVec>(train, kernel, C);
            //var Solver = new GPUSmoFanSolver(train, kernel, C);

            //var Solver = new SmoFirstOrderSolver2Cols<SparseVec>(train, kernel, C);
            //var Solver = new GPUSmoFOSolver(train, kernel, C);
            
            Console.WriteLine("User solver {0} and kernel {1}", Solver.ToString(), kernel.ToString());

            Stopwatch timer = Stopwatch.StartNew();//
            model = Solver.ComputeModel();
            Console.Write(model.ToString());


            SaveModel(trainningFile, model, kernel, Solver);

            var disSolver = Solver as IDisposable;
            if (disSolver != null)
                disSolver.Dispose();
            Solver = null;
            
            var disKernel = kernel as IDisposable;
            if (disKernel != null)
                disKernel.Dispose();


            

            train.Dispose();

            Console.WriteLine("Start Testing");


            Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);
            test.Elements = dataTransform.Transform(test.Elements);
            evaluator.Kernel = kernel;
            evaluator.TrainedModel = model;

            evaluator.Init();


            Stopwatch t = Stopwatch.StartNew();
            float[] predictions = evaluator.Predict(test.Elements); //new float[1]; //

            t.Stop();
           
            Console.WriteLine("prediction takes {0}  ms={1}", t.Elapsed, t.ElapsedMilliseconds);



            //todo: Free evaluator memories
            var disposeEvaluator = evaluator as IDisposable;
            if (disposeEvaluator != null)
                disposeEvaluator.Dispose();


            int correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                float predictedLabel = predictions[i];

                if (predictedLabel == test.Y[i])
                    ++correct;
            }
            test.Dispose();
            double accuracy = (float)correct / predictions.Length;
            Console.WriteLine("accuracy ={0}", accuracy);

        }

        private static void SaveModel(string trainningFile, Model<SparseVec> model, IKernel<SparseVec> kernel, Solver<SparseVec> Solver)
        {
            string dsName = Path.GetFileName(trainningFile);
            string kernelName = kernel.ToString();
            string solverName = Solver.ToString();
            string modelFile = string.Format("{0}_{1}_{2}.model", dsName, kernelName, solverName);
            model.C = C;
            model.KernelParams = new float[]{gamma};
            model.WriteToFile(modelFile);
        }



        private static void SVMClassify<TProbElement>(
           Problem<TProbElement> train,
           Problem<TProbElement> test,
           IKernel<TProbElement> kernel,
           Evaluator<TProbElement> evaluator,
           float paramC)
        {



            double acc = -10;

            Validation<TProbElement> validation = new Validation<TProbElement>();
            validation.TrainingProblem = train;
            validation.Kernel = kernel;
            validation.C = paramC;
            validation.Evaluator = evaluator;

            Stopwatch timer = new Stopwatch();

            timer.Start();

            acc = validation.TrainAndTestValidation(train, test);
            //.TestValidation(train, test, kernel, penaltyC[i]);
            timer.Stop();

            Console.WriteLine("Validation on test data best acuuracy = {0} C={1} time={2} ms={3}", acc, paramC, timer.Elapsed, timer.ElapsedMilliseconds);

        }

    }

}