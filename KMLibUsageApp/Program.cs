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


namespace KMLibUsageApp
{
    internal class Program
    {
        private static float C = 4f;
        static float gamma = 0.5f;
        private static void Main(string[] args)
        {
            if (args.Length < 1)
                throw new ArgumentException("to liitle arguments");

            Debug.Listeners.Add(new ConsoleTraceListener());

            string dataFolder = args[0];// @"D:\UWM\praca naukowa\doktorat\code\KMLib\KMLibUsageApp\Data";

            IList<Tuple<string, string, int>> dataSetsToTest = CreateDataSetList(dataFolder);

            //Console.WriteLine("press any key to start");
            //Console.ReadKey();
            //GroupedTestingDataSets(dataSetsToTest);
            //GroupedTestingLowLevelDataSets(dataSetsToTest);
           // TestOneDataSet(dataFolder);

            //TestOneDataSetWithCuda(dataFolder);

            //TestOneDataSetWithCuda(dataFolder);

            //TestMultiClasDataSet(dataFolder);

            string trainningFile;
            string testFile;
            int numberOfFeatures;
            ChooseDataSet(dataFolder, out trainningFile, out testFile, out numberOfFeatures);
           SVMClassifyLowLevel(trainningFile,testFile,numberOfFeatures, C);

            //SVMLinearClassifyLowLevel(trainningFile, testFile, numberOfFeatures, C);

            Console.WriteLine("Press any button");
            Console.ReadKey();

        }

        private static void TestOneDataSet(string dataFolder)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;
            ChooseDataSet(dataFolder, out trainningFile, out testFile, out numberOfFeatures);

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();
            Problem<SparseVec> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);

            Problem<SparseVec> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);

            //EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();
            
            EvaluatorBase<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            //EvaluatorBase<SparseVec> evaluator = new SequentialDualEvaluator<SparseVec>();

           // evaluator.Init();
            //IKernel<Vector> kernel = new PolinominalKernel(3, 0.5, 0.5);
            IKernel<SparseVec> kernel = new RbfKernel(gamma);
            //IKernel<SparseVec> kernel = new LinearKernel();
            SVMClassify(train, test, kernel, evaluator,C);

        }

        private static void TestMultiClasDataSet(string dataFolder)
        {
            //string trainningFile = dataFolder + "/glass.scale";
            //string testFile = dataFolder + "/glass.scale"; ;
            //int numberOfFeatures=9;

            string trainningFile = dataFolder + "/genresTrain_scale.train";
            string testFile = dataFolder + "/genresTest.arff_scale.t"; ;
            int numberOfFeatures=181;
            
                

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();
            Problem<SparseVec> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);

            Problem<SparseVec> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);

            //EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();

            EvaluatorBase<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            //EvaluatorBase<SparseVec> evaluator = new SequentialDualEvaluator<SparseVec>();

            // evaluator.Init();
            //IKernel<Vector> kernel = new PolinominalKernel(3, 0.5, 0.5);
            IKernel<SparseVec> kernel = new RbfKernel(gamma);
            //IKernel<SparseVec> kernel = new LinearKernel();

            
            //Solver = new ParallelSmoFanSolver<TProblemElement>(problem, kernel, C);
            //this solver works a bit faster and use less memory
            var mcSvm = new MultiClassSVM<SparseVec>(train, kernel, C, evaluator);

          
            Stopwatch timer = Stopwatch.StartNew();

            mcSvm.Train();

            Console.WriteLine("Models computed {0}  miliseconds={1}", timer.Elapsed, timer.ElapsedMilliseconds);


            Console.WriteLine("Start Testing");
            Stopwatch t = Stopwatch.StartNew();
           float[] predictions= mcSvm.Predict(test.Elements);

           
t.Stop();
            //toremove: only for tests
            Console.WriteLine("prediction takes {0}  ms={1}", t.Elapsed, t.ElapsedMilliseconds);


            SavePredictionToFile(predictions,Path.GetFileNameWithoutExtension(testFile)+".prediction");
         
            int correct = 0;
            for (int i = 0; i < test.ElementsCount; i++)
            {
                float predictedLabel = predictions[i];

                if (predictedLabel == test.Y[i])
                    ++correct;
            }
            test.Dispose();
            double accuracy = (float)correct / predictions.Length;
            Console.WriteLine("accuracy ={0}", accuracy);

        }

        private static void SavePredictionToFile(float[] predictions, string fileName)
        {
            using (Stream str = File.Open(fileName, FileMode.Create))
            using (StreamWriter sw = new StreamWriter(str, Encoding.ASCII, 1 << 12))
            {
                for (int i = 0; i < predictions.Length; i++)
                {

                    sw.WriteLine((int)predictions[i]);
                }
            }
        }


       
        /// <summary>
        /// Read train and test problem from dataset file and then train SVM using CUDA kernels
        /// </summary>
        /// <param name="dataFolder"></param>
        private static void TestOneDataSetWithCuda(string dataFolder)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;
            ChooseDataSet(dataFolder, out trainningFile, out testFile, out numberOfFeatures);

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();
            Problem<SparseVec> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);

            var trainSumArr = train.Elements.Sum(x => x.Indices.Length);
            var trainSum = train.Elements.Sum(x => x.Count);

            Problem<SparseVec> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);
            var testSumArr = test.Elements.Sum(x => x.Indices.Length);
            var testSum = test.Elements.Sum(x => x.Count);

            EvaluatorBase<SparseVec> evaluator = new SequentialDualEvaluator<SparseVec>();

            //EvaluatorBase<SparseVec> evaluator = new CudaLinearEvaluator();
           // EvaluatorBase<SparseVec> evaluator = new CudaRBFEvaluator(gamma);
         
           IKernel<SparseVec> kernel2 = new CudaLinearKernel();
           // IKernel<SparseVec> kernel2 = new CudaRBFKernel(gamma);

            SVMClassify(train, test, kernel2, evaluator, C);
        }

        private static void GroupedTestingDataSets(IList<Tuple<string, string, int>> dataSetsToTest)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;



            //EvaluatorBase<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            //IKernel<SparseVec> kernel = new RbfKernel(gamma);

            //EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();
            //IKernel<SparseVector> kernel = new LinearKernel();

            EvaluatorBase<SparseVec> evaluator = new CudaLinearEvaluator();
            IKernel<SparseVec> kernel = new CudaLinearKernel();

            foreach (var data in dataSetsToTest)
            {
                trainningFile = data.Item1;
                testFile = data.Item2;
                numberOfFeatures = data.Item3;

                Console.WriteLine("\n----------------------------------------------\n");
                Console.WriteLine("DataSets , trainning={1} testing={2} , atr={0}", numberOfFeatures, trainningFile, testFile);
                Console.WriteLine();
                

                Problem<SparseVec> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);

                Problem<SparseVec> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);
                
                SVMClassify(train, test, kernel, evaluator, C);
                Console.WriteLine("***************************\n");

            }
        }

        private static void GroupedTestingLowLevelDataSets(IList<Tuple<string, string, int>> dataSetsToTest)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;
            
            foreach (var data in dataSetsToTest)
            {
                trainningFile = data.Item1;
                testFile = data.Item2;
                numberOfFeatures = data.Item3;

                Console.WriteLine("\n----------------------------------------------\n");
                Console.WriteLine("DataSets , trainning={1} testing={2} , atr={0}", numberOfFeatures, trainningFile, testFile);
                Console.WriteLine();

                SVMClassifyLowLevel(trainningFile, testFile, numberOfFeatures, C);
                
                Console.WriteLine("***************************\n");

            }
        }

        private static IList<Tuple<string, string, int>> CreateDataSetList(string dataFolder)
        {
            List<Tuple<string, string, int>> dataSets = new List<Tuple<string, string, int>>(8);


            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/a1a.train",
                dataFolder + "/a1a.test",
                123));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/a9a",
                dataFolder + "/a9a.t",
                123));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/w8a",
                dataFolder + "/w8a.t",
                300));



            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/news20.binary",
                dataFolder + "/news20.binary",
                1335191));

            //string trainningFile = dataFolder + "/real-sim_small_3K";
            //string trainningFile = dataFolder + "/real-sim_med_6K";
            //string trainningFile = dataFolder + "/real-sim_med_10K";
            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/real-sim",
                dataFolder + "/real-sim",
                20958));
            
            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/mnist.scale",
                dataFolder + "/mnist.scale.t",
                784));
            



            //string testFile = dataFolder + "/rcv1_train_test.binary";
            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/rcv1_train.binary",
            //    dataFolder + "/rcv1_test.binary",
            //    47236));

            return dataSets;
        }


        /// <summary>
        /// Continas hard coded paths to datasets
        /// </summary>
        /// <param name="dataFolder"></param>
        /// <param name="trainningFile"></param>
        /// <param name="testFile"></param>
        /// <param name="numberOfFeatures"></param>
        private static void ChooseDataSet(string dataFolder, out string trainningFile, out string testFile, out int numberOfFeatures)
        {


            trainningFile = dataFolder + "/a1a.train";
            testFile = dataFolder + "/a1a.test";
            //testFile = dataFolder + "/a1a.train";
            //in a1a problem max index is 123
            numberOfFeatures = 123;

            //trainningFile = dataFolder + "/a9a";
            //testFile = dataFolder + "/a9a.t";
            //numberOfFeatures = 123;

            //trainningFile = dataFolder + "/w8a";
            //testFile = dataFolder + "/w8a.t";
            //numberOfFeatures = 300;

            //string trainningFile = dataFolder + "/colon-cancer.train";
            //string testFile = dataFolder + "/colon-cancer.train";
            //int numberOfFeatures = 2000;

            //string trainningFile = dataFolder + "/leu";
            //string testFile = dataFolder + "/leu.t";
            //int numberOfFeatures = 7129;

            //string trainningFile = dataFolder + "/duke";
            //string testFile = dataFolder + "/duke.tr";
            //int numberOfFeatures = 7129;

            //trainningFile = dataFolder + "/rcv1_train.binary";
            //testFile = dataFolder + "/rcv1_test.binary";
            //trainningFile = dataFolder + "/rcv1_test.binary";
            //testFile = dataFolder + "/rcv1_train.binary";
            ////string testFile = dataFolder + "/rcv1_train_test.binary";
            //numberOfFeatures = 47236;

            //trainningFile = dataFolder + "/news20.binary";
            //testFile = dataFolder + "/news20.binary";
            //numberOfFeatures = 1335191;

            //trainningFile = dataFolder + "/mnist.scale";
            //testFile = dataFolder + "/mnist.scale.t";
            //numberOfFeatures = 784;


            //string trainningFile = dataFolder + "/real-sim_small_3K";
            //string trainningFile = dataFolder + "/real-sim_med_6K";
            //string trainningFile = dataFolder + "/real-sim_med_10K";
            //trainningFile = dataFolder + "/real-sim";
            //testFile = dataFolder + "/real-sim.t";
            //numberOfFeatures = 20958;

            //for test
            //trainningFile = dataFolder + "/liver-disorders_scale_small.txt";
            //testFile = dataFolder + "/liver-disorders_scale_small.txt";
            //string trainningFile = dataFolder + "/liver-disorders_scale.txt";
            //string testFile = dataFolder + "/liver-disorders_scale.txt";
            //numberOfFeatures = 6;
            //  string trainningFile = dataFolder + "/australian_scale.txt";
        }


        /// <summary>
        /// Train and test SVM, using low level api, construct kernels, solver and evaluator by hand
        /// </summary>
        private static void SVMClassifyLowLevel( string trainningFile,
            string testFile,
            int numberOfFeatures,
            float paramC)
        {

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();
            
            
            //EvaluatorBase<SparseVec> evaluator = new CudaLinearEvaluator();
            //EvaluatorBase<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            EvaluatorBase<SparseVec> evaluator = new SequentialDualEvaluator<SparseVec>();

            //IKernel<SparseVec> kernel = new CudaLinearKernel();
            //IKernel<SparseVec> kernel = new RbfKernel(gamma);
            IKernel<SparseVec> kernel = new LinearKernel();
            Model<SparseVec> model;

            Console.WriteLine("read vectors");
            Problem<SparseVec> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);
            Console.WriteLine("end read vectors");

            Console.WriteLine("kernel init");
            kernel.ProblemElements = train.Elements;
            kernel.Y = train.Y;
            kernel.Init();
           
            //
            //Solver = new ParallelSmoFanSolver<TProblemElement>(problem, kernel, C);
            //this solver works a bit faster and use less memory
            var Solver = new ParallelSmoFanSolver2<SparseVec>(train, kernel, C);

            Console.WriteLine("User solver {0} and kernel {1}", Solver.ToString(), kernel.ToString());

            Stopwatch timer = Stopwatch.StartNew();
            model = Solver.ComputeModel();
            Console.WriteLine("Model computed {0}  miliseconds={1}", timer.Elapsed, timer.ElapsedMilliseconds);

           
            

            var disSolver = Solver as IDisposable;
            if (disSolver != null)
                disSolver.Dispose();
            Solver = null;

            train.Dispose();
            


            Console.WriteLine("Start Testing");

            

            Problem<SparseVec> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);
           // evaluator.Kernel = kernel;
            evaluator.TrainedModel = model;
           
            evaluator.Init();
           

            Stopwatch t = Stopwatch.StartNew();
            float[] predictions = evaluator.Predict(test.Elements);
            t.Stop();
            //toremove: only for tests
            Console.WriteLine("prediction takes {0}  ms={1}",t.Elapsed, t.ElapsedMilliseconds);

            var disKernel = kernel as IDisposable;
            if (disKernel != null)
                disKernel.Dispose();
           
            //todo: Free evaluator memories
            var disposeEvaluator = evaluator as IDisposable;
            if (disposeEvaluator != null)
                disposeEvaluator.Dispose();
            
            
            
            int correct = 0;            
            for (int i = 0; i < test.ElementsCount; i++)
            {
                float predictedLabel = predictions[i];

                if (predictedLabel == test.Y[i])
                    ++correct;
            }
            test.Dispose();
            double accuracy = (float)correct / predictions.Length;
            Console.WriteLine("accuracy ={0}", accuracy);

        }

        /// <summary>
        /// Train and test Linear SVM, using low level api, construct kernels, solver and evaluator by hand
        /// </summary>
        private static void SVMLinearClassifyLowLevel(string trainningFile,
            string testFile,
            int numberOfFeatures,
            float paramC)
        {

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();


            EvaluatorBase<SparseVec> evaluator = new LinearPrimalEvaluator();
            Model<SparseVec> model;

            Console.WriteLine("read vectors");
            Problem<SparseVec> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);
            Console.WriteLine("end read vectors");


            //
            //Solver = new ParallelSmoFanSolver<TProblemElement>(problem, kernel, C);
            //this solver works a bit faster and use less memory
            var Solver = new LinearSolver(train, C);

            Console.WriteLine("User solver {0}", Solver.ToString());

            Stopwatch timer = Stopwatch.StartNew();
            model = Solver.ComputeModel();
            Console.WriteLine("Model computed {0}  miliseconds={1}", timer.Elapsed, timer.ElapsedMilliseconds);
            
            var disSolver = Solver as IDisposable;
            if (disSolver != null)
                disSolver.Dispose();
            Solver = null;

            train.Dispose();

            Console.WriteLine("Start Testing");



            Problem<SparseVec> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);
            // evaluator.Kernel = kernel;
            evaluator.TrainedModel = model;

            evaluator.Init();


            Stopwatch t = Stopwatch.StartNew();
            float[] predictions = evaluator.Predict(test.Elements);
            t.Stop();
            //toremove: only for tests
            Console.WriteLine("prediction takes {0}  ms={1}", t.Elapsed, t.ElapsedMilliseconds);

            //todo: Free evaluator memories
            var disposeEvaluator = evaluator as IDisposable;
            if (disposeEvaluator != null)
                disposeEvaluator.Dispose();

            int correct = 0;
            for (int i = 0; i < test.ElementsCount; i++)
            {
                float predictedLabel = predictions[i];

                if (predictedLabel == test.Y[i])
                    ++correct;
            }
            test.Dispose();
            double accuracy = (float)correct / predictions.Length;
            Console.WriteLine("accuracy ={0}", accuracy);

        }


        private static void SVMClassify<TProbElement>(
           Problem<TProbElement> train,
           Problem<TProbElement> test,
           IKernel<TProbElement> kernel,
           EvaluatorBase<TProbElement> evaluator,
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

        /// <summary>
        /// Do cross validation on train problem, uses different paramters C
        /// </summary>
        /// <param name="train"></param>
        /// <param name="kernel"></param>
        private static void DoCrossValidation(Problem<Vector> train, IKernel<Vector> kernel)
        {

            int folds = 3;

            //float[] penaltyC = new[] {0.125f, 0.025f, 0.5f, 1, 2,4,8,128};

            float[] penaltyC = new float[] { 0.5f, 4, 16, 128 };

            double acc = 0, bestC = 0;
            Validation<Vector> validation = new Validation<Vector>();
            validation.TrainingProblem = train;
            validation.Kernel = kernel;

            validation.Evaluator = new SequentialDualEvaluator<Vector>();
            Stopwatch timer = new Stopwatch();
            Stopwatch globalTimer = new Stopwatch();
            globalTimer.Start();
            for (int i = 0; i < penaltyC.Length; i++)
            {
                validation.C = penaltyC[i];
                timer.Reset();
                timer.Start();
                double tempAcc = validation.CrossValidation(folds);
                //.CrossValidation(train, kernel, penaltyC[i], folds);
                timer.Stop();
                Debug.WriteLine(string.Format("Tmp acuuracy = {0} C={1} time={2}", tempAcc, penaltyC[i], timer.Elapsed));
                if (tempAcc > acc)
                {
                    acc = tempAcc;
                    bestC = penaltyC[i];
                }
            }
            globalTimer.Stop();



            Console.WriteLine("Cross Validation nr folds={0} best acuuracy = {1} C={2} time={3}", folds, acc, bestC, globalTimer.Elapsed);
        }

        /// <summary>
        /// Testing methods for searching parameter C and Gamma
        /// </summary>
        /// <param name="train"></param>
        private static void FindParameterForRbf(Problem<Vector> train, IKernel<Vector> kernel)
        {
            var pSelection = kernel.CreateParameterSelection();
            pSelection.ShowDebug = true;



            float C;
            IKernel<Vector> bestKernel;
            Stopwatch sw = Stopwatch.StartNew();
            pSelection.SearchParams(train, out C, out bestKernel);

            sw.Stop();
            Console.WriteLine("Parameter selection time ={0}", sw.Elapsed);

        }


    

    /* some tests with Mahalanobis Kernel 
             /// <summary>
             /// 
             /// </summary>
             private static void ComputeLinearMahalanobisKernel()
             {
                 //original covariance matrix, but det==0
                 float[,] cov = new float[4, 4] {
                     { 30.0f/125, -5.0f/125, -30.0f/125, -15.0f/125},
                     { -5.0f/125, 30.0f/125, 5.0f/125, -10.0f/125},
                     { -30.0f/125, 5.0f/125, 30.0f/125, 15.0f/125},
                     { -15.0f/125, -10.0f/125, 15.0f/125, 20.0f/125}
                 };

                 FloatMatrix matrix = new FloatMatrix(cov);




                 FloatMatrix identity = 0.01f * FloatMatrix.Identity(4);

                 ////regularization, add to main diagonal some small value
                 matrix += identity;

                 Console.WriteLine("Matrix frobenius norm ={0}", MatrixFunctions.FrobNorm(matrix));

                 //matrix *= 4;
                 FloatSymmetricMatrix covMatrix = new FloatSymmetricMatrix(matrix);

                 Console.WriteLine("Matrix frobenius norm ={0}", covMatrix.DataVector.TwoNorm());

                 //DoubleSymmetricMatrix covMatrix = new DoubleSymmetricMatrix(3);


                 ShowMatrix(covMatrix);


                 Console.WriteLine("Det = " + MatrixFunctions.Determinant(covMatrix));
                 float norm = MatrixFunctions.OneNorm(covMatrix);
                 Console.WriteLine("Matrix one norm ={0}", norm);

                 FloatSymEigDecomp eigDecomp = new FloatSymEigDecomp(covMatrix);

                 Console.WriteLine("Eigen values=" + eigDecomp.EigenValues.ToString());



                 //DoubleSymmetricMatrix invertedCovMatrix = MatrixFunctions.Inverse(covMatrix);
                 FloatSymmetricMatrix invertedCovMatrix = MatrixFunctions.Inverse(covMatrix);

                 ShowMatrix(invertedCovMatrix);


                 // norm =MatrixFunctions.OneNorm(invertedCovMatrix);
                 //Console.WriteLine("Matrix one norm ={0}",norm);


                 //invertedCovMatrix *= invertedCovMatrix.Transpose();

                 norm = invertedCovMatrix.DataVector.TwoNorm();
                 Console.WriteLine("Matrix frobenius norm ={0}", norm);



                 invertedCovMatrix /= norm;

                 ShowMatrix(invertedCovMatrix);

                 Vector[] vectors = new Vector[]
                                        {
                                            new SparseVector(new double[]{1,2,3,1}),
                                            new SparseVector(new double[]{0,1,2,2}),
                                            new SparseVector(new double[]{1,1,3,2})

                                        };

                 Problem<Vector> train = new Problem<Vector>(vectors, new float[] { 1, 1, -1, -1 });
                 IKernel<Vector> kernel = new LinearMahalanobisKernel<Vector>(invertedCovMatrix);

                 kernel.ProblemElements = train.Elements;


                 for (int i = 0; i < vectors.Length; i++)
                 {
                     for (int j = i; j < vectors.Length; j++)
                     {
                         //Mahalanobis linear product
                         Console.WriteLine("M [{0},{1}]={2}", i, j, kernel.Product(i, j));
                         //Normal dot product
                         Console.WriteLine("N [{0},{1}]={2}", i, j, vectors[i].DotProduct(vectors[j]));
                         Console.WriteLine();
                     }
                 }


             }

       
             private static void ShowMatrix(FloatSymmetricMatrix covMatrix)
             {
                 for (int i = 0; i < covMatrix.Rows; i++)
                 {
                     for (int j = 0; j < covMatrix.Cols; j++)
                     {
                         Console.Write(" " + covMatrix[i, j]);
                     }
                     Console.WriteLine();
                 }
             }
             private static void ShowMatrix(DoubleSymmetricMatrix covMatrix)
             {
                 for (int i = 0; i < covMatrix.Rows; i++)
                 {
                     for (int j = 0; j < covMatrix.Cols; j++)
                     {
                         Console.Write(" " + covMatrix[i, j]);
                     }
                     Console.WriteLine();
                 }
             }
     */

    }

}