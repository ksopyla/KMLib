using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using KMLib;
using KMLib.Helpers;
using KMLib.Kernels;
using KMLib.GPU;
using System.Diagnostics;
using dnaLA = dnAnalytics.LinearAlgebra;
using KMLib.Evaluate;


namespace KMLibUsageApp
{
    internal class Program
    {
        private static float C = 4f;
        private static void Main(string[] args)
        {
            if (args.Length < 1)
                throw new ArgumentException("to liitle arguments");

            Debug.Listeners.Add(new ConsoleTraceListener());

            string dataFolder = args[0];// @"D:\UWM\praca naukowa\doktorat\code\KMLib\KMLibUsageApp\Data";

            IList<Tuple<string, string, int>> dataSetsToTest = CreateDataSetList(dataFolder);

          GroupedTestingDataSets(dataSetsToTest);
            
            //TestOneDataSet(dataFolder);

            //TestOneDataSetWithCuda(dataFolder);

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
            Problem<SparseVector> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);

            Problem<SparseVector> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);

            //EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();
            float gamma = 0.5f;
            //EvaluatorBase<SparseVector> evaluator = new RBFEvaluator(gamma);
            EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();

           // evaluator.Init();
            //IKernel<Vector> kernel = new PolinominalKernel(3, 0.5, 0.5);
            //IKernel<SparseVector> kernel = new RbfKernel(gamma);
            IKernel<SparseVector> kernel = new LinearKernel();
            SVMClassify(train, test, kernel, evaluator,C);

        }


        private static void TestOneDataSetWithCuda(string dataFolder)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;
            ChooseDataSet(dataFolder, out trainningFile, out testFile, out numberOfFeatures);

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();
            Problem<SparseVector> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);

            Problem<SparseVector> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);

            
            
            EvaluatorBase<SparseVector> evaluator = new CudaLinearEvaluator();
         
            IKernel<SparseVector> kernel2 = new CudaLinearKernel();

            SVMClassify(train, test, kernel2,evaluator, C);


           // SVMClassify(train, test, kernel2, evaluator, C);

            
        }

        private static void GroupedTestingDataSets(IList<Tuple<string, string, int>> dataSetsToTest)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;


            float gamma = 0.5f;
            //EvaluatorBase<SparseVector> evaluator = new RBFEvaluator(gamma);
            //IKernel<SparseVector> kernel = new RbfKernel(gamma);

            EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();
            IKernel<SparseVector> kernel = new LinearKernel();

            foreach (var data in dataSetsToTest)
            {
                trainningFile = data.Item1;
                testFile = data.Item2;
                numberOfFeatures = data.Item3;

                Console.WriteLine("\n----------------------------------------------\n");
                Console.WriteLine("DataSets , trainning={1} testing={2} , atr={0}", numberOfFeatures, trainningFile, testFile);
                Console.WriteLine();
                

                Problem<SparseVector> train = IOHelper.ReadDNAVectorsFromFile(trainningFile, numberOfFeatures);

                Problem<SparseVector> test = IOHelper.ReadDNAVectorsFromFile(testFile, numberOfFeatures);
                
                SVMClassify(train, test, kernel, evaluator, C);
                Console.WriteLine("***************************\n");

            }
        }

        private static IList<Tuple<string, string, int>> CreateDataSetList(string dataFolder)
        {
            List<Tuple<string, string, int>> dataSets = new List<Tuple<string, string, int>>(8);
            /*
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
            */
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
            //string testFile = dataFolder + "/a1a.train";
            //in a1a problem max index is 123
            numberOfFeatures = 123;

            //trainningFile = dataFolder + "/a9a";
            //testFile = dataFolder + "/a9a.t";
            //numberOfFeatures = 123;

            //string trainningFile = dataFolder + "/w8a";
            //string testFile = dataFolder + "/w8a.t";
            //int numberOfFeatures = 300;

            //string trainningFile = dataFolder + "/colon-cancer.train";
            //string testFile = dataFolder + "/colon-cancer.train";
            //int numberOfFeatures = 2000;

            //string trainningFile = dataFolder + "/leu";
            //string testFile = dataFolder + "/leu.t";
            //int numberOfFeatures = 7129;

            //string trainningFile = dataFolder + "/duke";
            //string testFile = dataFolder + "/duke.tr";
            //int numberOfFeatures = 7129;

            //string trainningFile = dataFolder + "/rcv1_train.binary";
            ////string trainningFile = dataFolder + "/rcv1_test.binary";
            //string testFile = dataFolder + "/rcv1_train_test.binary";
            //int numberOfFeatures = 47236;

            //string trainningFile = dataFolder + "/news20.binary";
            //string testFile = dataFolder + "/news20_test.binary";
            //int numberOfFeatures = 1335191;

            //string trainningFile = dataFolder + "/mnist.scale";
            //string testFile = dataFolder + "/mnist.scale.t";
            //int numberOfFeatures = 784;


            //string trainningFile = dataFolder + "/real-sim_small_3K";
            //string trainningFile = dataFolder + "/real-sim_med_6K";
            //string trainningFile = dataFolder + "/real-sim_med_10K";
            //string trainningFile = dataFolder + "/real-sim";
            //string testFile = dataFolder + "/real-sim.t";
            //int numberOfFeatures = 20958;

            //for test
            //string trainningFile = dataFolder + "/liver-disorders_scale_small.txt";
            //string testFile = dataFolder + "/liver-disorders_scale_small.txt";
            //////string trainningFile = dataFolder + "/liver-disorders_scale.txt";
            //////string testFile = dataFolder + "/liver-disorders_scale.txt";
            //int numberOfFeatures = 6;
            //  string trainningFile = dataFolder + "/australian_scale.txt";
        }


        /// <summary>
        /// Train CSVM on train problem, and classify elements in test problem using different C
        /// </summary>
        /// <param name="train"></param>
        /// <param name="test"></param>
        /// <param name="kernel"></param>
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

            validation.Evaluator = new SequentialEvaluator<Vector>();
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