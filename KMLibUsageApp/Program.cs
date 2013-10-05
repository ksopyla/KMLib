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

        //IDataTransform<SparseVec> dataTransform = new LpNorm(1);
        static IDataTransform<SparseVec> dataTransform = new NullTransform();

        private static void Main(string[] args)
        {
            if (args.Length < 1)
                throw new ArgumentException("to liitle arguments");
            string dataFolder = args[0];
            //dataFolder = @"./Data";

            Debug.Listeners.Add(new ConsoleTraceListener());

                        
            IList<Tuple<string, string, int>> dataSetsToTest = CreateDataSetList(dataFolder);

            //Console.WriteLine("press any key to start");
            //Console.ReadKey();
            //GroupedTestingDataSets(dataSetsToTest);
            
            //GroupedTestingLowLevelDataSets(dataSetsToTest);
            
            //TestOneDataSet(dataFolder);

            //TestOneDataSetWithCuda(dataFolder);

            //TestMultiClasDataSet(dataFolder);

            // TestRanking(dataFolder);

            string trainningFile;
            string testFile;
            int numberOfFeatures;
            ChooseDataSet(dataFolder, out trainningFile, out testFile, out numberOfFeatures);

                  
            SVMClassifyLowLevel(trainningFile, testFile, numberOfFeatures, C);
            //SVMClassifyLowLevelManyTests(trainningFile, testFile, numberOfFeatures, C,3);

            //SVMLinearClassifyLowLevel(trainningFile, testFile, numberOfFeatures, C);


            // PerformCrossValidation(dataFolder, folds);

            //SVMClassifyLowLevel(trainningFile, testFile, numberOfFeatures, C);
            Console.WriteLine("Press any button");
             Console.ReadKey();

        }

        private static void PerformCrossValidation(string dataFolder, int folds)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;
            ChooseDataSet(dataFolder, out trainningFile, out testFile, out numberOfFeatures);

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("Cross validation folds={0} \nDataSet1 atr={1}, trainning={2}", folds, numberOfFeatures, trainningFile);
            Console.WriteLine();
            Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);


            //EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();
            Evaluator<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            //EvaluatorBase<SparseVec> evaluator = new SequentialDualEvaluator<SparseVec>();

            // evaluator.Init();
            //IKernel<Vector> kernel = new PolinominalKernel(3, 0.5, 0.5);
            IKernel<SparseVec> kernel = new RbfKernel(gamma);
            //IKernel<SparseVec> kernel = new LinearKernel();

            DoCrossValidation(train, kernel, evaluator, folds);

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
            Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);
            
            Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);
            
            //Do dataset Normalization
                        
            train.Elements = dataTransform.Transform(train.Elements);
            test.Elements = dataTransform.Transform(test.Elements);

            
            //EvaluatorBase<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            Evaluator<SparseVec> evaluator = new DualEvaluator<SparseVec>();

            // evaluator.Init();
            //IKernel<Vector> kernel = new PolinominalKernel(3, 0.5, 0.5);
            IKernel<SparseVec> kernel = new RbfKernel(gamma);
            //IKernel<SparseVec> kernel = new LinearKernel();
           // IKernel<SparseVec> kernel = new ChiSquaredKernel();
            //IKernel<SparseVec> kernel = new ChiSquaredNormKernel();
            //IKernel<SparseVec> kernel = new ExpChiSquareKernel(gamma);
            
            SVMClassify(train, test, kernel, evaluator, C);

        }





        private static void TestMultiClasDataSet(string dataFolder)
        {
            //string trainningFile = dataFolder + "/glass.scale";
            //string testFile = dataFolder + "/glass.scale"; ;
            //int numberOfFeatures=9;

            string trainningFile = dataFolder + "/genresTrain_scale.train";
            string testFile = dataFolder + "/genresTest.arff_scale.t"; ;
            int numberOfFeatures = 181;



            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();
            Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);

            Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);

            //EvaluatorBase<SparseVector> evaluator = new SequentialEvaluator<SparseVector>();

            Evaluator<SparseVec> evaluator = new RBFDualEvaluator(gamma);
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
            float[] predictions = mcSvm.Predict(test.Elements);


            t.Stop();
            //toremove: only for tests
            Console.WriteLine("prediction takes {0}  ms={1}", t.Elapsed, t.ElapsedMilliseconds);


            SavePredictionToFile(predictions, Path.GetFileNameWithoutExtension(testFile) + ".prediction");

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


        private static void TestRanking(string dataFolder)
        {

            string trainningFile = dataFolder + "/rankingTrain.t";
            string testFile = dataFolder + "/rankingTest.t"; ;
            int numberOfFeatures = 2;



            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);
            Console.WriteLine("Ranking DataSets atr={0}, trainning={1} testing={2}", numberOfFeatures, trainningFile, testFile);
            Console.WriteLine();
            Problem<SparseVec> origtrain = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);



            Problem<SparseVec> train = CreateRankingProblem(origtrain);

            Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);

            Evaluator<SparseVec> evaluator = new LinearPrimalEvaluator();
            Model<SparseVec> model;

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

            evaluator.TrainedModel = model;

            evaluator.Init();



            float[] predictions = new float[test.Elements.Length];
            Stopwatch t = Stopwatch.StartNew();
            for (int i = 0; i < test.ElementsCount; i++)
            {
                predictions[i] = evaluator.PredictVal(test.Elements[i]);
            }
            t.Stop();
            //toremove: only for tests
            Console.WriteLine("ranking prediction takes {0}  ms={1}", t.Elapsed, t.ElapsedMilliseconds);

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
        /// Create ranking problem, computes vector parwise substraction
        ///  y_i < y_j  (x_i - x_j) -> 1
        /// </summary>
        /// <param name="origtrain"></param>
        /// <returns></returns>
        private static Problem<SparseVec> CreateRankingProblem(Problem<SparseVec> origtrain)
        {

            int k = origtrain.ElementsCount;
            int size = k * (k + 1) / 2;

            List<SparseVec> pairVector = new List<SparseVec>(size);
            List<float> pairLabels = new List<float>(size);

            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < (i + 1); j++)
                {
                    if (i == j)
                        continue;

                    int ii = i, jj = j;

                    //add some permutation
                    //if ( (j+i*(i+1)/2) % 2 == 0) { ii = j; jj = i; }

                    SparseVec subVec = origtrain.Elements[ii].Subtract(origtrain.Elements[jj]);
                    pairVector.Add(subVec);

                    float labelDiff = origtrain.Y[ii] - origtrain.Y[jj];
                    float label = 1;
                    if (labelDiff < 0)
                        label = -1;
                    pairLabels.Add(label);
                }
            }

            Problem<SparseVec> rankingProb = new Problem<SparseVec>(pairVector.ToArray(), pairLabels.ToArray(), 2, 2, new float[2] { -1, 1 });


            return rankingProb;

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
            Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);

            var trainSumArr = train.Elements.Sum(x => x.Indices.Length);
            var trainSum = train.Elements.Sum(x => x.Count);

            Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);
            var testSumArr = test.Elements.Sum(x => x.Indices.Length);
            var testSum = test.Elements.Sum(x => x.Count);

            Evaluator<SparseVec> evaluator = new DualEvaluator<SparseVec>();

            //EvaluatorBase<SparseVec> evaluator = new CudaLinearEvaluator();
            // EvaluatorBase<SparseVec> evaluator = new CudaRBFEvaluator(gamma);

            IKernel<SparseVec> kernel2 = new CuLinearKernel();
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

            Evaluator<SparseVec> evaluator = new CudaLinearCSREvaluator();
            IKernel<SparseVec> kernel = new CuLinearKernel();

            foreach (var data in dataSetsToTest)
            {
                trainningFile = data.Item1;
                testFile = data.Item2;
                numberOfFeatures = data.Item3;

                Console.WriteLine("\n----------------------------------------------\n");
                Console.WriteLine("DataSets , trainning={1} testing={2} , atr={0}", numberOfFeatures, trainningFile, testFile);
                Console.WriteLine();


                Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);

                Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);

                SVMClassify(train, test, kernel, evaluator, C);
                Console.WriteLine("***************************\n");

            }
        }

        private static void GroupedTestingLowLevelDataSets(IList<Tuple<string, string, int>> dataSetsToTest)
        {
            string trainningFile;
            string testFile;
            int numberOfFeatures;

            //FileStream filestream = new FileStream("out.txt", FileMode.Create);
            //var streamwriter = new StreamWriter(filestream);
            //streamwriter.AutoFlush = true;
            //Console.SetOut(streamwriter);
            //Console.SetError(streamwriter);

            Trace.Listeners.Clear();

            String fileName = string.Format("results_{0:yyyy-MM-dd_H_mm_ss}.txt", DateTime.Now);
            //String fileName = "out.txt";
            FileStream filestream = new FileStream(fileName, FileMode.Create);
            var streamwriter = new StreamWriter(filestream);
            TextWriterTraceListener twL = new TextWriterTraceListener(streamwriter);
            //TextWriterTraceListener twL = new TextWriterTraceListener("out.txt");
            twL.Name = "TextLogger";
            twL.TraceOutputOptions = TraceOptions.ThreadId | TraceOptions.DateTime;
            ConsoleTraceListener ctl = new ConsoleTraceListener(false);
            ctl.TraceOutputOptions = TraceOptions.DateTime;

            Trace.Listeners.Add(twL);
            Trace.Listeners.Add(ctl);
            Trace.AutoFlush = true;


            


            Evaluator<SparseVec> evaluator = new DualEvaluator<SparseVec>();
            Model<SparseVec> model=null;

            IList<IKernel<SparseVec>> kernelsCollection = CreateKernels();

            IList<string> solversStr = new List<string> { "ParallelSMO","GpuFanSolver" };// "ParallelSMO", 

            int numTests = 3;

            foreach (var data in dataSetsToTest)
            {
                trainningFile = data.Item1;
                testFile = data.Item2;
                numberOfFeatures = data.Item3;

                //Read sets into problem class
                Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);
                train.Elements = dataTransform.Transform(train.Elements);
                Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);
                test.Elements = dataTransform.Transform(test.Elements);

                var tr = Path.GetFileName(trainningFile);
                var tst = Path.GetFileName(testFile);

                Trace.WriteLine(DateTime.Now);
                Trace.WriteLine(string.Format("DataSet: tr={0} tst={1} , el={2}/{3}  atr={4}", tr, tst, train.ElementsCount, test.ElementsCount, numberOfFeatures));
                Trace.WriteLine("-----------------------------------------------------------------");


                foreach (var solverStr in solversStr)
                {
                    //dominionstat dla parallel smo
                    if (solverStr.Equals("ParallelSMO") && data.Item3 == 596)
                        continue;

                    Trace.WriteLine(string.Format("Solver: {0}", solverStr));
                    Trace.WriteLine(string.Format("Results time[s]:{0,68} {1,9} {2,12} {3,9}", "it", "obj","nSV","acc" ));
                    foreach (var kernel in kernelsCollection)
                    {
                        string kernelStr = kernel.ToString();
                        Trace.Write(string.Format("{0}  {1,-17}:", DateTime.Now, kernelStr));

                        Solver<SparseVec> solver = null;

                        try
                        {
                            kernel.ProblemElements = train.Elements;
                            kernel.Y = train.Y;
                            kernel.Init();
                            solver = CreateSolver(solverStr, train, kernel, C);

                            long[] modelTimes = new long[numTests];


                            for (int i = 0; i < numTests; i++)
                            {
                                model = solver.ComputeModel();

                                modelTimes[i] = model.ModelTimeMs;
                                double mTime = modelTimes[i] / 1000.0;
                                Trace.Write(string.Format(" {0,9}; ", mTime.ToString("0.0")));

                            }
                        }
                        catch (Exception e)
                        {

                            Trace.WriteLine(string.Format(" {0,9}; ", "xxxx"));
                            //Trace.TraceError(e.Message);
                            continue;
                        }
                        finally
                        {

                            var disSolver = solver as IDisposable;
                            if (disSolver != null)
                                disSolver.Dispose();
                            solver = null;

                            var disKernel = kernel as IDisposable;
                            if (disKernel != null)
                                disKernel.Dispose();

                        }

                        evaluator.Kernel = kernel;
                        evaluator.TrainedModel = model;

                        evaluator.Init();
 
                        float[] predictions = evaluator.Predict(test.Elements);
                        double acc = GetAccuracy(test, predictions);

                        int it = model.Iter;
                        float obj = model.Obj;
                        int nSv = model.SupportElements.Length;

                        Trace.WriteLine(string.Format("{0,7}\t{1}\t{2}\t{3} ", it, obj.ToString("0.00"), nSv, acc.ToString("n5")));
                        
                    }

                }


                train.Dispose();
                test.Dispose();
                Trace.WriteLine("***************************\n");

            }
        }

        private static IList<IKernel<SparseVec>> CreateKernels()
        {
            IList<IKernel<SparseVec>> kernelsCollection = new List<IKernel<SparseVec>> { 
                new CuRBFCSRKernel(gamma), 
                new CuRBFEllpackKernel(gamma),
                new CuRBFSlEllKernel(gamma),
                new CuRBFEllILPKernel(gamma),
                new CuRBFERTILPKernel(gamma),
                new CuRBFSERTILPKernel(gamma)
            };
            return kernelsCollection;
        }

        private static double GetAccuracy(Problem<SparseVec> test, float[] predictions)
        {
            int correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                float predictedLabel = predictions[i];

                if (predictedLabel == test.Y[i])
                    ++correct;
            }

            double acc = (float)correct / predictions.Length;
            return acc;
        }

        private static Solver<SparseVec> CreateSolver(string solverType,Problem<SparseVec> train, IKernel<SparseVec> kernel, float C)
        {
            if (solverType.Equals("ParallelSMO"))
                return new ParallelSmoFanSolver2<SparseVec>(train, kernel, C);
            else if(solverType.Equals("GpuFanSolver"))
                return new GPUSmoFanSolver(train, kernel, C);

            return null;
 
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



            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/w8a",
            //    dataFolder + "/w8a.t",
            //    300));

            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/a9a",
            //    dataFolder + "/a9a.t",
            //    123));

            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/news20.binary",
            //    dataFolder + "/news20.binary",
            //    1335191));

           
            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/real-sim",
            //    dataFolder + "/real-sim",
            //    20958));

            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/rcv1_test.binary",
            //    dataFolder + "/rcv1_train.binary",
            //    47236));

            //dataSets.Add(new Tuple<string, string, int>(
            //    dataFolder + "/mnist.scale",
            //    dataFolder + "/mnist.scale.t",
            //    784));



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
                dataFolder + "/kytea-msr_first_1M.train",
                dataFolder + "/kytea-msr.test",
                8683737));

            dataSets.Add(new Tuple<string, string, int>(
                dataFolder + "/kytea-msr_first_500k.train",
                dataFolder + "/kytea-msr.test",
                8683737));

            

            dataSets.Add(new Tuple<string, string, int>(
               dataFolder + "/url_combined_1.5M.train",
               dataFolder + "/url_combined_800k.test",
               3231961));

            dataSets.Add(new Tuple<string, string, int>(
               dataFolder + "/kdda_2M",
               dataFolder + "/kdda.t",
               20216830));

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


            trainningFile = dataFolder + "/a1a.train";
            //testFile = dataFolder + "/a1a.test";
            ////testFile = dataFolder + "/a1a.train";
            testFile = dataFolder + "/a1a.train";
            //in a1a problem max index is 123
            numberOfFeatures = 123;


            //trainningFile = dataFolder + "/a9a";
            //testFile = dataFolder + "/a9a.t";
            ////testfile = datafolder + "/a9a";
            //numberOfFeatures = 123;

            //trainningFile = dataFolder + "/a9a_128.train";
            //testFile = dataFolder + "/a9a.t";
            ////testFile = dataFolder + "/a9a";
            //numberOfFeatures = 123;

            //trainningFile = dataFolder + "/w8a";
            //testFile = dataFolder + "/w8a.t";
            //numberOfFeatures = 300;

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
            
            Evaluator<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new DualEvaluator<SparseVec>();
            //Evaluator<SparseVec> evaluator = new CuRBFEllILPEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new CuRBFEllpackEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new CuRBFERTILPEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new CuRBFSlEllEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new CuRBFSERTILPEvaluator(gamma);


            #region Cuda kernels

            //IKernel<SparseVec> kernel = new CuLinearKernel();
            //IKernel<SparseVec> kernel = new CuRBFCSRKernel(gamma );
            //IKernel<SparseVec> kernel = new CuRBFEllpackKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFEllILPKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFERTILPKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFSlEllKernel(gamma);
            IKernel<SparseVec> kernel = new CuRBFSERTILPKernel(gamma);

            //IKernel<SparseVec> kernel = new CuRBFEllILPKernelCol2(gamma);

            //IKernel<SparseVec> kernel = new CuChi2EllKernel();
            //IKernel<SparseVec> kernel = new CuNChi2EllKernel();
            //IKernel<SparseVec> kernel = new CuExpChiEllKernel(gamma);


            #endregion
            //IKernel<SparseVec> kernel = new RbfKernel(gamma);
            //IKernel<SparseVec> kernel = new LinearKernel();

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

            Stopwatch timer = Stopwatch.StartNew();
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



        /// <summary>
        /// Train and test SVM, using low level api, construct kernels, solver and evaluator by hand
        /// </summary>
        private static void SVMClassifyLowLevelManyTests(string trainningFile,
            string testFile,
            int numberOfFeatures,
            float paramC,int numTests)
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

            Model<SparseVec> model=null;
            //EvaluatorBase<SparseVec> evaluator = new CudaLinearEvaluator();
            //EvaluatorBase<SparseVec> evaluator = new CudaRBFEvaluator(gamma);
            //Evaluator<SparseVec> evaluator = new RBFDualEvaluator(gamma);
            Evaluator<SparseVec> evaluator = new DualEvaluator<SparseVec>();

            #region Cuda kernels

            //IKernel<SparseVec> kernel = new CuLinearKernel();
            //IKernel<SparseVec> kernel = new CuRBFCSRKernel(gamma );
            //IKernel<SparseVec> kernel = new CuRBFEllpackKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFEllILPKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFEllRTILPKernel(gamma);
            //IKernel<SparseVec> kernel = new CuRBFSlEllKernel(gamma);
            IKernel<SparseVec> kernel = new CuRBFSERTILPKernel(gamma);


            //IKernel<SparseVec> kernel = new CuChi2EllKernel();
            //IKernel<SparseVec> kernel = new CuNChi2EllKernel();
            //IKernel<SparseVec> kernel = new CuExpChiEllKernel(gamma);


            #endregion
            //IKernel<SparseVec> kernel = new RbfKernel(gamma);
            //IKernel<SparseVec> kernel = new LinearKernel();

            Console.WriteLine("kernel init");
            kernel.ProblemElements = train.Elements;
            kernel.Y = train.Y;
            kernel.Init();

            //var Solver = new ParallelSmoFanSolver<SparseVec>(train, kernel, C);
            //this solver works a bit faster and use less memory
            //var Solver = new ParallelSmoFanSolver2<SparseVec>(train, kernel, C);
            //var Solver = new SmoFanSolver<SparseVec>(train, kernel, C);
            //var Solver = new SmoRandomSolver<SparseVec>(train, kernel, C);

            var Solver = new GPUSmoFanSolver(train, kernel, C);

            Console.WriteLine("User solver {0} and kernel {1}", Solver.ToString(), kernel.ToString());

            long[] modelTimes = new long[numTests];

            Stopwatch timer = Stopwatch.StartNew();

            for (int i = 0; i < numTests; i++)
            {
                model = Solver.ComputeModel();

                modelTimes[i]= model.ModelTimeMs;
                
                Console.Write(model.ToString());
            }

            Console.WriteLine("average time of {0} runs is:{1}", numTests, modelTimes.Average());



            //model.WriteToFile("modelFileCU.txt");

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
            //toremove: only for tests
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

            Evaluator<SparseVec> evaluator = new LinearPrimalEvaluator();
            Model<SparseVec> model;

            Console.WriteLine("read vectors");
            Problem<SparseVec> train = IOHelper.ReadVectorsFromFile(trainningFile, numberOfFeatures);
            Console.WriteLine("end read vectors");


            //var Solver = new CUDALinSolver(train, C);
            //var Solver = new ConjugateLinSolver(train, C);

            //C =(float) Math.Sqrt(C)*C / train.ElementsCount;
            var Solver = new BBLinSolver(train, C);
            //var Solver = new GPUnmBBLinSolver(train, C);
            //var Solver = new GPUstdBBLinSolver(train, C);
            // var Solver = new LinearSolver(train, C);

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



            Problem<SparseVec> test = IOHelper.ReadVectorsFromFile(testFile, numberOfFeatures);
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

        /// <summary>
        /// Do cross validation on train problem, uses different paramters C
        /// </summary>
        /// <param name="train"></param>
        /// <param name="kernel"></param>
        private static void DoCrossValidation(Problem<SparseVec> train, IKernel<SparseVec> kernel, Evaluator<SparseVec> evaluator, int folds)
        {

            float[] penaltyC = new[] { 0.125f, 0.025f, 0.5f, 1, 2, 4, 8, 16, 32, 64, 128 };

            //float[] penaltyC = new float[] { 0.5f, 4, 16, 128 };

            double acc = 0, bestC = 0;
            Validation<SparseVec> validation = new Validation<SparseVec>();
            validation.TrainingProblem = train;
            validation.Kernel = kernel;

            validation.Evaluator = evaluator;
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
                Console.WriteLine(string.Format("Tmp acuuracy = {0} C={1} time={2}", tempAcc, penaltyC[i], timer.Elapsed));
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