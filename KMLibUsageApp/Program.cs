using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using dnAnalytics.LinearAlgebra;
using KMLib;
using KMLib.Helpers;
using KMLib.Kernels;
using System.Diagnostics;
using dnaLA = dnAnalytics.LinearAlgebra;

namespace KMLibUsageApp
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            string dataFolder = @"D:\UWM\praca naukowa\doktorat\code\KMLib\KMLibUsageApp\Data";


            string trainningFile = dataFolder + "/a1a.train";
            string testFile = dataFolder + "/a1a.test";

            //  string trainningFile = dataFolder + "/australian_scale.txt";

            // Problem<Vector> train = IOHelper.ReadVectorsFromFile(trainningFile);

            //in a1a problem max index is 123
            int numberOfFeatures = 123;
            Problem<Vector> train = IOHelper.ReadDNAVectorsFromFile(trainningFile,numberOfFeatures);

            Problem<Vector> test = IOHelper.ReadDNAVectorsFromFile(testFile,numberOfFeatures);




            IKernel<Vector> kernel = new RbfKernel(0.5f, train.Elements);
            //IKernel<Vector> kernel = new CosineKernel(train.Elements);
            //IKernel<Vector> kernel = new LinearKernel(train.Elements); 
            //IKernel<Vector> kernel = new PolinominalKernel(3, 0, 0.5, train.Elements);


           // SVMClassify(train,test, kernel);

            //DoCrossValidation(train, kernel);


            FindParameterForRbf(train);


            Console.ReadKey();

        }



        /// <summary>
        /// Testing methods for searching parameter C and Gamma
        /// </summary>
        /// <param name="train"></param>
        private static void FindParameterForRbf(Problem<Vector> train)
        {
            ParameterSelection pSelection = new ParameterSelection();
            pSelection.ShowDebug = true;
            

            double G;
            float C;
            Stopwatch sw = Stopwatch.StartNew();
            pSelection.GridSearchForRbfKerel(train,out C,out G);
            sw.Stop();
            Console.WriteLine("Parameter selection time ={0}",sw.Elapsed);

        }



        /// <summary>
        /// Train CSVM on train problem, and classify elements in test problem using different C
        /// </summary>
        /// <param name="train"></param>
        /// <param name="test"></param>
        /// <param name="kernel"></param>
        private static void SVMClassify(Problem<Vector> train, Problem<Vector> test, IKernel<Vector> kernel)
        {
           

           //float[] penaltyC = new[] {0.125f, 0.025f, 0.5f, 1, 2,4,8,128};

            float[] penaltyC = new float[] { 0.5f, 4 ,32, 128 };

            double acc = 0, bestC = 0;

            Stopwatch timer = new Stopwatch();
            Stopwatch globalTimer = new Stopwatch();
            globalTimer.Start();
            for (int i = 0; i < penaltyC.Length; i++)
            {

                timer.Reset();
                timer.Start();

                double tempAcc = Validation.TestValidation(train, test, kernel, penaltyC[i]);
                timer.Stop();
                Console.WriteLine("Tmp acuuracy = {0} C={1} time={2}", tempAcc, penaltyC[i], timer.Elapsed);
                if (tempAcc > acc)
                {
                    acc = tempAcc;
                    bestC = penaltyC[i];
                }
            }
            globalTimer.Stop();

            Console.WriteLine("Validation on test data best acuuracy = {0} C={1} time={2}", acc, bestC, globalTimer.Elapsed);
            
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

            Stopwatch timer = new Stopwatch();
            Stopwatch globalTimer = new Stopwatch();
            globalTimer.Start();
            for (int i = 0; i < penaltyC.Length; i++)
            {

                timer.Reset();
                timer.Start();
                double tempAcc = Validation.CrossValidation(train, kernel, penaltyC[i], folds);
                timer.Stop();
                Console.WriteLine("Tmp acuuracy = {0} C={1} time={2}", tempAcc, penaltyC[i], timer.Elapsed);
                if (tempAcc > acc)
                {
                    acc = tempAcc;
                    bestC = penaltyC[i];
                }
            }
            globalTimer.Stop();



            Console.WriteLine("Cross Validation nr folds={0} best acuuracy = {1} C={2} time={3}", folds, acc, bestC, globalTimer.Elapsed);
        }



        
    }
}