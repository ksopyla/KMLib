using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using KMLib.Kernels;
using KMLib.Evaluate;

namespace KMLib
{

    /// <summary>
    /// Contains validation procedures for test and trainning data.
    /// </summary>
    public class Validation<TProblemElement>
    {

        public float C { get; set; }

        /// <summary>
        /// Gets or sets the kernel.
        /// </summary>
        /// <value>The kernel.</value>
        public IKernel<TProblemElement> Kernel { get; set; }

        /// <summary>
        /// Gets or sets the training problem.
        /// </summary>
        /// <value>The training problem.</value>
        public Problem<TProblemElement> TrainingProblem { get; set; }

        /// <summary>
        /// Gets or sets the evaluator.
        /// </summary>
        /// <value>The evaluator.</value>
        public Evaluator<TProblemElement> Evaluator { get; set; }


        /// <summary>
        /// Empty constructor
        /// </summary>
        public Validation() { }


        /// <summary>
        /// Initializes a new instance of the <see cref="Validation&lt;TProblemElement&gt;"/> class.
        /// </summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="trainningProblem">The trainning problem.</param>
        /// <param name="evaluator">The evaluator.</param>
        public Validation(IKernel<TProblemElement> kernel,
                            Problem<TProblemElement> trainningProblem,
                            Evaluator<TProblemElement> evaluator)
        {
            Kernel = kernel;
            TrainingProblem = trainningProblem;
            Evaluator = evaluator;

        }



        /// <summary>
        /// Perform cross validation procedure, randomly divide array of problem elements
        /// on <see cref="nrFolds"/> parts, then train and test SVM
        /// </summary>
        /// <typeparam name="TProblemElement">The type of the problem element.</typeparam>
        /// <param name="nrFolds">Number of folds.</param>
        /// <returns>Average accuracy</returns>
        public double CrossValidation(int nrFolds)
        {
            List<TProblemElement>[] foldsElements;
            List<float>[] foldsLabels;
            int probSize = TrainingProblem.ElementsCount;
            //Validation<TProblemElement>.MakeFoldsSplit(TrainingProblem, nrFolds, out foldsElements, out foldsLabels);
            MakeFoldsSplit<TProblemElement>(TrainingProblem, nrFolds, out foldsElements, out foldsLabels);

            double accuracy = CrossValidateOnFolds(probSize, foldsElements, foldsLabels);
            return accuracy;
        }


        /// <summary>
        /// Split the problem elements into <see cref="nrFolds"/> parts.
        /// </summary>
        /// <typeparam name="TProblemElement">The type of the problem element.</typeparam>
        /// <param name="Problem">The problem.</param>
        /// <param name="nrFolds">Number of folds (number of parts).</param>
        /// <param name="foldsElements">Array of folds, each list contains elements which belonds to fold .</param>
        /// <param name="foldsLabels">Arrat of fodls, each list contains elements labels.</param>
        public static void MakeFoldsSplit<TElement>(Problem<TElement> Problem,
            int nrFolds, out List<TElement>[] foldsElements,
            out List<float>[] foldsLabels)
        {
            int probSize = Problem.ElementsCount;

            //array which stores computed permutation of problem element indexes
            int[] permutation = new int[probSize];

            int lastIndexOfFirstClass = GroupSameLabeledElements(Problem, permutation);


            Random rand = new Random();
            //stores information of which indexes are for each fold
            //int[] foldStart = new int[nrFolds + 1];

            //number of classes
            int nr_class = 2;

            //number of elements in each class
            int[] count = { lastIndexOfFirstClass + 1, probSize - 1 - lastIndexOfFirstClass };
            int[] start = { 0, lastIndexOfFirstClass + 1 };

            //do permutation in each group class
            for (int c = 0; c < nr_class; c++)
            {
                for (int i = 0; i < count[c]; i++)
                {
                    //method for permuation without repetition, in each class
                    int randInt = rand.Next(count[c] - i);
                    // int j = i + (int)(rand.NextDouble() * (count[c] - i));
                    int j = i + randInt;
                    int tmp = permutation[start[c] + j];
                    permutation[start[c] + j] = permutation[start[c] + i];
                    permutation[start[c] + i] = tmp;
                }
            }

            foldsElements = new List<TElement>[nrFolds];
            foldsLabels = new List<float>[nrFolds];

            for (int i = 0; i < nrFolds; i++)
            {
                foldsElements[i] = new List<TElement>(1 + probSize / nrFolds);
                foldsLabels[i] = new List<float>(1 + probSize / nrFolds);
            }

            //array for equaly distributed indexes into folds
            int foldIndex = 0;
            for (int c = 0; c < nr_class; c++)
            {
                for (int j = 0; j < count[c]; j++)
                {
                    int modIndex = foldIndex % nrFolds;
                    int eleIndex = permutation[j + start[c]];

                    foldsElements[modIndex].Add(Problem.Elements[eleIndex]);
                    foldsLabels[modIndex].Add(Problem.Y[eleIndex]);
                    foldIndex++;
                }
                foldIndex = 0;
            }
            //return probSize;
        }


        /// <summary>
        /// Do cross validation procedure on specified folds. 
        /// </summary>
        /// <typeparam name="TProblemElement">The type of the problem element.</typeparam>
        /// <param name="probSize">Size of the problem.</param>
        /// <param name="foldsElements">Array of folds, each list contains elements which belonds to fold .</param>
        /// <param name="foldsLabels">Arrat of fodls, each list contains elements labels.</param>
        /// <param name="Kernel">The kernel.</param>
        /// <param name="C">Penalty parameter C.</param>
        /// <returns>Accuracy</returns>
        public double CrossValidateOnFolds(
                                                int probSize,
                                                List<TProblemElement>[] foldsElements,
                                                List<float>[] foldsLabels)
        {
            Debug.Assert(foldsElements.Length == foldsLabels.Length, "array lenght should have the same lenght");
            int nrFolds = foldsElements.Length;
            long correct = 0L;
            for (int i = 0; i < nrFolds; i++)
            {
                //array of sub problem elements (comes form permutation)
                int subProbSize = 0;
                var trainFoldIndexes = from t in Enumerable.Range(0, nrFolds)
                                       where t != i
                                       select t;

                foreach (var s in trainFoldIndexes)
                {
                    subProbSize += foldsElements[s].Count;
                }

                float[] subLabels;
                TProblemElement[] subProbElem = CreateSubProblem(foldsElements, foldsLabels, trainFoldIndexes, subProbSize, out subLabels);

                //create sub problem based on previous subProblemElements and subProbLabels
                Problem<TProblemElement> trainSubprob = new Problem<TProblemElement>(subProbElem, subLabels);

                CSVM<TProblemElement> svm = new CSVM<TProblemElement>(trainSubprob, Kernel, C, Evaluator);

                svm.Init();
                svm.Train();

                for (int j = 0; j < foldsElements[i].Count; j++)
                {
                    var element = foldsElements[i][j];

                    var prediction = svm.Predict(element);
                    if (prediction == foldsLabels[i][j])
                    {
                        ++correct;
                    }
                }
            }

            //todo: accuracy count for cross validation
            return (float)correct / probSize;

        }

        /// <summary>
        /// Creates the sub problem.
        /// </summary>
        /// <typeparam name="TProblemElement">The type of the problem element.</typeparam>
        /// <param name="foldsElements">The array of list, each list contains folds elements.</param>
        /// <param name="foldsLabels">The array of list, each list contains folds elements labels.</param>
        /// <param name="trainFoldIndexes">Set of indexes, it indicates which folds in <see cref="foldsElements"/> are for trainning</param>
        /// <param name="subProbSize">Size of the original problem</param>
        /// <param name="subLabels">Out array for sub problem  labels.</param>
        /// <returns>Trainning sub problem elements</returns>
        private static TElement[] CreateSubProblem<TElement>(
            List<TElement>[] foldsElements,
            List<float>[] foldsLabels, IEnumerable<int> trainFoldIndexes,
            int subProbSize, out float[] subLabels)
        {
            TElement[] subProbElem = new TElement[subProbSize];

            List<TElement> groupElements = new List<TElement>(subProbSize);
            List<float> groupLabels = new List<float>(subProbSize);
            foreach (var trainFoldIndex in trainFoldIndexes)
            {
                Debug.Assert(foldsLabels[trainFoldIndex].Count == foldsElements[trainFoldIndex].Count, "---Two lists should have the same lenght");
                groupElements.AddRange(foldsElements[trainFoldIndex]);
                groupLabels.AddRange(foldsLabels[trainFoldIndex]);

            }

            subLabels = groupLabels.ToArray();
            subProbElem = groupElements.ToArray();

            Debug.Assert(subProbElem.Length == subLabels.Length, "two array should have same sizes");
            return subProbElem;
        }

        /// <summary>
        /// Group elements indexes from Problem into blocks(with same label) of indexes in permuation array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="Problem"></param>
        /// <param name="permutation"></param>
        /// <returns>index of last element from class one</returns>
        private static int GroupSameLabeledElements<T>(Problem<T> Problem, int[] permutation)
        {

            int startIndex = -1;
            int endindex = Problem.ElementsCount - 1;

            int i = 0;
            while (startIndex < endindex)
            {
                if (Problem.Y[i] <= 0)
                {
                    //we start from -1 so incrementation should be first
                    startIndex++;
                    permutation[startIndex] = i;

                }
                else
                {
                    //we start from last index so decrementation should be after
                    permutation[endindex] = i;
                    endindex--;
                }
                i++;
            }

            return startIndex;

        }




        /// <summary>
        /// Perform validation procedure, train on trainProblem and test on testProblem
        /// </summary>
        /// <typeparam name="TProblemElement">Problem element</typeparam>
        /// <param name="TrainingProblem">Train <see cref="Problem{TProblemElement}">Problem</see>  </param>
        /// <param name="TestProblem">Problem to test</param>
        /// <param name="Kernel">SVM kernel</param>
        /// <param name="C">Penalty parameter fo SVM solver</param>
        /// <returns>Accuracy</returns>
        public double TrainAndTestValidation(
            Problem<TProblemElement> TrainingProblem, Problem<TProblemElement> TestProblem)
        {

            CSVM<TProblemElement> svm = new CSVM<TProblemElement>(TrainingProblem, Kernel, C, Evaluator);

            Stopwatch t = Stopwatch.StartNew();
            svm.Init();
            
            Console.WriteLine("SVM init time {0}", t.Elapsed);
            
            svm.Train();
            
            Console.WriteLine("Svm train takes {0}", svm.model.ModelTimeMs);
            int correct = 0;
            
            Debug.WriteLine("Start Predict");
            t.Restart();

            var predictions = svm.Predict(TestProblem.Elements);
            t.Stop();
            for (int i = 0; i < TestProblem.ElementsCount; i++)
            {
                float predictedLabel = predictions[i];

                if (predictedLabel == TestProblem.Y[i])
                    ++correct;
            }

            double accuracy = (float)correct / TestProblem.ElementsCount;
          
            Console.WriteLine(string.Format("init, dispose and prediction on {0} elements takes {1}, correct={2}", TestProblem.ElementsCount, t.Elapsed,correct));
            return accuracy;
        }
    }
}
