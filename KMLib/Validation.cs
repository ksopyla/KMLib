using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using KMLib.Kernels;

namespace KMLib
{
    public class Validation
    {
        // private int nrFolds = 5;
        // private float percent = 0.3f;


        /// <summary>
        /// Empty constructor
        /// </summary>
        public Validation() { }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="nrOfFolds">Number of folds for validation</param>
        //public CrossValidation(int nrOfFolds)
        //{
        //    if (nrOfFolds < 2)
        //        throw new ArgumentOutOfRangeException("nrOfFolds", "should be more then 1");
        //    nrFolds = nrOfFolds;
        //}


        /// <summary>
        /// Perform cross validation procedure, randomly divide array of elements
        /// on <see cref="nrFolds"/> parts, then train and test SVM
        /// </summary>
        /// <typeparam name="TProblemElement">Problem element</typeparam>
        /// <param name="problem">problem wich will be split in folds and validate</param>
        /// <param name="kernel">used kernel</param>
        /// <param name="C">penalty param C i C-Svm</param>
        /// <param name="nrFolds">number of folds</param>
        /// <returns>average accuracy</returns>
        public static double CrossValidationOld<TProblemElement>(
                                                Problem<TProblemElement> problem,
                                                IKernel<TProblemElement> kernel, float C, int nrFolds)
        {
            int probSize = problem.ElementsCount;

            //array for all folds predictions
            float[] foldPredictions = new float[probSize];

            //array which stores computed permutation of problem element indexes
            int[] permutation = new int[probSize];

            int lastIndexOfFirstClass = GroupSameLabeledElements(problem, permutation);


            Random rand = new Random();
            //stores information of which indexes are for each fold
            int[] foldStart = new int[nrFolds + 1];

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


            //initialize permutation array
            //for (int i = 0; i < probSize; i++)
            //    permutation[i] = i;

            //sets each fold start index
            SetFoldsStarts(probSize, nrFolds, foldStart);


            //array for equaly distributed indexes into folds
            int[] indexes = new int[probSize];
            for (int c = 0; c < nr_class; c++)
            {
                int begin = -1;
                int end = 0;
                for (int i = 0; i < nrFolds; i++)
                {
                    begin = start[c] + i * count[c] / nrFolds;

                    end = start[c] + (i + 1) * count[c] / nrFolds;
                    for (int j = begin; j < end; j++)
                    {
                        indexes[foldStart[i]] = permutation[j];
                        foldStart[i]++;
                    }
                }
            }
            //above we modify folds starst so we have to set it again
            SetFoldsStarts(probSize, nrFolds, foldStart);
            //for (int i = 0; i <= nrFolds; i++)
            //    foldStart[i] = i * probSize / nrFolds;



            ////compute permutation, randomly swap pair of indexes
            for (int i = 0; i < probSize; i++)
            {
                //method for permuation without repetition
                int j = i + (int)(rand.NextDouble() * (probSize - i));

                //change new index j with current i
                int temp = permutation[i];
                permutation[i] = permutation[j];
                permutation[j] = temp;
            }


            //todo: good for paralle computing, think about it later, do validation!
            for (int i = 0; i < nrFolds; i++)
            {

                //Problem<TProblemElement> test;
                //Problem<TProblemElement> train;
                //Split(problem, out test, out train);

                //SVM<TProblemElement> svMachine = new SVM<TProblemElement>(train,kernel,C);

                int begin = foldStart[i];
                int end = foldStart[i + 1];

                //count number of elements for i-th fold
                int subProbSize = probSize - (end - begin);

                //array of sub problem elements (comes form permutation)
                TProblemElement[] subProbElem = new TProblemElement[subProbSize];
                float[] subLabels = new float[subProbSize];

                int k = 0;
                //to subProblem add elements from [0,begin) and [end,probSize)
                //elements from [begin,end) are used for testing
                for (int j = 0; j < begin; j++)
                {
                    subProbElem[k] = problem.Elements[indexes[j]];
                    subLabels[k] = problem.Labels[indexes[j]];
                    ++k;
                }
                for (int j = end; j < probSize; j++)
                {
                    subProbElem[k] = problem.Elements[indexes[j]];
                    subLabels[k] = problem.Labels[indexes[j]];
                    ++k;
                }
                //create sub problem based on previous subProblemElements and subProbLabels
                Problem<TProblemElement> trainSubprob = new Problem<TProblemElement>(subProbElem, subLabels);
                kernel.ProblemElements = subProbElem;
                CSVM<TProblemElement> svm = new CSVM<TProblemElement>(trainSubprob, kernel, C);

                svm.Train();

                for (int j = begin; j < end; j++)
                {
                    //target[perm[j]] = svm_predict(submodel, prob.X[perm[j]]);

                    foldPredictions[indexes[j]] = svm.Predict(problem.Elements[indexes[j]]);
                }
            }

            int correct = 0;
            for (int i = 0; i < probSize; i++)
            {
                if (foldPredictions[i] > -1 && foldPredictions[i] < 1)
                    Debug.WriteLine("foldPredictions[{0}] has strange value={1}", i, foldPredictions[i]);

                if (foldPredictions[i] == problem.Labels[i])
                {
                    ++correct;
                }
            }

            //todo: accuracy count for cross validation
            return (float)correct / probSize;

        }

        public static double CrossValidation<TProblemElement>(
                                                Problem<TProblemElement> problem,
                                                IKernel<TProblemElement> kernel, float C, int nrFolds)
        {
            List<TProblemElement>[] foldsElements;
            List<float>[] foldsLabels;
            int probSize = problem.ElementsCount;
            MakeFoldsSplit(problem, nrFolds, out foldsElements, out foldsLabels);

            double accuracy=CrossValidateOnFolds(probSize, foldsElements, foldsLabels, kernel, C);
            return accuracy;

            //todo: good for paralle computing, think about it later, do validation!
            //long correct = 0l;
            //for (int i = 0; i < nrFolds; i++)
            //{

            //    //array of sub problem elements (comes form permutation)
            //    int subProbSize = 0;
            //    var trainFoldIndexes = from t in Enumerable.Range(0, nrFolds)
            //                           where t != i
            //                           select t;

            //    foreach (var s in trainFoldIndexes)
            //    {
            //        subProbSize += foldsElements[s].Count;
            //    }

            //    float[] subLabels;
            //    TProblemElement[] subProbElem = CreateSubProblem(foldsElements, foldsLabels, trainFoldIndexes, subProbSize, out subLabels);


            //    //create sub problem based on previous subProblemElements and subProbLabels
            //    Problem<TProblemElement> trainSubprob = new Problem<TProblemElement>(subProbElem, subLabels);
            //    kernel.ProblemElements = subProbElem;
            //    CSVM<TProblemElement> svm = new CSVM<TProblemElement>(trainSubprob, kernel, C);

            //    svm.Train();

            //    for (int j = 0; j < foldsElements[i].Count; j++)
            //    {
            //        var element = foldsElements[i][j];

            //        var prediction = svm.Predict(element);
            //        if (prediction == foldsLabels[i][j])
            //        {
            //            ++correct;
            //        }
            //    }
            //}

            ////todo: accuracy count for cross validation
            //return (float)correct / probSize;

        }

        public static void MakeFoldsSplit<TProblemElement>(Problem<TProblemElement> problem, int nrFolds, out List<TProblemElement>[] foldsElements, out List<float>[] foldsLabels)
        {
            int probSize = problem.ElementsCount;

            //array which stores computed permutation of problem element indexes
            int[] permutation = new int[probSize];

            int lastIndexOfFirstClass = GroupSameLabeledElements(problem, permutation);


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

            foldsElements = new List<TProblemElement>[nrFolds];
            foldsLabels = new List<float>[nrFolds];

            for (int i = 0; i < nrFolds; i++)
            {
                foldsElements[i] = new List<TProblemElement>(1 + probSize / nrFolds);
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

                    foldsElements[modIndex].Add(problem.Elements[eleIndex]);
                    foldsLabels[modIndex].Add(problem.Labels[eleIndex]);
                    foldIndex++;
                }
                foldIndex = 0;
            }
            //return probSize;
        }

        public static double CrossValidateOnFolds<TProblemElement>(
                                                int probSize,
                                                List<TProblemElement>[] foldsElements,
                                                List<float>[] foldsLabels,
                                                IKernel<TProblemElement> kernel, float C)
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
                
                CSVM<TProblemElement> svm = new CSVM<TProblemElement>(trainSubprob, kernel, C);

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

        private static TProblemElement[] CreateSubProblem<TProblemElement>(List<TProblemElement>[] foldsElements, List<float>[] foldsLabels, IEnumerable<int> trainFoldIndexes, int subProbSize, out float[] subLabels)
        {
            TProblemElement[] subProbElem = new TProblemElement[subProbSize];

            List<TProblemElement> groupElements = new List<TProblemElement>(subProbSize);
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

        private static void SetFoldsStarts(int probSize, int nrFolds, int[] foldStart)
        {
            int lastFoldStart = -1;
            for (int i = 0; i <= nrFolds; i++)
            {
                int newFoldStart = i * probSize / nrFolds;

                if (lastFoldStart == newFoldStart)
                    newFoldStart += 1;
                else if (lastFoldStart + 2 == newFoldStart)
                {
                    newFoldStart -= 1;
                }
                foldStart[i] = newFoldStart;
                lastFoldStart = foldStart[i];
            }
        }



        /// <summary>
        /// Group elements indexes from Problem into blocks(with same label) of indexes in permuationa array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="problem"></param>
        /// <param name="permutation"></param>
        /// <returns>index of last element from class one</returns>
        private static int GroupSameLabeledElements<T>(Problem<T> problem, int[] permutation)
        {

            int startIndex = -1;
            int endindex = problem.ElementsCount - 1;

            int i = 0;
            while (startIndex < endindex)
            {
                if (problem.Labels[i] <= 0)
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
        /// <param name="trainProblem">Train <see cref="Problem{TProblemElement}">Problem</see>  </param>
        /// <param name="testProblem"></param>
        /// <param name="kernel"></param>
        /// <param name="C"></param>
        /// <returns></returns>
        public static double TestValidation<TProblemElement>(Problem<TProblemElement> trainProblem,
                                                        Problem<TProblemElement> testProblem,
                                                        IKernel<TProblemElement> kernel,
                                                        float C)
        {

            CSVM<TProblemElement> svm = new CSVM<TProblemElement>(trainProblem, kernel, C);
            svm.Init();

            svm.Train();

            int correct = 0;

            Console.WriteLine("Start Predict");
            Stopwatch t = Stopwatch.StartNew();
            for (int i = 0; i < testProblem.ElementsCount; i++)
            {

                float predictedLabel = svm.Predict(testProblem.Elements[i]);

                if (predictedLabel == testProblem.Labels[i])
                    ++correct;

            }
            t.Stop();
            Debug.Write(string.Format("test validation on {0} elements takes {1}",testProblem.ElementsCount,t.Elapsed ));
            return (float)correct / testProblem.ElementsCount;
        }



        /*
                /// <summary>
                /// Randomly split data in test and train 
                /// </summary>
                /// <typeparam name="TProblemElement"></typeparam>
                /// <param name="elements">splitting array</param>
                /// <param name="subElements1">first part of elements = percent*AllElements</param>
                /// <param name="subElements2">second part of elemens rest</param>
                private void Split<TProblemElement>(Problem<TProblemElement> problem,
                                                    out Problem<TProblemElement> subProblem1,
                                                    out Problem<TProblemElement> subProblem2)
                {


                    double percent = 0.3;
                    Random r = new Random();

                    int elementsCount = problem.ElementsCount;

                    int sub1Count = (int)((elementsCount + 0.0) * percent);

                    List<int> subProblem1Indexes = new List<int>(sub1Count);

                    List<TProblemElement> probElements = new List<TProblemElement>(problem.Elements);
                    List<float> probLabels = new List<float>(problem.Labels);

                    List<TProblemElement> sub1List = new List<TProblemElement>(sub1Count);
                    List<float> sub1Labels = new List<float>(problem.Labels);

                    for (int i = 0; i < sub1Count; i++)
                    {
                        //random index
                        int index = r.Next(probElements.Count);
                        //remeber element and its label at position==index
                        TProblemElement el = probElements[index];
                        float label = probLabels[index];
                        //remove element and its label
                        probElements.RemoveAt(index);
                        probLabels.RemoveAt(index);
                        //add to sub1List
                        sub1List.Add(el);
                        sub1Labels.Add(label);
                    }

                    //in allElements list stays elements not choose to sub1List
                    subProblem1 = new Problem<TProblemElement>(sub1List.ToArray(), sub1Labels.ToArray());

                    subProblem2 = new Problem<TProblemElement>(probElements.ToArray(), probLabels.ToArray());

                }

        */


    }
}
