using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
        public static double CrossValidation<TProblemElement>(
                                                Problem<TProblemElement> problem,
                                                IKernel<TProblemElement> kernel, float C, int nrFolds)
        {
            int probSize = problem.ElementsCount;

            //array for all folds predictions
            float[] foldPredictions = new float[probSize];

            //array which stores computed permutation of problem element indexes
            int[] permutation = new int[probSize];

            Random rand = new Random();
            //stores information of which indexes are for each fold
            int[] foldStart = new int[nrFolds + 1];

            //initialize permutation array
            for (int i = 0; i < probSize; i++)
                permutation[i] = i;

            //compute permutation, swap random pair indexes
            for (int i = 0; i < probSize; i++)
            {
                //method for permuation without repetition
                int j = i + (int) (rand.NextDouble()*(probSize - i));

                //change new index j with current i
                int temp = permutation[i];
                permutation[i] = permutation[j];
                permutation[j] = temp;
            }

            for (int i = 0; i <= nrFolds; i++)
                foldStart[i] = i*probSize/nrFolds;

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
                    subProbElem[k] = problem.Elements[permutation[j]];
                    subLabels[k] = problem.Labels[permutation[j]];
                    ++k;
                }
                for (int j = end; j < probSize; j++)
                {
                    subProbElem[k] = problem.Elements[permutation[j]];
                    subLabels[k] = problem.Labels[permutation[j]];
                    ++k;
                }
                //create sub problem based on previous subProblemElements and subProbLabels
                Problem<TProblemElement> trainSubprob = new Problem<TProblemElement>(subProbElem, subLabels);

                CSVM<TProblemElement> svm = new CSVM<TProblemElement>(trainSubprob, kernel, C);

                svm.Train();

                for (int j = begin; j < end; j++)
                {
                    //target[perm[j]] = svm_predict(submodel, prob.X[perm[j]]);

                    foldPredictions[permutation[j]] = svm.Predict(problem.Elements[permutation[j]]);
                }
            }

            int correct=0;
            for (int i = 0; i < probSize; i++)
            {
                if (foldPredictions[i] == problem.Labels[i])
                {
                    
                    ++correct;
                }
            }

            //todo: accuracy count for cros validation
            return (float)correct / probSize;

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
        public static double  TestValidation<TProblemElement>(Problem<TProblemElement> trainProblem, 
                                                        Problem<TProblemElement> testProblem,
                                                        IKernel<TProblemElement> kernel,
                                                        float  C)
        {

            CSVM<TProblemElement> svm = new CSVM<TProblemElement>(trainProblem,kernel,C);

            svm.Train();

            int correct = 0;
            for (int i = 0; i < testProblem.ElementsCount; i++)
            {

                float predictedLabel = svm.Predict(testProblem.Elements[i]);

                if(predictedLabel==testProblem.Labels[i])
                    ++correct;
                
            }

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
