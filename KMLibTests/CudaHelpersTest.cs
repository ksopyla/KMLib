using KMLib.GPU;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using KMLib.Helpers;
using System.Collections.Generic;

namespace KMLibTests
{


    /// <summary>
    ///This is a test class for CudaHelpersTest and is intended
    ///to contain all CudaHelpersTest Unit Tests
    ///</summary>
    [TestClass()]
    public class CudaHelpersTest
    {
        static SparseVec[] problemElements = null; // TODO: Initialize to an appropriate value

        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        #region Additional test attributes
        // 
        //You can use the following additional attributes as you write your tests:
        //
        //Use ClassInitialize to run code before running the first test in the class
        [ClassInitialize()]
        public static void Initialize(TestContext testContext)
        {

            List<SparseVec> elements = new List<SparseVec>(5);
            int dim = 11;

            for (int i = 0; i < 5; i++)
            {

                IList<KeyValuePair<int, float>> idxVal = new List<KeyValuePair<int, float>>();
                for (int j = 1; j < dim; j++)
                {
                    if ((i + j+1) % 2 == 0)
                    {
                        idxVal.Add(new KeyValuePair<int, float>(j,i ));
                    }
                }

                elements.Add(new SparseVec(dim, idxVal));
            }

            problemElements = elements.ToArray();

        }
        //
        //Use ClassCleanup to run code after all tests in a class have run
        //[ClassCleanup()]
        //public static void MyClassCleanup()
        //{
        //}
        //
        //Use TestInitialize to run code before running each test
        //[TestInitialize()]
        //public void MyTestInitialize()
        //{
        //}
        //
        //Use TestCleanup to run code after each test has run
        //[TestCleanup()]
        //public void MyTestCleanup()
        //{
        //}
        //
        #endregion


        /// <summary>
        ///A test for TransformToCSCFormat
        ///</summary>
        [TestMethod()]
        public void TransformToCSCFormatTest()
        {
            float[] vecVals = null; // TODO: Initialize to an appropriate value
            int[] vecIdx = null; // TODO: Initialize to an appropriate value
            int[] vecLenght = null; // TODO: Initialize to an appropriate value


            int[] vecIdxExpected = new int[] { 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3 };
            float[] vecValsExpected = new float[] { 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3 };
            int[] vecLenghtExpected = new int[] { 0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25 };

            CudaHelpers.TransformToCSCFormat(out vecVals, out vecIdx, out vecLenght, problemElements);
           
            for (int i = 0; i < vecIdxExpected.Length; i++)
            {
                Assert.AreEqual(vecIdxExpected[i], vecIdx[i], "vec idx different at position" + i);
            }

            for (int i = 0; i < vecValsExpected.Length; i++)
            {
                Assert.AreEqual(vecValsExpected[i], vecVals[i], "vec val different at position" + i);
            }

            for (int i = 0; i < vecLenghtExpected.Length; i++)
            {
                Assert.AreEqual(vecLenghtExpected[i], vecLenght[i], "vec lenght different at position" + i);
            }

            
        }

        /// <summary>
        ///A test for TransformToCSRFormat
        ///</summary>
        [TestMethod()]
        public void TransformToCSRFormatTest()
        {
            float[] vecVals = null; // TODO: Initialize to an appropriate value
            float[] vecValsExpected = null; // TODO: Initialize to an appropriate value
            int[] vecIdx = null; // TODO: Initialize to an appropriate value
            int[] vecIdxExpected = null; // TODO: Initialize to an appropriate value
            int[] vecLenght = null; // TODO: Initialize to an appropriate value
            int[] vecLenghtExpected = null; // TODO: Initialize to an appropriate value
            SparseVec[] problemElements = null; // TODO: Initialize to an appropriate value
            CudaHelpers.TransformToCSRFormat(out vecVals, out vecIdx, out vecLenght, problemElements);
            Assert.AreEqual(vecValsExpected, vecVals);
            Assert.AreEqual(vecIdxExpected, vecIdx);
            Assert.AreEqual(vecLenghtExpected, vecLenght);
            Assert.Inconclusive("A method that does not return a value cannot be verified.");
        }

        [TestMethod()]
        public void TransformToSliceEllpackTest_2threads_2slice_5vectors()
        {
            float[] vecVals = null; // TODO: Initialize to an appropriate value
            int[] vecIdx = null; // TODO: Initialize to an appropriate value
            int[] vecLenght = null; // TODO: Initialize to an appropriate value
            int[] sliceStart=null;

            int[] vecIdxExpected = new int[] { 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3 };
            float[] vecValsExpected = new float[] { 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3 };
            int[] vecLenghtExpected = new int[] { 0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25 };
            int[] vecsliceExpected = new int[] { 0, 12, 24, 36 };

            CudaHelpers.TransformToSlicedEllpack(out vecVals, out vecIdx, out sliceStart,out vecLenght, problemElements,2,2);

            for (int i = 0; i < vecIdxExpected.Length; i++)
            {
                Assert.AreEqual(vecIdxExpected[i], vecIdx[i], "vec idx different at position" + i);
            }

            for (int i = 0; i < vecValsExpected.Length; i++)
            {
                Assert.AreEqual(vecValsExpected[i], vecVals[i], "vec val different at position" + i);
            }

            for (int i = 0; i < vecLenghtExpected.Length; i++)
            {
                Assert.AreEqual(vecLenghtExpected[i], vecLenght[i], "vec lenght different at position" + i);
            }

            for (int i = 0; i < vecsliceExpected.Length; i++)
            {
                Assert.AreEqual(vecsliceExpected[i], sliceStart[i], "vec slice different at position" + i);
            }
           
        }


        [TestMethod()]
        public void TransformToSliceEllpackTest_different_slice_sizes()
        {

            int sliceSize = 4;
            int threads = 2;
            var problem = GenerateTestProblemForSliceEllpack(13,sliceSize,15,new int[] { 3, 6, 5, 4 });
            float[] vecVals = null; // TODO: Initialize to an appropriate value
            int[] vecIdx = null; // TODO: Initialize to an appropriate value
            int[] vecLenght = null; // TODO: Initialize to an appropriate value
            int[] sliceStart = null;

            int[] vecIdxExpected = new int[] {3,5,4,6,5,7,6,8,7,0,8,0,9,0,10,0 };
            float[] vecValsExpected = new float[] { 1,1,2,2,3,3,4,4,1,0,2,0,3,0,4,0,5,5,6,6,7,7,8,8,5,5,6,6,7,7,8,8,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,9,9,10,10,11,11,12,12,9,0,10,0,11,0,12,0,13,13,0,0,0,0,0,0,13,13,0,0,0,0,0,0 };
            int[] vecLenghtExpected = new int[] { 3,3,3,3,6,6,6,6,5,5,5,5,4 };
            int[] sliceStarExpected = new int[] { 0, 16, 40, 64, 80 };

            CudaHelpers.TransformToSlicedEllpack(out vecVals, out vecIdx, out sliceStart, out vecLenght, problem, threads ,sliceSize);

            for (int i = 0; i < sliceStarExpected.Length; i++)
            {
                Assert.AreEqual(sliceStarExpected[i], sliceStart[i], "vec slice different at position" + i);
            }

            for (int i = 0; i < vecLenghtExpected.Length; i++)
            {
                Assert.AreEqual(vecLenghtExpected[i], vecLenght[i], "vec length different at position" + i);
            }


            for (int i = 0; i < vecValsExpected.Length; i++)
            {
                Assert.AreEqual(vecValsExpected[i], vecVals[i], "vec val different at position" + i);
            }

            for (int i = 0; i < vecIdxExpected.Length; i++)
            {
                Assert.AreEqual(vecIdxExpected[i], vecIdx[i], "vec idx different at position" + i);
            }

            

        }

        private SparseVec[] GenerateTestProblemForSliceEllpack(int rows, int sliceSize, int dim, int[] maxSliceSizes)
        {

            //int rows = 13;
            //int sliceSize = 4;
            //int thread = 2;
            //int[] maxSliceSizes = new int[] { 3, 6, 5, 4 };
            //int dim = 15;
            
            List<SparseVec> elements = new List<SparseVec>(rows);
            

            int sliceNr = 0;
            int rowInSlice = 0;
            for (int i = 0; i < rows; i++)
            {
                sliceNr = i / sliceSize;
                rowInSlice = i % sliceSize;

                List<KeyValuePair<int, float>> idxVal = new List<KeyValuePair<int, float>>();
                for (int j = 1; j <= maxSliceSizes[sliceNr]; j++)
                {
                        idxVal.Add(new KeyValuePair<int, float>( (j*2+i)%dim+1 , i+1));
                }

                //idxVal.Add(new KeyValuePair<int, float>(1, 100));
                
                
                idxVal.Sort((a, b) => a.Key.CompareTo(b.Key));

                elements.Add(new SparseVec(dim, idxVal));
            }

            return elements.ToArray();
        }
    }
}
