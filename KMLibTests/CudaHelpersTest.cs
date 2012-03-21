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
        public void TransformToSliceEllpackTest()
        {
            float[] vecVals = null; // TODO: Initialize to an appropriate value
            int[] vecIdx = null; // TODO: Initialize to an appropriate value
            int[] vecLenght = null; // TODO: Initialize to an appropriate value
            int[] sliceStart=null;

            int[] vecIdxExpected = new int[] { 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3 };
            float[] vecValsExpected = new float[] { 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3 };
            int[] vecLenghtExpected = new int[] { 0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25 };

            CudaHelpers.TransformToSlicedEllpack(out vecVals, out vecIdx, out sliceStart,out vecLenght, problemElements,2,2);

            Assert.Fail("not implemented yet");
           
        }
    }
}
