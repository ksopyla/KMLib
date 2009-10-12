using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using KMLib.DataStructers;
using dnaLA = dnAnalytics.LinearAlgebra;

namespace KMLib.Helpers
{
    public class IOHelper
    {
        /// <summary>
        /// Reads vectos from file.
        /// File format like in LIbSVM
        /// label index:value index2:value .....
        /// -1        4:0.5       10:0.9 ....
        /// </summary>
        /// <param name="fileName">Data set file name</param>
        /// <returns></returns>
        public static Problem<Vector> ReadVectorsFromFile(string fileName)
        {



            List<float> labels = new List<float>();

            List<Vector> vectors = new List<Vector>();

            using (FileStream fileStream = File.OpenRead(fileName))
            {
                using (StreamReader input = new StreamReader(fileStream))
                {

                    int max_index = 0;

                    while (input.Peek() > -1)
                    {
                        string[] parts = input.ReadLine().Trim().Split();

                        //label
                        labels.Add(float.Parse(parts[0]));

                        //other parts with index and value
                        int m = parts.Length - 1;


                        List<Vector.Node> nodes = new List<Vector.Node>();


                        int index = 0;
                        float value;
                        for (int j = 0; j < m; j++)
                        {

                            string[] nodeParts = parts[j + 1].Split(':');
                            index = int.Parse(nodeParts[0]);
                            value = float.Parse(nodeParts[1], System.Globalization.CultureInfo.InvariantCulture);

                            nodes.Add(new Vector.Node(index, (float)value));
                            // v[index] = value;

                        }

                        if (m > 0)
                        {
                            max_index = Math.Max(max_index, index);
                        }
                        vectors.Add(new Vector(nodes));
                    }

                    //assing max index as  vector Dimension,
                    //not always true for different data sets
                    for (int i = 0; i < vectors.Count; i++)
                    {
                        vectors[i].Dimension = max_index;
                    }

                }
            }
            /*
            Problem<Vector> vectorProblem = new Problem<Vector>(vectors.ToArray(),labels.ToArray());
            
            
            vectorProblem.Elements = vectors.ToArray();
            vectorProblem.ElementsCount = vectors.Count;
            vectorProblem.Labels = labels.ToArray();
            
            return vectorProblem;
            */
            return new Problem<Vector>(vectors.ToArray(), labels.ToArray());
        }


        /// <summary>
        /// Reads vectos from file. Creates Vectors from dnAnalitycs library.
        /// File format like in LIbSVM
        /// label index:value index2:value .....
        /// -1        4:0.5       10:0.9 ....
        /// </summary>
        /// <param name="fileName">Data set file name</param>
        /// <returns></returns>
        public static Problem<dnaLA.Vector> ReadDNAVectorsFromFile(string fileName,int numberOfFeatures)
        {


            //list of labels
            List<float> labels = new List<float>();




            //list of array, each array symbolize vector
            List<KeyValuePair<int, double>[]> vectors = new List<KeyValuePair<int, double>[]>();
            //new List<List<KeyValuePair<int, double>>>();

            int max_index = 0;

            using (FileStream fileStream = File.OpenRead(fileName))
            {
                using (StreamReader input = new StreamReader(fileStream))
                {



                    while (input.Peek() > -1)
                    {
                        string[] parts = input.ReadLine().Trim().Split();

                        //label
                        labels.Add(float.Parse(parts[0]));

                        //other parts with index and value
                        int m = parts.Length - 1;

                        //list of index and value for one vector
                        List<KeyValuePair<int, double>> vec = new List<KeyValuePair<int, double>>();


                        int index = 0;
                        float value;
                        //extract index and value
                        for (int j = 0; j < m; j++)
                        {

                            string[] nodeParts = parts[j + 1].Split(':');
                            index = int.Parse(nodeParts[0]);
                            value = float.Parse(nodeParts[1], System.Globalization.CultureInfo.InvariantCulture);

                            vec.Add(new KeyValuePair<int, double>(index, value));
                            // v[index] = value;

                        }

                        if (m > 0)
                        {
                            max_index = Math.Max(max_index, index);
                        }


                        //we implictie set numberOfFeatures if max_index is less then numberOfFeatures
                        if (max_index < numberOfFeatures)
                            max_index = numberOfFeatures;


                        vectors.Add(vec.ToArray());
                    }//end while
                }
            }

            //list of Vectors, currently use SparseVector implementation from dnAnalitycs
            List<dnaLA.Vector> dnaVectors = new List<dnaLA.Vector>(vectors.Count);
            //assing max index as  vector Dimension,
            //not always true for different data sets
            for (int i = 0; i < vectors.Count; i++)
            {
                dnaLA.Vector oneVec = new dnaLA.SparseVector(max_index+1);
                
                for (int j = 0; j < vectors[i].Length; j++)
                {
                    int index = vectors[i][j].Key;
                    double val = vectors[i][j].Value;
                    oneVec[index] = val;
                }

                dnaVectors.Add(oneVec);

            }



            return new Problem<dnaLA.Vector>(dnaVectors.ToArray(), labels.ToArray());
        }
    }
}
