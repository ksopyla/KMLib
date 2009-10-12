using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.DataStructers
{
    /// <summary>
    /// Sparse Vector implementation, holds array of Nodes(index and Value)
    ///  first index is 1 (not zero)
    /// </summary>
    /// <remarks>currently not used</remarks>
    public class Vector
    {

        //todo: maybe it should be a struct
        /// <summary>
        /// Node containing vector index and value on index position
        /// 
        /// </summary>
        public class Node : IComparable<Node>
        {
            public int Index;
            public float Value;

            public Node(int index, float val)
            {
                Index = index;
                Value = val;
            }

            #region IComparable<Node> Members

            /// <summary>
            /// Nesessary for Nodes sorting by index
            /// </summary>
            /// <param name="other"></param>
            /// <returns></returns>
            public int CompareTo(Node other)
            {
                return this.Index.CompareTo(other.Index);
            }

            #endregion
        }

        public Node[] Data;

        /// <summary>
        /// 
        /// </summary>
        public int Dimension;

     
        /// <summary>
        /// creates empty(zero) vector with specific dimension
        /// </summary>
        /// <param name="dimension">dimension</param>
        public Vector(int dimension)
        {
            Dimension = dimension;
        }


        /// <summary>
        /// Construct vector with specified Node List, dimension not set
        /// because last index could be eg. 100 but vectos has 200 dim. and last 100 is zero
        /// so user should set Dimension property by himself
        /// </summary>
        /// <param name="nodes"></param>
        public Vector(List<Node> nodes)
        {
            //sort nodes by index see. Node.ComperTo method
            nodes.Sort();

            Data = nodes.ToArray();

        }
        /// <summary>
        /// Constructor for dense vectors, 
        /// </summary>
        /// <param name="values"></param>
        public Vector(float[] values)
        {
            /*
            for (int i = 0; i < values.Length; i++)
            {
                this[i+1] = values[i];
            }
             */
            Dimension = values.Length;
            Data = new Node[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                Data[i] = new Node(i + 1, (float)values[i]);
            }
        }


    }
}