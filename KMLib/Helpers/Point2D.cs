using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{

    /// <summary>
    /// Point in 2D space
    /// </summary>
    public struct Point2D
    {
        /// <summary>
        /// X coridnate
        /// </summary>
        public int X;


        /// <summary>
        /// Y cordinate
        /// </summary>
        public int Y;

        public Point2D(int x, int y)
        {
            X = x;
            Y = y;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            //this is faster           
            return X.GetHashCode()*29 + Y.GetHashCode();

            //return X.GetHashCode() ^ Y.GetHashCode();
        }

    }
}
