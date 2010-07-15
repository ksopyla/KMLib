using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    internal class AlphaInfo
    {


        /// <summary>
        /// label
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// alpah
        /// </summary>
        public float Alpha { get; set; }

        /// <summary>
        /// computed error
        /// </summary>
        public float Error { get; set; }

        /// <summary>
        /// index for this alpha
        /// </summary>
        public int Index { get; set; }

        public float AlphaStep { get; set; }



        /// <summary>
        /// self product
        /// </summary>
        public float Product { get; set; }

        public AlphaInfo(int k, float alpha1, float y1, float E1, float product)
        {
            Index = k;
            Alpha = alpha1;
            Y = y1;
            Error = E1;
            Product = product;

        }


        public override string ToString()
        {
            return string.Format("index={0} , alpha={1}, y={2}", Index, Alpha, Y);
        }
    }
}
