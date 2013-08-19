/*
author: Krzysztof Sopyla
mail: krzysztofsopyla@gmail.com
License: MIT
web page: http://wmii.uwm.edu.pl/~ksopyla/projects/svm-net-with-cuda-kmlib/
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.GPU
{
   public static class CUDAConfig
    {

        /// <summary>
        /// The size of block in x-axis, default value 128
        /// </summary>
        /// <remarks>It is also used for one dimensional cuda block</remarks>
       public const int XBlockSize =256;


       /// <summary>
       /// The size of block in y-axis
       /// </summary>
       public const int YBlockSize = 128;
    }
}
