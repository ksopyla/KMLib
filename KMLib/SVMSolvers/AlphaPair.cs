using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.SVMSolvers
{
    internal class AlphaPair
    {
        public int FirstIndex { get; set; }

        public int SecondIndex { get; set; }

        public float FirstAlpha { get; set; }

        public float SecondAlpha { get; set; }

        public float Threshold { get; set; }
    }
}
