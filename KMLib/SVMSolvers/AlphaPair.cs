using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    internal class AlphaPair
    {
        public int FirstIndex { get; set; }

        public int SecondIndex { get; set; }

        public float FirstAlpha { get; set; }

        public float SecondAlpha { get; set; }

        public float Threshold { get; set; }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendFormat("Index1={0} Alpha1={1}; Index2={2} Alpha2={3}; Rho={4}", FirstIndex, FirstAlpha, SecondIndex, SecondAlpha, Threshold);
            
            return sb.ToString();
        }
    }
}
