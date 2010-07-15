using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    internal class Pair<T, U>
    {
        public T First { get; set; }
        public U Second { get; set; }

        public Pair(T first, U second)
        {
            First = first;
            Second = second;
        }

        public override string ToString()
        {
            return string.Format("It1={0} It2={1}", First.ToString(), Second.ToString());
        }
    }
}
