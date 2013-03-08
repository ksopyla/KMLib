using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    internal class ByAscValueComparer: IComparer<KeyValuePair<int,float>>
    {
        #region IComparer<KeyValuePair<int,float>> Members

        public int Compare(KeyValuePair<int, float> x, KeyValuePair<int, float> y)
        {
            return x.Value.CompareTo(y.Value);

           // return y.Value.CompareTo(x.Value);
        }

        #endregion
    }


    internal class ByDescValueComparer : IComparer<KeyValuePair<int, float>>
    {
        #region IComparer<KeyValuePair<int,float>> Members

        public int Compare(KeyValuePair<int, float> x, KeyValuePair<int, float> y)
        {
            return y.Value.CompareTo(x.Value);
        }

        #endregion
    }
}
