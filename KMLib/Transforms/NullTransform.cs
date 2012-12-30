using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KMLib.Helpers;

namespace KMLib.Transforms
{

    /// <summary>
    /// Does nothing, just NULL object
    /// </summary>
    public class NullTransform: IDataTransform<SparseVec>
    {
        public SparseVec[] Transform(SparseVec[] input)
        {
            return input;
        }

        public SparseVec Transform(SparseVec input)
        {
            return input;
        }
    }
}
