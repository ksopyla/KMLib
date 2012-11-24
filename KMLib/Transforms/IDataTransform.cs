using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Transforms
{
    public interface IDataTransform<T>
    {

        T[] Transform(T[] input);

        T Transform(T input);
    }
}
