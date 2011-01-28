using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    public static class ListHelper
    {

        /// <summary>
        /// Extensios method for swapping indexes in array
        /// </summary>
        /// <typeparam name="T">Array element type</typeparam>
        /// <param name="array">array</param>
        /// <param name="i">first index</param>
        /// <param name="j">second index</param>
        public static void SwapIndex<T>(this T[] array, int i, int j)
        {
            T tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }


        /// <summary>
        /// Extension method for comping array to new with differnet size array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="newLenght"></param>
        /// <returns></returns>
        public static T[] CopyToNewArray<T>(this T[] source, int newLenght)
        {
            T[] newArray = new T[newLenght];
            Array.Copy(source, newArray, Math.Min(newLenght, source.Length));
            return newArray;

        }
    }
}
