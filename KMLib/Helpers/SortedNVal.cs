using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    /// <summary>
    /// represents N sorted value in specific order (ascending, descending)
    /// </summary>
    public class SortedNVal
    {
        public enum SortMode
        {
            Asc,
            Desc
        }

        

        LinkedList<KeyValuePair<int, float>> sortedList;

        public int Count { get { return sortedList.Count; } }

        /// <summary>
        /// maximal stored items
        /// </summary>
        int nSize;
        float Min = float.MaxValue;

        float Max = float.MinValue;

        SortMode mode = SortMode.Asc;

        public SortedNVal(int size)
        {
            nSize = size;
            sortedList = new LinkedList<KeyValuePair<int, float>>();

            //for (int i = 0; i < nSize; i++)
            //{
            //    sortedList.AddFirst(new KeyValuePair<int,float>(int.MinValue,float.MinValue);
            //}
        }

        public SortedNVal(int size, SortMode mod)
            : this(size)
        {
            mode = mod;
        }



        public bool Add(int index, float value)
        {
            //todo: change the way it stores only one or two elements
            if (mode == SortMode.Asc)
            {
                return AddAsc(new KeyValuePair<int, float>(index, value));
            }
            else
            {
                return AddDesc(new KeyValuePair<int, float>(index, value));
            }


        }

        private bool AddDesc(KeyValuePair<int, float> pair)
        {
            float val = pair.Value;

            if (pair.Value <= Min && sortedList.Count >= nSize)
                return false;

            if (pair.Value > Max)
            {
                sortedList.AddFirst(pair);

                Max = pair.Value;
            }
            else
            {
                //find place for element
                //LinkedListNode<KeyValuePair<int,float>> toInsert = new LinkedListNode<KeyValuePair<int,float>>(pair);

                var start = sortedList.First.Next;
                var end = sortedList.Last;

                if (start == end)
                {
                    sortedList.AddBefore(start, pair);
                }
                else
                {

                    for (var i = start; i != end; i = i.Next)
                    {

                        if (i.Value.Value < pair.Value)
                        {
                            sortedList.AddBefore(i, pair);
                            break;
                        }
                    }

                }
            }

            if (sortedList.Count > nSize)
                sortedList.RemoveLast();

            Min = sortedList.Last.Value.Value;

            return true;
        }



        private bool AddAsc(KeyValuePair<int, float> pair)
        {
            float val = pair.Value;

            if (val >= Max && sortedList.Count >= nSize)
                return false;

            if (val < Min)
            {
                sortedList.AddFirst(pair);

                Min = val;
            }
            else
            {
                if (sortedList.Count == 0)
                    sortedList.AddFirst(pair);
                else
                {

                    var start = sortedList.First.Next;
                    if (start == null)
                    {
                        sortedList.AddLast(pair);
                    }
                    //else if (start == end)
                    //{
                    //    sortedList.AddBefore(start, pair);
                    //}
                    else
                    {

                        for (var i = start; i != null; i = i.Next)
                        {

                            if (i.Value.Value >= val)
                            {
                                sortedList.AddBefore(i, pair);
                                break;
                            }
                        }

                    }
                }
            }

            if (sortedList.Count > nSize)
                sortedList.RemoveLast();

            Max = sortedList.Last.Value.Value;

            return true;

        }



        public KeyValuePair<int, float>[] ToArray()
        {
            return sortedList.ToArray();
        }
    }
}
