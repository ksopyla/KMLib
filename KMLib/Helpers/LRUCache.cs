using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{

    /// <summary>
    /// Provides cache for kernel computing
    /// </summary>
    public class LRUCache<TKey, TValue> 
    {
        int m_iMaxItems;
        Dictionary<TKey, LRUNode<TKey, TValue>> m_oMainDict;

        private LRUNode<TKey, TValue> m_oHead;
        private LRUNode<TKey, TValue> m_oTail;
        private LRUNode<TKey, TValue> m_oCurrent;

        public LRUCache(int iSize)
        {
            m_iMaxItems = iSize;
            m_oMainDict = new Dictionary<TKey, LRUNode<TKey, TValue>>();

            m_oHead = null;
            m_oTail = null;
        }

        public TValue this[TKey key]
        {
            get
            {
                m_oCurrent = m_oMainDict[key];

                if (m_oCurrent == m_oHead)
                {
                    //do nothing
                }
                else if (m_oCurrent == m_oTail)
                {
                    m_oTail = m_oCurrent.Next;
                    m_oTail.Prev = null;

                    m_oHead.Next = m_oCurrent;
                    m_oCurrent.Prev = m_oHead;
                    m_oCurrent.Next = null;
                    m_oHead = m_oCurrent;
                }
                else
                {
                    m_oCurrent.Prev.Next = m_oCurrent.Next;
                    m_oCurrent.Next.Prev = m_oCurrent.Prev;

                    m_oHead.Next = m_oCurrent;
                    m_oCurrent.Prev = m_oHead;
                    m_oCurrent.Next = null;
                    m_oHead = m_oCurrent;
                }

                return m_oCurrent.Value;
            }
        }

        public void Add(TKey key, TValue value)
        {
            if (m_oMainDict.Count >= m_iMaxItems)
            {
                //remove old
                m_oMainDict.Remove(m_oTail.Key);

                //reuse old
                LRUNode<TKey, TValue> oNewNode = m_oTail;
                oNewNode.Key = key;
                oNewNode.Value = value;

                m_oTail = m_oTail.Next;
                m_oTail.Prev = null;

                //add new
                m_oHead.Next = oNewNode;
                oNewNode.Prev = m_oHead;
                oNewNode.Next = null;
                m_oHead = oNewNode;
                m_oMainDict.Add(key, oNewNode);
            }
            else
            {
                LRUNode<TKey, TValue> oNewNode = new LRUNode<TKey, TValue>(key, value);
                if (m_oHead == null)
                {
                    m_oHead = oNewNode;
                    m_oTail = oNewNode;
                }
                else
                {
                    m_oHead.Next = oNewNode;
                    oNewNode.Prev = m_oHead;
                    m_oHead = oNewNode;
                }
                m_oMainDict.Add(key, oNewNode);
            }
        }

        public bool ContainsKey(TKey key)
        {
            return m_oMainDict.ContainsKey(key);
        }
    }


    internal class LRUNode<TKey, TValue>
    {
        public LRUNode(TKey key, TValue val)
        {
            Key = key;
            Value = val;
        }

        public TKey Key;
        public TValue Value;
        public LRUNode<TKey, TValue> Next;
        public LRUNode<TKey, TValue> Prev;
    }
}
