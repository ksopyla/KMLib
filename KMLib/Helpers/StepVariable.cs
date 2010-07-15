using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib.Helpers
{
    internal class StepPairVariable
    {
        public AlphaInfo First { get; set; }

        public AlphaInfo Second { get; set; }

        #region old fields
        //public float Y2 { get; set; }
        //public float Alpha2 { get; set; }
        //public float Alpha2Step { get; set; }
        //public float E2 { get; set; }
        //public int Index2 { get; set; }



        //public float Y1 { get; set; }
        //public float Alpha1 { get; set; }
        //public float Alpha1Step { get; set; }
        //public float E1 { get; set; }
        //public int Index1 { get; set; }

        #endregion

        public float Product { get; set; }
        public float Si { get; set; }
        public float Eta { get; set; }

        //public StepPairVariable(int index1, int index2,float alpha1, float alpha2,float y1, float y2,float e1, float e2, float product)
        //{
        //    Index1 = index1;
        //    Index2 = index2;

        //    Alpha1 = alpha1;
        //    Alpha2 = alpha2;

        //    Y1 = y1;
        //    Y2 = y2;
        //    E1 = e1;
        //    E2 = e2;

        //    Product = product;

        //    if (index1 == index2)
        //        Si = 1;
        //}
        // public StepPairVariable(int index1, int index2,float alpha1, float alpha2,float y1, float y2,float e1, float e2, float product, float si) :
        //    this(index1, index2, alpha1, alpha2, y1, y2, e1, e2,product)
        //{

        //    Si = si;
        //}


        ///// <summary>
        ///// constructor for one the same pair
        ///// </summary>
        ///// <param name="index1"></param>
        ///// <param name="alpha1"></param>
        ///// <param name="y1"></param>
        ///// <param name="e1"></param>
        ///// <param name="prod"></param>
        // public StepPairVariable(int index1, float alpha1, float y1, float e1, float prod)
        // {
        //     Index1 = Index2 = index1;

        //     Alpha1 = Alpha2 = alpha1;

        //     Y1 = Y2 = y1;

        //     E1 = E2 = e1;

        //     Product = prod;

        //     Si = 1;

        // }



         public StepPairVariable(AlphaInfo st1, AlphaInfo st2,float product, float si, float eta)
         {

             First = st1;
             Second = st2;

             Product = product;
             Si = si;

             Eta = eta;
         }

         public override string ToString()
         {
             return string.Format("eta={0}, product={1}, first={2}, Second={3}", Eta, Product, First.ToString(), Second.ToString());
         }
        
    }
}
