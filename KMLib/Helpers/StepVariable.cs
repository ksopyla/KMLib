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

        public float Product { get; set; }
        public float Si { get; set; }
        public float Eta { get; set; }

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
