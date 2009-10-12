using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace KMLib
{
    public abstract class Solver<TProblemElement>
    {
        protected Problem<TProblemElement> problem;
        protected float C;
        protected IKernel<TProblemElement> kernel;


        public Solver(Problem<TProblemElement> problem, IKernel<TProblemElement> kernel, float C)
        {
            this.problem = problem;
            this.kernel = kernel;
            this.C = C;
        }

        public abstract Model<TProblemElement> ComputeModel();

    }
}