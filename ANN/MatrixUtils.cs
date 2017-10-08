/*
Copyright 2017 Tommy Petersen.

This file is part of "ANN".

"ANN" is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

"ANN" is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with "ANN".  If not, see <http://www.gnu.org/licenses/>. 
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANN
{
    static class MatrixUtils
    {
        public static Matrix hadamardProduct(Matrix P, Matrix Q)
        {
            if (P.M != Q.M || P.N != Q.N)
            {
                throw new Exception("Matrix: Dimension mismatch for hadamardProduct." +
                    " P.M = " + P.M.ToString() + ", Q.M = " + Q.M.ToString() +
                    ", P.N = " + P.N.ToString() + ", Q.N = " + Q.N.ToString());
            }

            Matrix R = new Matrix(P.M, P.N);

            for (int m = 0; m < R.M; m++)
            {
                for (int n = 0; n < R.N; n++)
                {
                    R[m, n] = P[m, n] * Q[m, n];
                }
            }

            return R;
        }
    }
}
