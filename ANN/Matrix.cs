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
    public class Matrix
    {
        /// <summary>
        /// Represents an M x N matrix.
        /// </summary>
        /// <param name="args"></param>

        private double[,] A;
        public int M { get; }
        public int N { get; }

        public static Matrix operator ~(Matrix P)
        {
            Matrix R = new Matrix(P.N, P.M);

            for (int n = 0; n < P.N; n++)
            {
                for (int m = 0; m < P.M; m++)
                {
                    R[n, m] = P[m, n];
                }
            }

            return R;
        }

        public static Matrix operator *(Matrix P, double d)
        {
            Matrix R = new Matrix(P.M, P.N);

            for (int m = 0; m < R.M; m++)
            {
                for (int n = 0; n < R.N; n++)
                {
                    R[m, n] = P[m, n] * d;
                }
            }

            return R;
        }

        public static Matrix operator *(Matrix P, Matrix Q)
        {
            if (P.N != Q.M)
            {
                throw new Exception("Matrix: Dimension mismatch for operator *." +
                    " P.N = " + P.N.ToString() + ", Q.M = " + Q.M.ToString());
            }

            Matrix R = new Matrix(P.M, Q.N);
            double dot_product = 0D;

            for (int m = 0; m < P.M; m++)
            {
                for (int n = 0; n < Q.N; n++)
                {
                    dot_product = 0D;

                    for (int j = 0; j < P.N; j++)
                    {
                        dot_product = dot_product + (P[m, j] * Q[j, n]);
                    }

                    R[m, n] = dot_product;
                }
            }

            return R;
        }

        public static Matrix operator -(Matrix P, Matrix Q)
        {
            if (P.M != Q.M || P.N != Q.N)
            {
                throw new Exception("Matrix: Dimension mismatch for operator -." +
                    " P.M = " + P.M.ToString() + ", Q.M = " + Q.M.ToString() +
                    ", P.N = " + P.N.ToString() + ", Q.N = " + Q.N.ToString());
            }

            Matrix R = new Matrix(P.M, P.N);

            for (int m = 0; m < P.M; m++)
            {
                for (int n = 0; n < P.N; n++)
                {
                    R[m, n] = P[m, n] - Q[m, n];
                }
            }

            return R;
        }

        public static Matrix operator +(Matrix P, Matrix Q)
        {
            if (P.M != Q.M || P.N != Q.N)
            {
                throw new Exception("Matrix: Dimension mismatch for operator +." +
                    " P.M = " + P.M.ToString() + ", Q.M = " + Q.M.ToString() +
                    ", P.N = " + P.N.ToString() + ", Q.N = " + Q.N.ToString());
            }

            Matrix R = new Matrix(P.M, P.N);

            for (int m = 0; m < P.M; m++)
            {
                for (int n = 0; n < P.N; n++)
                {
                    R[m, n] = P[m, n] + Q[m, n];
                }
            }

            return R;
        }

        public double this[int m, int n]
        {
            get {
                if (m < 0 || m >= M || n < 0 || n >= N)
                {
                    throw new Exception("Matrix: Invalid index arguments for get." +
                        " m = " + m.ToString() + ", n = " + n.ToString());
                }

                return A[m, n];
            }

            set {
                if (m < 0 || m >= M || n < 0 || n >= N)
                {
                    throw new Exception("Matrix: Invalid index arguments for set." +
                        " m = " + m.ToString() + ", n = " + n.ToString());
                }

                A[m, n] = value;
            }
        }

        public override string ToString()
        {
            String s = "";

            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    s = s + A[m, n].ToString();
                }
                s = s + "\n";
            }

            return s;
        }

        public Matrix(int _M, int _N)
        {
            if (_M <= 0 || _N <= 0)
            {
                throw new Exception("Matrix: Invalid constructor argument." +
                    " _M = " + _M.ToString() + ", _N = " + _N.ToString());
            }

            A = new double[_M, _N];
            M = _M;
            N = _N;

            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    A[m, n] = 0;
                }
            }
        }
    }
}
