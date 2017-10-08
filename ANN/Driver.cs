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
    class Driver
    {
        static void testMatrix()
        {
            RandomUtils random = new RandomUtils();

            int matrix_P_M = 3;
            int matrix_P_N = 4;

            int matrix_Q_M = 4;
            int matrix_Q_N = 7;

            Matrix matrix_P = newMatrix(matrix_P_M, matrix_P_N, random);
            Matrix matrix_Q = newMatrix(matrix_Q_M, matrix_Q_N, random);
            Matrix matrix_R = (matrix_P * (matrix_Q * (~matrix_Q))) + matrix_P;

            writeMatrix(matrix_P, "matrix_P");
            writeMatrix(matrix_Q, "matrix_Q");
            writeMatrix(matrix_R, "matrix_R");

            Matrix matrix_A = newMatrix(5, 6, random);
            Matrix matrix_B = newMatrix(5, 6, random);
            Matrix matrix_H = MatrixUtils.hadamardProduct(matrix_A, matrix_B);

            writeMatrix(matrix_A, "matrix_A");
            writeMatrix(matrix_B, "matrix_B");
            writeMatrix(matrix_H, "matrix_H");

            Console.ReadLine();
        }

        static void writeMatrix(Matrix P, string header)
        {
            Console.WriteLine("---- " + header + " ----");
            for (int m = 0; m < P.M; m++)
            {
                for (int n = 0; n < P.N; n++)
                {
                    Console.Write(P[m, n].ToString() + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine("--------------");
        }

        static Matrix newMatrix(int _M, int _N, RandomUtils random)
        {
            Matrix matrix = new Matrix(_M, _N);

            for (int m = 0; m < _M; m++)
            {
                for (int n = 0; n < _N; n++)
                {
                    matrix[m, n] = random.NextNormal(0, 1);
                }
            }

            return matrix;
        }

        static void testANN()
        {
            RandomUtils random = new RandomUtils();
            List<int> layerSizes = new List<int>();
            layerSizes.Add(3);
            layerSizes.Add(4);
            layerSizes.Add(2);
            ANN ann = new ANN(layerSizes);
            double eta = 0.01;
            int miniBatchSize = 10;
            int nrOfEpochs = 2;

            //testFeedForward();
            //testSigmoid();
            //testActivation();
            //testFeedForward();
            //testComputeSingletonGradient();
            //testComputeMiniBatchGradient();
            //testAdjustLearningParameters();
            testTrainANN();

            void testActivation()
            {
                Matrix Z = newMatrix(3, 4, random);
                Matrix A = ann.activation(Z);

                writeMatrix(Z, "Z");
                writeMatrix(A, "A");
                Console.ReadLine();
            }

            void testSigmoid()
            {
                double sigmoidRes = ann.sigmoid(-0.5);
                Console.WriteLine("sigmoidRes = " + sigmoidRes.ToString());
                Console.ReadLine();
            }

            void testFeedForward()
            {
                Matrix X = newMatrix(3, 1, random);
                Tuple<List<Matrix>, List<Matrix>> zaLists = ann.feedForward(X);
                List<Matrix> zList = zaLists.Item1;
                List<Matrix> aList = zaLists.Item2;

                Console.WriteLine("zList.Count = " + zList.Count());
                Console.WriteLine("aList.Count = " + aList.Count());

                for (int i = 0; i < zList.Count; i++)
                {
                    if (i > 0)
                    {
                        writeMatrix(zList.ElementAt(i), "zList(" + i.ToString() + ")");
                    }
                    writeMatrix(aList.ElementAt(i), "aList(" + i.ToString() + ")");
                }

                Console.ReadLine();
            }

            void testComputeSingletonGradient()
            {
                Matrix X = newMatrix(3, 1, random);
                Matrix Y = newMatrix(2, 1, random);

                ann.computeSingletonGradient(X, Y);
            }

            void testComputeMiniBatchGradient()
            {
                List<Matrix> X = new List<Matrix>();
                List<Matrix> Y = new List<Matrix>();

                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));

                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));

                Tuple<List<Matrix>, List<Matrix>> wbGradientList =
                    ann.computeMiniBatchGradient(X, Y);
            }

            void testAdjustLearningParameters()
            {
                List<Matrix> X = new List<Matrix>();
                List<Matrix> Y = new List<Matrix>();

                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));
                X.Add(newMatrix(3, 1, random));

                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));
                Y.Add(newMatrix(2, 1, random));

                Tuple<List<Matrix>, List<Matrix>> wbGradientList =
                    ann.computeMiniBatchGradient(X, Y);

                ann.adjustLearningParameters(eta, wbGradientList.Item1, wbGradientList.Item2);
            }

            void testTrainANN()
            {
                List<Matrix> trainX = new List<Matrix>();
                List<Matrix> trainY = new List<Matrix>();

                int inputSize = layerSizes.First<int>();
                int outputSize = layerSizes.Last<int>();

                int nrOfTrainExamples = 500;

                for (int i = 0; i < nrOfTrainExamples; i++)
                {
                    trainX.Add(newMatrix(inputSize, 1, random));
                    trainY.Add(newMatrix(outputSize, 1, random));
                }

                ann.trainANN(trainX, trainY, miniBatchSize, eta, nrOfEpochs);
            }
        }

        static void testListUtils()
        {
            List<int> iList = new List<int>()
            {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            };

            List<int> rList = ListUtils.permute(iList);
        }

        static void Main(string[] args)
        {
            testANN();
        }
    }
}
