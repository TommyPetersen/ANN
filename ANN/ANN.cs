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
    public class ANN
    {
        RandomUtils random = new RandomUtils();
        List<int> layerSizes = new List<int>();
        List<Matrix> wList = new List<Matrix>();
        List<Matrix> bList = new List<Matrix>();
        List<Matrix> trainX = null;
        List<Matrix> trainY = null;

        public void trainANN(
            List<Matrix> _trainX, 
            List<Matrix> _trainY, 
            int miniBatchSize, 
            double eta, 
            int nrOfEpochs)
        {
            if (_trainX == null || _trainY == null)
            {
                string nullMSG = _trainX == null ? "_trainX is null" : "_trainY is null";
                throw new Exception("ANN: Invalid training data in trainANN: " + nullMSG);
            }

            if (_trainX.First<Matrix>().M != layerSizes.First<int>())
            {
                throw new Exception("ANN.trainANN: Invalid input dimension." + 
                    " _trainX.nrOfRows = " + _trainX.First<Matrix>().M +
                    ", layerSizes.inputSize = " + layerSizes.First<int>());
            }

            if (_trainY.First<Matrix>().M != layerSizes.Last<int>())
            {
                throw new Exception("ANN.trainANN: Invalid output dimension." +
                    " _trainY.nrOfRows = " + _trainY.First<Matrix>().M +
                    ", layerSizes.outputSize = " + layerSizes.Last<int>());
            }

            if (miniBatchSize <= 0 || eta <= 0D || nrOfEpochs <= 0)
            {
                throw new Exception("ANN: Invalid meta parameters in trainANN:" +
                    " miniBatchSize = " + miniBatchSize.ToString() + 
                    ", eta = " + eta.ToString() +
                    ", nrOfEpochs = " + nrOfEpochs.ToString());
            }

            trainX = _trainX;
            trainY = _trainY;
            runAllEpochs(miniBatchSize, eta, nrOfEpochs);
        }

        public void runAllEpochs(int miniBatchSize, double eta, int nrOfEpochs)
        {
            Console.WriteLine("ANN: Number of epochs: " + nrOfEpochs);
            for (int epoch = 1; epoch <= nrOfEpochs; epoch++)
            {
                Console.WriteLine("ANN: Epoch number: " + epoch);
                runSingleEpoch(miniBatchSize, eta);
            }
        }

        public void runSingleEpoch(int miniBatchSize, double eta)
        {
            //Permute the training set:
            List<int> iList = new List<int>();

            for (int i = 0; i < trainX.Count; i++)
            {
                iList.Add(i);
            }

            List<int> rList = ListUtils.permute(iList);

            List<Matrix> trainX1 = new List<Matrix>();
            List<Matrix> trainY1 = new List<Matrix>();

            for (int i = 0; i < rList.Count; i++)
            {
                trainX1.Add(trainX.ElementAt<Matrix>(rList.ElementAt<int>(i)));
                trainY1.Add(trainY.ElementAt<Matrix>(rList.ElementAt<int>(i)));
            }

            trainX = trainX1;
            trainY = trainY1;
            //--- Permutation done ---

            //Partition into mini batches and train:
            List<Matrix> miniBatchX = null;
            List<Matrix> miniBatchY = null;

            for (int i = 0; i < trainX.Count; i = i + miniBatchSize)
            {
                miniBatchX = trainX.GetRange(i, Math.Min(miniBatchSize, trainX.Count - i));
                miniBatchY = trainY.GetRange(i, Math.Min(miniBatchSize, trainY.Count - i));

                Tuple<List<Matrix>, List<Matrix>> wbGradientList = computeMiniBatchGradient(miniBatchX, miniBatchY);
                adjustLearningParameters(eta, wbGradientList.Item1, wbGradientList.Item2);
            }
        }

        public void adjustLearningParameters(double eta, List<Matrix> wGradientList, List<Matrix> bGradientList)
        {
            if (wGradientList.Count != bGradientList.Count)
            {
                throw new Exception("ANN: Mismatching list sizes in adjusLearningParameters." +
                    " wGradientList.Count = " + wGradientList.Count + ", bGradientList.Count = " + bGradientList.Count);
            }

            Matrix[] wArray = wList.ToArray<Matrix>();
            Matrix[] bArray = bList.ToArray<Matrix>();

            for (int l = 1; l < wArray.Count<Matrix>(); l++)
            {
                wArray[l] = wArray[l] + (wGradientList[l] * (-eta));
                bArray[l] = bArray[l] + (bGradientList[l] * (-eta));
            }

            wList = wArray.ToList<Matrix>();
            bList = bArray.ToList<Matrix>();
        }

        public Tuple<List<Matrix>, List<Matrix>> computeMiniBatchGradient(List<Matrix> X, List<Matrix> Y)
        {
            if (X.Count != Y.Count)
            {
                throw new Exception("ANN: List sizes mismatch in computeMiniBatchGradient." +
                    " X.Count = " + X.Count.ToString() + ", Y.Count = " + Y.Count.ToString());
            }

            if (X.Count == 0)
            {
                throw new Exception("ANN: Invalid list sizes in computeMiniBatchGradient." +
                    " X.Count = " + X.Count.ToString() + ", Y.Count = " + Y.Count.ToString());
            }

            Matrix[] wSingletonGradientAggrArray = null;
            Matrix[] bSingletonGradientAggrArray = null;
            
            List<Matrix> wSingletonGradientList = null;
            List<Matrix> bSingletonGradientList = null;

            Tuple<List<Matrix>, List<Matrix>> wbSingletonGradients = null;

            wbSingletonGradients = computeSingletonGradient(X.ElementAt(0), Y.ElementAt(0));

            wSingletonGradientAggrArray = wbSingletonGradients.Item1.ToArray<Matrix>();
            bSingletonGradientAggrArray = wbSingletonGradients.Item2.ToArray<Matrix>();

            for (int i = 1; i < X.Count; i++)
            {
                wbSingletonGradients = computeSingletonGradient(X.ElementAt(i), Y.ElementAt(i));
                wSingletonGradientList = wbSingletonGradients.Item1;
                bSingletonGradientList = wbSingletonGradients.Item2;

                for (int l = 1; l < wSingletonGradientList.Count; l++)
                {
                    wSingletonGradientAggrArray[l] =
                        wSingletonGradientAggrArray[l] + wSingletonGradientList.ElementAt(l);
                    bSingletonGradientAggrArray[l] =
                        bSingletonGradientAggrArray[l] + bSingletonGradientList.ElementAt(l);
                }
            }

            for (int l = 1; l < wSingletonGradientAggrArray.Count<Matrix>(); l++)
            {
                wSingletonGradientAggrArray[l] = wSingletonGradientAggrArray[l] * (1D / X.Count);
                bSingletonGradientAggrArray[l] = bSingletonGradientAggrArray[l] * (1D / X.Count);
            }

            List<Matrix> wGradientList = wSingletonGradientAggrArray.ToList<Matrix>();
            List<Matrix> bGradientList = bSingletonGradientAggrArray.ToList<Matrix>();
            
            return Tuple.Create(wGradientList, bGradientList);
        }

        public Tuple<List<Matrix>, List<Matrix>> computeSingletonGradient(Matrix X, Matrix Y)
        {
            List<Matrix> wSingletonGradientList = new List<Matrix>();
            List<Matrix> bSingletonGradientList = new List<Matrix>();
            List<Matrix> deltaList = new List<Matrix>();

            //Feed forward:
            Tuple<List<Matrix>, List<Matrix>> zaLists = feedForward(X);
            List<Matrix> zList = zaLists.Item1;
            List<Matrix> aList = zaLists.Item2;

            //Back propagate:
            //BP1:
            deltaList.Add(
                MatrixUtils.hadamardProduct(
                    gradientCostOutput(aList.Last(), Y), 
                    derivativeActivation(zList.Last())
                )
            );

            //BP2:
            Matrix BP2Matrix = null;

            for (int l = layerSizes.Count - 2; l > 0; l--)
            {
                BP2Matrix =
                    MatrixUtils.hadamardProduct(
                        (~(wList.ElementAt(l + 1))) * deltaList.Last(),
                        derivativeActivation(zList.ElementAt(l))
                    );
                deltaList.Add(BP2Matrix);
            }
            deltaList.Add(null);
            deltaList.Reverse();

            //BP3, BP4:
            wSingletonGradientList.Add(null);
            bSingletonGradientList.Add(null);

            for (int l = 1; l < layerSizes.Count; l++)
            {
                wSingletonGradientList.Add(deltaList.ElementAt(l) * (~(aList.ElementAt(l - 1))));
                bSingletonGradientList.Add(deltaList.ElementAt(l));
            }

            return Tuple.Create(wSingletonGradientList, bSingletonGradientList);
        }

        public Tuple<List<Matrix>, List<Matrix>> feedForward(Matrix X)
        {
            List<Matrix> zList = new List<Matrix>();
            List<Matrix> aList = new List<Matrix>();

            zList.Add(null);
            aList.Add(X);

            Matrix Z = null;

            for (int l = 1; l < layerSizes.Count; l++)
            {
                Z = wList.ElementAt(l) * aList.ElementAt(l - 1) + bList.ElementAt(l);
                zList.Add(Z);
                aList.Add(activation(Z));
            }

            return Tuple.Create(zList, aList);
        }

        public Matrix activation(Matrix Z)
        {
            Matrix A = new Matrix(Z.M, Z.N);

            for (int m = 0; m < A.M; m++)
            {
                for (int n = 0; n < A.N; n++)
                {
                    A[m, n] = sigmoid(Z[m, n]);
                }
            }

            return A;
        }

        public Matrix derivativeActivation(Matrix Z)
        {
            Matrix D = new Matrix(Z.M, Z.N);

            for (int m = 0; m < D.M; m++)
            {
                for (int n = 0; n < D.N; n++)
                {
                    D[m, n] = sigmoid(Z[m, n]) * (1D - sigmoid(Z[m, n]));
                }
            }

            return D;
        }

        public double sigmoid(double z)
        {
            return 1D / (1D + Math.Exp(-z));
        }

        public Matrix gradientCostOutput(Matrix A_L, Matrix Y)
        {
            if (A_L.M != Y.M || A_L.N != Y.N)
            {
                throw new Exception("ANN: Dimension mismatch for gradientCostOutput." +
                    " A_L.M = " + A_L.M.ToString() + ", Y.M = " + Y.M.ToString() +
                    ", A_L.N = " + A_L.N.ToString() + ", Y.N = " + Y.N.ToString());
            }

            Matrix G = new Matrix(A_L.M, A_L.N);

            for (int m = 0; m < G.M; m++)
            {
                for (int n = 0; n < G.N; n++)
                {
                    G[m, n] = A_L[m, n] - Y[m, n];
                }
            }

            return G;
        }

        public ANN(List<int> _layerSizes)
        {
            if (_layerSizes == null)
            {
                throw new Exception("ANN: Invalid constructor argument.");
            }

            layerSizes = _layerSizes;

            Matrix W = null;
            Matrix b = null;

            wList.Add(W);
            bList.Add(b);

            for (int l = 1; l < layerSizes.Count; l++)
            {
                W = new Matrix(layerSizes.ElementAt(l), layerSizes.ElementAt(l - 1));
                for (int m = 0; m < W.M; m++)
                {
                    for (int n = 0; n < W.N; n++)
                    {
                        W[m, n] = random.NextNormal(0, 1);
                    }
                }

                b = new Matrix(layerSizes.ElementAt(l), 1);
                for (int m = 0; m < b.M; m++)
                {
                    for (int n = 0; n < b.N; n++)
                    {
                        b[m, n] = random.NextNormal(0, 1);
                    }
                }

                wList.Add(W);
                bList.Add(b);
            }
        }
    }
}
