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
    public class RandomUtils : Random
    {
        public RandomUtils()
            : base()
        {
            ;
        }

        /// <summary>
        /// NextNormal uses Box-Muller transform to return the next double
        /// sampled from the normal distribution.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        /// <returns></returns>
        public double NextNormal(double mean, double variance)
        {
            if (variance < 0)
            {
                throw new Exception("RandomUtils: invalid variance in NextNormal." +
                    " variance = " + variance.ToString());
            }

            double u1 = 1.0 - this.NextDouble();
            double u2 = 1.0 - this.NextDouble();

            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            double randNormal = mean + Math.Sqrt(variance) * randStdNormal;

            return randNormal;
        }
    }
}
