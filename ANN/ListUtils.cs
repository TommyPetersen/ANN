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
    class ListUtils
    {
        public static List<int> permute(List<int> inputList, Random random)
        {
            if (inputList == null)
            {
                throw new Exception("ListUtils: inputList is null.");
            }

            List<int> sList = new List<int>();
            for (int l = 0; l < inputList.Count; l++)
            {
                sList.Add(inputList.ElementAt<int>(l));
            }

            List<int> rList = new List<int>();
            int i = 0;
            int number = 0;

            while (sList.Count > 0)
            {
                i = random.Next(sList.Count);
                number = sList.ElementAt<int>(i);
                rList.Add(number);
                sList.RemoveAt(i);
            }

            return rList;
        }
    }
}
