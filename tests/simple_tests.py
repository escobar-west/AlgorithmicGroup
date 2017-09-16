import numpy as np
import pandas as pd
import unittest
from init import Stock, pdb, debug_on
"""
This code tests the basic functionality of all stock indicators.
In order for your code to be committed, your repository must pass all of the tests in this file.
To run this test case, type "python test_simple.py" in the command line.
"""
A = Stock()
old_copy = A.df.copy()

class TestAsset(unittest.TestCase):

    def test_init(self):
        for val in dir(A):
            print('testing ', val, '...')
            res = eval('A.'+val+'()')

            msg = 'Return value of indicator must be a DataFrame or Series'
            self.assertTrue(
                    isinstance(res, pd.DataFrame) or isinstance(res, pd.Series), msg)

            msg = 'If return value is Series, it must have a valid name (non-empty string)'
            if isinstance(res, pd.Series):
                self.assertIsInstance(res.name, str, msg)
                self.assertIsNot(res.name, '', msg)

            msg = 'If return value is DataFrame, all column names must be non-empty strings'
            if isinstance(res, pd.DataFrame):
                for col in res.columns:
                    self.assertIsInstance(col, str, msg)
                    self.assertIsNot(col, '', msg)

            msg = 'Function must not alter the original Asset DataFrame'
            self.assertTrue(np.alltrue((A.df == old_copy) | (A.df.isnull())), msg)

            msg = 'Index of return value must equal index of Asset DataFrame'
            self.assertTrue(np.alltrue((res.index == A.df.index) | (A.df.index.isnull())), msg)

if __name__ == '__main__':
    unittest.main()
