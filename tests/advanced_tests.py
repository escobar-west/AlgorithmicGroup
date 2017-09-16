import numpy as np
import pandas as pd
import unittest
from init import Asset, pdb, debug_on

class TestAsset(unittest.TestCase):
    def test_RSI(self):
        close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
                 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25,
                 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13]

        target = np.array([70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 
                    62.38, 54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77])

        asset = Asset(pd.DataFrame(close, columns = ['Close']))
        RSI = asset.RSI().values[14:]

        print('RSI:\n', RSI)
        print('target:\n', target)
        self.assertTrue(np.allclose(RSI, target, atol=1.0e-1))

if __name__ == '__main__':
    unittest.main()
