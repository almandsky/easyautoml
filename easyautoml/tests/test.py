from unittest import TestCase

from easyautoml import tpotutils

class TestSetup(TestCase):
    def test_msele(self):
        result = tpotutils.msele([1,2,3], [1,2,3])
        self.assertTrue(result == 0.0)