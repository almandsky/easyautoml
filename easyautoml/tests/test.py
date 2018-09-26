from unittest import TestCase

import easyautoml

class TestSetup(TestCase):
    def test_msele(self):
        result = easyautoml.msele([1,2,3], [1,2,3])
        self.assertTrue(result == 0.0)