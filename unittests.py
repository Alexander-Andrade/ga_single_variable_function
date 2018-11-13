import unittest
from main import *
from sympy.combinatorics.graycode import gray_to_bin, bin_to_gray, random_bitstring
from bitstring import Bits


class GaTests(unittest.TestCase):

    def setUp(self):
        self.ga = GA(f=lambda x: x, target='max', solution_area=(0, 5), accuracy=3)

    def test_calc_chromosome_length(self):
        ga = GA(f=lambda x: x, target='max', solution_area=(-10, 10), accuracy=3)
        self.assertEqual(ga.calc_chromosome_length(), 15)
        ga = GA(f=lambda x: x, target='max', solution_area=(0, 5), accuracy=3)
        self.assertEqual(ga.calc_chromosome_length(), 13)

    def test_segment_value(self):
        self.assertEqual(self.ga.segment_value(Bits('0b1111111111111').uint), 5.)
        self.assertEqual(self.ga.segment_value(Bits('0b0000000000000').uint), 0.)

    def test_gray_to_uint(self):
        gray_code = bin_to_gray(Bits(uint=23, length=13).bin)
        self.assertEqual(gray_to_uint(gray_code), 23)


if __name__ == '__main__':
    unittest.main()
