import unittest
import numpy as np
from numpy.testing import assert_array_equal
from mbwind.system import ArrayProxy


class TestArrayProxy(unittest.TestCase):
    def test_target_must_be_array(self):
        with self.assertRaises(ValueError):
            ArrayProxy([1, 2, 3])

    def test_without_subset(self):
        target = np.array([3, 5, 4, 1])
        proxy = ArrayProxy(target)
        self.assertEqual(len(proxy), 4)
        for i in range(len(target)):
            self.assertEqual(target[i], proxy[i])

        target[1] = 77
        self.assertEqual(proxy[1], 77)

        proxy[2] = 0
        self.assertEqual(target[2], 0)

    def test_with_subset(self):
        target = np.array([3, 5, 4, 1])
        proxy = ArrayProxy(target, subset=[1, 3])
        self.assertEqual(len(proxy), 2)
        self.assertEqual(proxy[0], target[1])
        self.assertEqual(proxy[1], target[3])

        target[1] = 77
        assert_array_equal(proxy[:], [77, 1])

        proxy[1] = 0
        assert_array_equal(target, [3, 77, 4, 0])

        proxy[:] = [-1, -2]
        assert_array_equal(target, [3, -1, 4, -2])

        # Length of value must match
        with self.assertRaises(ValueError):
            proxy[:] = [-1, -2, -3]

        # Test `in` operator
        self.assertFalse(0 in proxy)
        self.assertFalse(2 in proxy)
        self.assertFalse(4 in proxy)
        self.assertTrue(1 in proxy)
        self.assertTrue(3 in proxy)


if __name__ == '__main__':
    unittest.main()
