import unittest
import numpy as np
from numpy.testing import assert_array_equal
from mbwind.system import StateArray


class TestStateArray(unittest.TestCase):
    def setUp(self):
        a = StateArray()
        a.new_states('fred', 'node', 3, 'node-0')
        a.new_states('bob', 'strain', 1, 'strains')
        a.new_states('fred', 'constraint', 2)
        self.a = a

    def test_len(self):
        self.assertEqual(len(self.a), 6)

    def test_cannot_add_duplicate_names(self):
        with self.assertRaises(ValueError):
            self.a.new_states('joe', 'node', 2, 'node-0')

    def test_owners_and_types(self):
        self.assertEqual(self.a.owners, ['fred', 'fred', 'fred',
                                         'bob',
                                         'fred', 'fred'])
        self.assertEqual(self.a.types, ['node', 'node', 'node',
                                        'strain',
                                        'constraint', 'constraint'])

    def test_getting_and_setting(self):
        a = self.a
        a[:] = np.arange(6)

        # Get by slice/index
        assert_array_equal(a[:], [0, 1, 2, 3, 4, 5])
        assert_array_equal(a[1:3], [1, 2])
        assert_array_equal(a[2], 2)

        # Get by name
        assert_array_equal(a['node-0'], [0, 1, 2])
        assert_array_equal(a['strains'], [3])

        # Set by slice/index
        a[2] = 99
        a[0:2] = 99
        assert_array_equal(a[:], [99, 99, 99, 3, 4, 5])

        # Set by name
        a['node-0'] = -1
        a['strains'] = [77]
        assert_array_equal(a[:], [-1, -1, -1, 77, 4, 5])

        # Length must be right
        with self.assertRaises(ValueError):
            a['node-0'] = [1, 2]

    def test_indices(self):
        self.assertEqual(self.a.indices(1), [1])
        self.assertEqual(self.a.indices(slice(2, 4)), [2, 3])
        self.assertEqual(self.a.indices('node-0'), [0, 1, 2])
        with self.assertRaises(KeyError):
            self.a.indices('not-a-name')

    def test_indices_by_type(self):
        self.assertEqual(self.a.indices_by_type('node'), [0, 1, 2])
        self.assertEqual(self.a.indices_by_type('strain'), [3])
        self.assertEqual(self.a.indices_by_type('constraint'), [4, 5])
        self.assertEqual(self.a.indices_by_type(['node', 'strain']),
                         [0, 1, 2, 3])
        self.assertEqual(self.a.indices_by_type('not-a-type'), [])

    def test_names_by_type(self):
        self.assertEqual(self.a.names_by_type('node'), ['node-0'])
        self.assertEqual(self.a.names_by_type('strain'), ['strains'])
        self.assertEqual(self.a.names_by_type('constraint'), [])
        self.assertEqual(self.a.names_by_type(['node', 'strain']),
                         ['node-0', 'strains'])
        self.assertEqual(self.a.names_by_type('not-a-type'), [])

    def test_get_type(self):
        self.assertEqual(self.a.get_type('node-0'), 'node')
        self.assertEqual(self.a.get_type(slice(0, 3)), 'node')
        self.assertEqual(self.a.get_type(3), 'strain')
        with self.assertRaises(ValueError):
            self.a.get_type(slice(2, 4))  # mixed types -> error
        # Returns None if not found
        self.assertEqual(self.a.get_type(slice(0, 0)), None)
        with self.assertRaises(KeyError):
            self.a.get_type('not-a-name')


if __name__ == '__main__':
    unittest.main()
