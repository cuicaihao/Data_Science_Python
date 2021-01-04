import unittest


class Test_DCQS(unittest.TestCase):
    def test_loop_sum(self):
        import a_loop_sum
        input = [10, 2, 3, 4, 6]
        target = 25
        output = a_loop_sum.sum(input)
        self.assertEqual(output, target)

    def test_recursive_sum(self):
        import b_recursive_sum
        input = [10, 2, 3, 4, 6]
        target = 25
        output = b_recursive_sum.sum(input)
        self.assertEqual(output, target)

    def test_recursive_count(self):
        import c_recursive_count
        input = [10, 2, 3, 4, 6]
        target = 5
        output = c_recursive_count.count(input)
        self.assertEqual(output, target)

    def test_recursive_max(self):
        import d_recursive_max
        input = [10, 2, 3, 4, 6]
        target = 10
        output = d_recursive_max.max_(input)
        self.assertEqual(output, target)

    def test_quicksort(self):
        import e_quicksort
        input = [10, 2, 3, 4, 6]
        target = [2, 3, 4, 6, 10]
        output = e_quicksort.quicksort(input)
        self.assertEqual(output, target)


if __name__ == '__main__':
    unittest.main()
