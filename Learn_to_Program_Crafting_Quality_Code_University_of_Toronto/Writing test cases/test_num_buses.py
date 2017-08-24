import a1
import unittest


class TestNumBuses(unittest.TestCase):
    """ Test class for function a1.num_buses. """

    # Add your test methods for a1.num_buses here.
    def test_zero(self,):
        """
        people number equals zero
        """
        self.assertEqual(a1.num_buses(0),0)
    def test_one(self,):
        """
        people number equals one
        """
        self.assertEqual(a1.num_buses(1),1)
    def test_fifty(self):
        """
        people number equals fifty
        """
        self.assertEqual(a1.num_buses(50),1)

    def test_gt_fifty(self):
        """
        people number greater than fifty
        """
        self.assertEqual(a1.num_buses(51),2)

if __name__ == '__main__':
    unittest.main(exit=False)
