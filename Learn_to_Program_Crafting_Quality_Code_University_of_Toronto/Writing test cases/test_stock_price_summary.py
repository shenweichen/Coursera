import a1
import unittest


class TestStockPriceSummary(unittest.TestCase):
    """ Test class for function a1.stock_price_summary. """

    # Add your test methods for a1.stock_price_summary here.

    def test_empty_list(self,):
        """
        stock_price contains no changes
        """
        self.assertTrue(a1.stock_price_summary([])==(0,0))
    def test_only_zero(self,):
        """
        stock_price contains only one zero
        """
        self.assertTrue(a1.stock_price_summary([0,])==(0,0))
    def test_only_gain(self,):
        """
        stock_price contains only one gain and no loss
        """
        self.assertTrue(a1.stock_price_summary([1,])==(1,0))
    def test_only_loss(self,):
        """
        stock_price contains only one loss and no gain
        """
        self.assertTrue(a1.stock_price_summary([-1,])==(0,-1))
    def test_normal(self,):
        """
        stock_price contains gains and losses
        """
        self.assertTrue(a1.stock_price_summary([-1,1,0.5,-0.2,0])==(1.5,-1.2))
if __name__ == '__main__':
    unittest.main(exit=False)
