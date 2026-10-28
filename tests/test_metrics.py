from types import SimpleNamespace
import unittest
import numpy as np
import torch

from utils.metrics import CORR, R2, metric, metric_report
from utils.losses import divide_no_nan, mape_loss, smape_loss, mase_loss


class MetricTests(unittest.TestCase):
    def test_pearson_matches_numpy_and_bounds(self):
        truth = np.array([[1., 6.], [2., 8.], [4., 10.], [7., 11.]])
        prediction = np.array([[2., 10.], [3., 8.], [3., 7.], [9., 2.]])
        expected = np.mean([np.corrcoef(truth[:, c], prediction[:, c])[0, 1] for c in range(2)])
        self.assertAlmostEqual(CORR(prediction, truth), expected)
        self.assertAlmostEqual(CORR(truth, truth), 1.)
        self.assertAlmostEqual(CORR(-truth, truth), -1.)

    def test_six_value_contract_known_values(self):
        values = metric(np.array([2., 4.]), np.array([1., 3.]))
        np.testing.assert_allclose(values, [1., 1., 1., 2 / 3, 5 / 9, 0.])

    def test_zero_denominators_are_explicitly_undefined(self):
        values = metric([1., 2.], [0., 2.])
        self.assertTrue(np.isnan(values[3]))
        self.assertTrue(np.isnan(values[4]))
        self.assertTrue(np.isnan(R2([1., 1.], [1., 1.])))
        report = metric_report([1., 2.], [0., 2.])
        self.assertEqual(report['zero_target_count'], 1)
        self.assertEqual(report['nonfinite_metrics'], ['mape', 'mspe'])

    def test_channel_r2_is_distinct_from_pooled(self):
        truth = np.array([[[0., 100.]], [[1., 101.]]])
        report = metric_report(truth + 1, truth)
        self.assertGreater(report['overall']['r2'], .99)
        self.assertEqual([r['r2'] for r in report['per_channel']], [-3., -3.])
        self.assertEqual(len(report['per_horizon']), 1)

    def test_shape_mismatch_cannot_broadcast_silently(self):
        with self.assertRaisesRegex(ValueError, 'same nonempty shape'):
            metric(np.ones((2, 1)), np.ones((2, 2)))

    def test_divide_no_nan_removes_both_infinity_signs(self):
        values = divide_no_nan(torch.tensor([-1., 1., 0.]), torch.zeros(3))
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(values.tolist(), [0., 0., 0.])

    def test_safe_divide_has_finite_gradients_at_zero_denominators(self):
        numerator = torch.tensor([-1., 1., 4.], requires_grad=True)
        denominator = torch.tensor([0., 0., 2.], requires_grad=True)
        divide_no_nan(numerator, denominator).sum().backward()
        self.assertTrue(torch.isfinite(numerator.grad).all())
        self.assertTrue(torch.isfinite(denominator.grad).all())
        self.assertEqual(numerator.grad.tolist(), [0., 0., .5])

    def test_short_forecast_loss_definitions_preserve_mask_mean(self):
        # The public scientific losses retain their historic mean over the full
        # mask shape; exp's adapter separately weights only valid elements.
        context = torch.tensor([[1., 2., 3., 4.]])
        target = torch.tensor([[1., 3., 9.]])
        forecast = torch.tensor([[2., 6., 10.]])
        mask = torch.tensor([[1., 1., 0.]])
        self.assertAlmostEqual(mape_loss()(context, 1, forecast, target, mask).item(), 2 / 3, places=6)
        self.assertAlmostEqual(smape_loss()(context, 1, forecast, target, mask).item(), 400 / 9, places=5)
        self.assertAlmostEqual(mase_loss()(context, 1, forecast, target, mask).item(), 4 / 3, places=6)


if __name__ == '__main__':
    unittest.main()
