"""Tests that the CLI accepts the single/global CA-EWC condition schemes."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._heavy_stubs import install_stubs  # noqa: E402

install_stubs()

from src import cli  # noqa: E402


_REQUIRED = [
    "--year_path", "/x",
    "--eral_path", "/x",
    "--erap_path", "/x",
    "--gfs_path", "/x",
    "--meic_path", "/x",
    "--pm25_path", "/x",
    "--model_name", "unet",
    "--mode", "rolling_train_predict",
    "--work_dir", "/tmp/pm25_test_workdir",
    "--patch_h", "4",
    "--patch_w", "4",
]


class CliSchemeChoicesTest(unittest.TestCase):
    def _parse(self, scheme):
        argv = ["prog", *_REQUIRED, "--ca_ewc_condition_scheme", scheme]
        with mock.patch.object(sys, "argv", argv):
            return cli.parse_args()

    def test_supported_schemes_accepted(self):
        for scheme in ("season4", "month12", "single", "global"):
            args = self._parse(scheme)
            self.assertEqual(args.ca_ewc_condition_scheme, scheme)

    def test_invalid_scheme_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse("not_a_scheme")


if __name__ == "__main__":
    unittest.main()
