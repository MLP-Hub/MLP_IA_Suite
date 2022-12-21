# coding=utf-8
"""Resources test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'claire.wright.mi@gmail.com'
__date__ = '2022-12-15'
__copyright__ = 'Copyright 2022, Claire Wright | Mountain Legacy Project'

import unittest

from qgis.PyQt.QtGui import QIcon



class MLP_IA_SuiteDialogTest(unittest.TestCase):
    """Test rerources work."""

    def setUp(self):
        """Runs before each test."""
        pass

    def tearDown(self):
        """Runs after each test."""
        pass

    def test_icon_png(self):
        """Test we can click OK."""
        path = ':/plugins/MLP_IA_Suite/icon.png'
        icon = QIcon(path)
        self.assertFalse(icon.isNull())

if __name__ == "__main__":
    suite = unittest.makeSuite(MLP_IA_SuiteResourcesTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)



