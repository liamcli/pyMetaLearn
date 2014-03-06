import unittest

import smac_workflow

class TestSMACWorkflow(unittest.TestCase):
    def test_read_state_runs(self):
        smac_workflow.read_state_runs()