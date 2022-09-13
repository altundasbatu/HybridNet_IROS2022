"""
Created on Mon Sep  15 20:43:13 2021

@author: baltundas3

ScheduleNet Environment based on OpenAI Code Test Environment
"""
from scheduling_env import SchedulingEnv
from mrc_problem import MRCProblem
from hybrid_team import HybridTeam

import unittest


class SingleRoundTest(unittest.TestCase):
    def setUp(self):
        self.fname = "tmp/test_file"
        self.problem = MRCProblem(fname=self.fname)
        self.team = HybridTeam(self.problem)

        # initialize env
        self.env = SchedulingEnv(self.problem, self.team)
        self.optimal, passed = self.problem.get_optimal_with_gurobi("tmp/test_file_gurobi.log", threads=1)
        self.assertTrue(passed)

    def tearDown(self):

        pass

    def test(self):
        # Testing Specifically for Tasks with Robots
        a = self.env.step(1, 0)
        b = self.env.step(2, 1)
        print(a, b)
        sum_val = -1.0 * (a[1] + b[1])
        # print(sum_val)
        print(sum_val, self.problem.optimal)
        self.assertAlmostEqual(sum_val, self.problem.optimal, 5)
        # Suboptimal
        self.env.reset()
        a = self.env.step(2, 1)
        b = self.env.step(1, 1)
        print(a, b)
        sum_val = -1.0 * (a[1] + b[1])
        self.assertNotAlmostEqual(sum_val, self.problem.optimal, 5)
        self.assertTrue(True)
        pass


if __name__ == '__main__':
    unittest.main()
