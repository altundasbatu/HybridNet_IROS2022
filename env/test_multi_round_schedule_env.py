"""
Created on Mon Sep  15 20:43:13 2021

@author: baltundas3

ScheduleNet Environment based on OpenAI Code Test Environment
"""
from scheduling_env import SchedulingEnv
from multi_round_scheduling_env import MultiRoundSchedulingEnv
from mrc_problem import MRCProblem
from hybrid_team import HybridTeam

import unittest


class SingleRoundTest(unittest.TestCase):
    def setUp(self):
        self.fname = "tmp/test_file"
        self.problem = MRCProblem(fname=self.fname)
        self.team = HybridTeam(self.problem)
        # initialize env
        self.env = MultiRoundSchedulingEnv(self.problem, self.team)
        self.optimal, passed = self.problem.get_optimal_with_gurobi("tmp/test_file_gurobi.log", threads=1)
        self.assertTrue(passed)

    def tearDown(self):

        pass

    def test(self):
        # Optimal Schedule Test
        schedule = [(2, 1, 1.0), (1, 0, 1.0)]
        ret = self.env.step(schedule)
        self.assertAlmostEqual(ret[1] * -1.0, self.optimal, 5)

        # Suboptimal Schedule Test
        schedule2 = [(1, 1, 1.0), (2, 2, 1.0)]
        ret = self.env.step(schedule2)
        self.assertNotEqual(-1 * ret[1], self.optimal)
        self.assertAlmostEqual(ret[1], -9.0, 5)

        # a = self.env.step(2, 0)
        # b = self.env.step(1, 1)
        # print(a, b)
        self.assertTrue(True)
        pass


if __name__ == '__main__':
    unittest.main()
