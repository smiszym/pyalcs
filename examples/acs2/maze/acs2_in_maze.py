import argparse
import logging

import gym
# noinspection PyUnresolvedReferences
import gym_maze

from examples.acs2.maze.utils import calculate_performance
from lcs.agents.acs2 import ACS2, Configuration

# Configure logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("integration")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="MazeF4-v0")
    parser.add_argument("--epsilon", default=1.0, type=float)
    parser.add_argument("--ga", action="store_true")
    parser.add_argument("--explore-trials", default=50, type=int)
    parser.add_argument("--exploit-trials", default=10, type=int)
    args = parser.parse_args()

    # Load desired environment
    maze = gym.make(args.environment)

    # Configure and create the agent
    cfg = Configuration(8, 8,
                        epsilon=args.epsilon,
                        do_ga=args.ga,
                        performance_fcn=calculate_performance)
    logger.info(cfg)

    # Explore the environment
    agent = ACS2(cfg)
    population, explore_metrics = agent.explore(maze, args.explore_trials)

    # Exploit the environment
    agent = ACS2(cfg, population)
    population, exploit_metric = agent.exploit(maze, args.exploit_trials)

    for metric in exploit_metric:
        logger.info(metric)
