import argparse
import logging

import gym
# noinspection PyUnresolvedReferences
import gym_maze

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.acs2.maze.utils import *
from lcs.agents.acs2 import ACS2, Configuration, ClassifiersList

# Configure logger
FORMAT = '%(module)s:%(lineno)d:%(levelname)s:%(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

logger = logging.getLogger("integration")


def find_best_classifier(population, situation, cfg):
    match_set = ClassifiersList.form_match_set(population, situation)
    anticipated_change_cls = [cl for cl in match_set if
                              cl.does_anticipate_change()]

    if (len(anticipated_change_cls) > 0):
        return max(anticipated_change_cls, key=lambda cl: cl.fitness)

    return None


def build_maze_image_matrix(env, population, cfg):
    original = env.env.maze.matrix
    fitness = original.copy()

    for index, x in np.ndenumerate(original):
        # Wall
        if x == 1:
            fitness[index] = 1

        # Reward or path
        if x == 9 or x == 0:
            fitness[index] = 0

    return fitness


def build_maze_text_matrix(env, population, cfg):
    original = env.env.maze.matrix
    action = original.copy().astype(str)

    # Think about more 'functional' way of doing this
    for index, x in np.ndenumerate(original):
        # Path or wall
        if x == 0 or x == 1:
            action[index] = ' '

        # Reward
        if x == 9:
            action[index] = '$'

    return action


def build_action_matrix(env, population, cfg):
    ACTION_LOOKUP = {
        0: u'↑', 1: u'↗', 2: u'→', 3: u'↘',
        4: u'↓', 5: u'↙', 6: u'←', 7: u'↖'
    }

    original = env.env.maze.matrix
    action = original.copy().astype(str)

    # Think about more 'functional' way of doing this
    for index, x in np.ndenumerate(original):
        # Path - best classfier fitness
        if x == 0:
            perception = env.env.maze.perception(index[1], index[0])
            best_cl = find_best_classifier(population, perception, cfg)
            if best_cl:
                action[index] = ACTION_LOOKUP[best_cl.action]
            else:
                action[index] = '?'

        # Wall - fitness = 0
        if x == 1:
            action[index] = ' '

        # Reward - inf fitness
        if x == 9:
            action[index] = '$'

    return action


def plot_maze(env, agent, cfg, ax=None, text_matrix_fcn=None):
    if ax is None:
        ax = plt.gca()

    if text_matrix_fcn is None:
        text_matrix_fcn = build_maze_text_matrix

    ax.set_aspect("equal")

    max_x = env.env.maze.max_x
    max_y = env.env.maze.max_y

    fitness_matrix = build_maze_image_matrix(env, agent.population, cfg)
    action_matrix = text_matrix_fcn(env, agent.population, cfg)

    # Render maze as image
    plt.imshow(fitness_matrix, interpolation='nearest', cmap='Greys',
               aspect='auto',
               extent=[0, max_x, max_y, 0])

    # Add labels to each cell
    for (y, x), val in np.ndenumerate(action_matrix):
        plt.text(x + 0.5, y + 0.5, val, fontsize=20, horizontalalignment='center',
      verticalalignment='center')

    plt.tick_params(which='both',
                    bottom=False, left=False,
                    labelbottom=False, labelleft=False)

    plt.xticks(np.arange(0, max_x, step=1))
    ax.grid(True)


def parse_metrics_to_df(metrics):
    def extract_details(row):
        row['trial'] = row['agent']['trial']
        row['numerosity'] = row['agent']['numerosity']
        row['reliable'] = row['agent']['reliable']
        row['correct_anticipations'] = row['agent']['correct_anticipations']
        row['knowledge'] = row['performance']['knowledge']
        return row

    df = pd.DataFrame(metrics)
    df = df.apply(extract_details, axis=1)
    df.drop(['agent', 'environment', 'performance'], axis=1, inplace=True)
    df['ca'] = df['correct_anticipations'].rolling(20).mean()
    df.set_index('trial', inplace=True)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="MazeF4-v0")
    parser.add_argument("--epsilon", default=0.5, type=float)
    parser.add_argument("--ga", action="store_true")
    parser.add_argument("--pee", action="store_true")
    parser.add_argument("--import-df", type=str)
    parser.add_argument("--pee-df", type=str)
    parser.add_argument("--plot-numerosity", action="store_true")
    parser.add_argument("--plot-knowledge", action="store_true")
    parser.add_argument("--plot-correct-rate", action="store_true")
    parser.add_argument("--plot-maze", action="store_true")
    parser.add_argument("--plot-policy", action="store_true")
    parser.add_argument("--explore-trials", default=50, type=int)
    parser.add_argument("--exploit-trials", default=10, type=int)
    args = parser.parse_args()

    if args.import_df is None:
        # Load desired environment
        maze = gym.make(args.environment)

        # Configure and create the agent
        cfg = Configuration(8, 8,
                            epsilon=args.epsilon,
                            do_ga=args.ga,
                            do_pee=args.pee,
                            performance_fcn=calculate_performance)
        logger.info(cfg)

        # Explore the environment
        agent = ACS2(cfg)
        population, explore_metrics = agent.explore(maze, args.explore_trials)

        # Exploit the environment
        agent = ACS2(cfg, population)
        population, exploit_metrics = agent.exploit(maze, args.exploit_trials)

        for metric in explore_metrics:
            logger.info(metric)

        for metric in exploit_metrics:
            logger.info(metric)

        print_detailed_knowledge(maze, population)

        df = parse_metrics_to_df(explore_metrics)

        df.to_csv("{}-{}-{}explore.csv".format(
            args.environment.lower().replace("-v0", ""),
            "pee" if args.pee else "nopee",
            args.explore_trials
        ))
    else:
        df = pd.read_csv(args.import_df)

    if args.plot_numerosity:
        fig, ax = plt.subplots(figsize=(5, 4))

        df['reliable'].plot(color='black', linestyle='-', linewidth=1.5, ax=ax,
                            label="wiarygodnych")
        df['numerosity'].plot(color='black', linestyle=':', linewidth=1.5, ax=ax,
                              label="wszystkich")

        if args.pee_df:
            pee_df = pd.read_csv(args.pee_df)
            pee_df['reliable'].plot(color='black', linestyle='--',
                                    linewidth=1.5,ax=ax,
                                    label="wiarygodnych (PEE)")
            pee_df['numerosity'].plot(color='black', linestyle='-.',
                                      linewidth=1.5, ax=ax,
                                      label="wszystkich (PEE)")

        ax.set_xlabel('Numer epizodu')
        ax.set_ylabel('Liczba klasyfikatorów')
        ax.legend()
        plt.savefig("{}-{}-{}explore-numerosity.pdf".format(
            args.environment.lower().replace("-v0", ""),
            "pee" if args.pee else "nopee",
            args.explore_trials
        ))

    if args.plot_knowledge:
        fig, ax = plt.subplots()

        df['knowledge'].plot(color='#000000', linewidth=1.5, ax=ax)

        ax.set_xlabel('Numer epizodu')
        ax.set_ylabel('Wiedza [%]')
        plt.savefig("{}-{}-{}explore-knowledge.pdf".format(
            args.environment.lower().replace("-v0", ""),
            "pee" if args.pee else "nopee",
            args.explore_trials
        ))

    if args.plot_correct_rate:
        fig, ax = plt.subplots()

        if args.pee_df is not None:
            pee_df = pd.read_csv(args.pee_df)
            df['ca'].plot(color='black', linewidth=1.5, ax=ax,
                          linestyle=':', label="bez PEE")
            pee_df['ca'].plot(color='black', linewidth=1.5, ax=ax,
                          linestyle='-', label="z PEE")
        else:
            df['ca'].plot(color='black', linewidth=1.5, ax=ax)

        ax.set_xlabel('Numer epizodu')
        ax.set_ylabel('Liczba poprawnych przewidywań [%]')
        ax.legend()
        plt.savefig("{}-{}-{}explore-correct-rate.pdf".format(
            args.environment.lower().replace("-v0", ""),
            "pee" if args.pee else "nopee",
            args.explore_trials
        ))

    if args.plot_maze:
        fig, ax = plt.subplots()
        plot_maze(maze, agent, cfg, ax)
        plt.savefig("{}.pdf".format(
            args.environment.lower().replace("-v0", ""),
            "pee" if args.pee else "nopee",
            args.explore_trials
        ))

    if args.plot_policy:
        fig, ax = plt.subplots()
        plot_maze(maze, agent, cfg, ax, build_action_matrix)
        plt.savefig("{}-{}-{}explore-policy.pdf".format(
            args.environment.lower().replace("-v0", ""),
            "pee" if args.pee else "nopee",
            args.explore_trials
        ))
