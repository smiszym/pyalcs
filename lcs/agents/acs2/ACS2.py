import logging
from typing import Optional

from lcs import Perception

from . import ClassifiersList, Configuration
from ...agents import Agent
from ...agents.Agent import Metric
from ...strategies.action_selection import choose_action


logger = logging.getLogger(__name__)


class ACS2(Agent):
    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList=None) -> None:
        self.cfg = cfg

        if population:
            self.population = population
        else:
            self.population = ClassifiersList()

    def explore(self, env, trials):
        """
        Explores the environment in given set of trials.
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_explore)

    def exploit(self, env, trials):
        """
        Exploits the environments in given set of trials (always executing
        best possible action - no exploration).
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_exploit)

    def explore_exploit(self, env, trials):
        """
        Alternates between exploration and exploitation phases.
        :param env: environment
        :param trials: number of trials
        :return: population of classifiers and metrics
        """
        def switch_phases(env, steps, current_trial):
            if current_trial % 2 == 0:
                return self._run_trial_explore(env, steps)
            else:
                return self._run_trial_exploit(env, None)

        return self._evaluate(env, trials, switch_phases)

    def _evaluate(self, env, max_trials, func):
        """
        Runs the classifier in desired strategy (see `func`) and collects
        metrics.

        Parameters
        ----------
        env:
            OpenAI Gym environment
        max_trials: int
            maximum number of trials
        func: Callable
            Function accepting three parameters: env, steps already made,
             current trial

        Returns
        -------
        tuple
            population of classifiers and metrics
        """
        current_trial = 0
        steps = 0

        metrics = []
        while current_trial < max_trials:
            logger.info("** Running trial {}/{} using strategy `{}` **"
                        .format(current_trial, max_trials, func))

            steps_in_trial, reward, corr_pcnt = func(env, steps, current_trial)
            steps += steps_in_trial

            trial_metrics = self._collect_metrics(
                env, current_trial, steps_in_trial, steps, reward)
            trial_metrics['agent']['correct_anticipations'] = corr_pcnt

            metrics.append(trial_metrics)

            if current_trial % 25 == 0:
                logger.info(trial_metrics)

            current_trial += 1

        return self.population, metrics

    def _run_trial_explore(self, env, time, current_trial=None):
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = Perception(self.cfg.environment_adapter.to_genotype(raw_state))
        action = None
        reward = None
        total_reward = 0
        prev_state = None
        action_set = ClassifiersList()
        done = False
        correct_anticipations = 0
        all_anticipations = 0

        while not done:
            match_set = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                d_correct, d_all = ClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                correct_anticipations += d_correct
                all_anticipations += d_all
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma
                )
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        action_set,
                        state,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)

            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                self.cfg.epsilon)
            internal_action = self.cfg.environment_adapter.to_env_action(action)
            logger.info("Step {} of exploring the environment:".format(steps))
            logger.info(" * current state: {}".format(state))
            logger.info(" * current environment:\n{}".format(env.render('ansi')))
            logger.info(" * decision: executing action {}".format(action))
            action_set = match_set.form_action_set(action)

            prev_state = state
            raw_state, reward, done, _ = env.step(internal_action)
            state = Perception(self.cfg.environment_adapter.to_genotype(raw_state))

            if done:
                d_correct, d_all = ClassifiersList.apply_alp(
                    self.population,
                    None,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg)
                correct_anticipations += d_correct
                all_anticipations += d_all
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    0,
                    self.cfg.beta,
                    self.cfg.gamma)
            if self.cfg.do_ga:
                ClassifiersList.apply_ga(
                    time + steps,
                    self.population,
                    None,
                    action_set,
                    state,
                    self.cfg.theta_ga,
                    self.cfg.mu,
                    self.cfg.chi,
                    self.cfg.theta_as,
                    self.cfg.do_subsumption,
                    self.cfg.theta_exp)

            total_reward += reward
            steps += 1

        return steps, total_reward, 100.0 * correct_anticipations / all_anticipations if all_anticipations > 0 else 50.0

    def _run_trial_exploit(self, env, time=None, current_trial=None):
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)

        reward = None
        total_reward = 0
        action_set = ClassifiersList()
        done = False
        sum_rating = 0
        episode = []

        while not done:
            logger.info("Step {} of exploiting the environment:".format(steps))
            logger.info(
                " * current environment:\n{}".format(env.render('ansi')))

            match_set = self.population.form_match_set(state)
            logger.info(" * match set:\n{}".format(match_set))

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            # Here when exploiting always choose best action
            action = choose_action(
                match_set,
                self.cfg.number_of_possible_actions,
                epsilon=0.0)
            internal_action = self.cfg.environment_adapter.to_env_action(action)
            logger.info(" * decision: executing action {}".format(action))

            rating = None
            if 'rate_action' in env.env.__dir__():
                rating = env.env.rate_action(internal_action)
                if rating is not None:
                    sum_rating += rating
                    logger.info(" * action rating: {}".format(rating))

            action_set = match_set.form_action_set(action)
            logger.info(" * action set:\n{}".format(action_set))

            episode.append(internal_action)
            raw_state, reward, done, _ = env.step(internal_action)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            if done:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, reward, 0, self.cfg.beta, self.cfg.gamma)

            total_reward += reward
            steps += 1

        logger.info("This episode: {}".format("".join(str(x) for x in episode)))
        logger.info("Average action rating in this trial: {} in {} steps".format(sum_rating / steps, steps))

        return steps, total_reward, -1.0

    def _collect_agent_metrics(self, trial, steps, total_steps, reward) -> Metric:
        return {
            'population': len(self.population),
            'numerosity': sum(cl.num for cl in self.population),
            'reliable': len([cl for cl in
                             self.population if cl.is_reliable()]),
            'quality': (sum(cl.q for cl in self.population) /
                        len(self.population)),
            'fitness': (sum(cl.fitness for cl in self.population) /
                        len(self.population)),
            'trial': trial,
            'steps': steps,
            'total_steps': total_steps,
            'reward': reward
        }

    def _collect_environment_metrics(self, env) -> Optional[Metric]:
        if self.cfg.environment_metrics_fcn:
            return self.cfg.environment_metrics_fcn(env)

        return None

    def _collect_performance_metrics(self, env) -> Optional[Metric]:
        if self.cfg.performance_fcn:
            return self.cfg.performance_fcn(
                env, self.population, **self.cfg.performance_fcn_params)

        return None
