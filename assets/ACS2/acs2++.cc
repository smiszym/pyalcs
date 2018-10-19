/*
/       ACS2 in C++
/	------------------------------------
/       choice without/with GA, subsumption, PEEs
/
/     (c) by Martin V. Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 02-23-2001
/
/     main program
*/

#include<iostream>
#include<fstream>

#include <sys/resource.h>
#include <unistd.h>

#include "Perception.h"
#include "Action.h"
#include "ACSConstants.h"
#include "Classifier.h"
#include "ClassifierList.h"
#include "MazeEnvironment.h"
#include "MPEnvironment.h"

using namespace std;

int Perception::length = 0;
Environment *Action::env = 0;

/**
 * Keeps the maximum knowedge percentage reached so far.
 */
int knowledge;

void startExperiments(Environment *env);

void startOneExperiment(Environment *env, ofstream *out);

int startOneTrialExplore(ClassifierList *population, Environment *env, int time, ofstream *out);

int startOneTrialExploit(ClassifierList *population, Environment *env);

void startCRRatExperiment(Environment *env, ofstream *out);

void printTestSortedClassifierList(ClassifierList *list, Environment *env, ofstream *out);

void testModel(ClassifierList *pop, ofstream *out, Environment *env, int time);

void writeRewardPerformance(ClassifierList *pop, int *steps, int time, int trial, ofstream *out);

void randomize(void);

/**
 * main requires the input of one default parameter. 
 * This is a maze file in a MazeEnvironment run and nothing otherwise.
 */
int main(int args, char *argv[]) {
    /* set the priority */
    setpriority(PRIO_PROCESS, getpid(), 0);

    randomize();

    ENVIRONMENT_CLASS *env;
    if (args == 2) {
        env = new ENVIRONMENT_CLASS(argv[1]);
    } else {
        cout << "usage: acs++.out (MazeFile)Name" << endl;
        exit(0);
    }

    Perception::length = env->getPerceptionLength();
    Action::env = env;

    startExperiments(env);

    delete env;
}

/**
 * Controls the execution of the specified number of experiments.
 */
void startExperiments(Environment *env) {
    ofstream *out = new ofstream(RESULT_FILE, ios::out);

    *out << "# beta: " << BETA << " gamma: " << GAMMA << " theta_i: " << THETA_I << " theta_e: " << THETA_R
         << " r_ini: " << R_INI << " q_ini:" << Q_INI << " avt_ini: " << AVT_INI << " q_alp_min: " << Q_ALP_MIN
         << " q_ga_decrease: " << Q_GA_DECREASE << endl;
    *out << "# umax: " << U_MAX << " doPees: " << DO_PEES << " epsilon: " << EPSILON << " prob_exploration_bias: "
         << PROB_EXPLORATION_BIAS << " exploration bias method: " << EXPLORATION_BIAS_METHOD << endl;
    *out << "# do_ga: " << DO_GA << " theta_ga: " << THETA_GA << " mu: " << MU << " X.type: " << X_TYPE << " chi: "
         << CHI << " theta_as: " << THETA_AS << " theta_exp: " << THETA_EXP << " do_subsumption: " << DO_SUBSUMPTION
         << endl;
    *out << "# max_steps: " << MAX_STEPS << " max_trial_steps: " << MAX_TRIAL_STEPS << " anz_experiments: "
         << ANZ_EXPERIMENTS << " reward_test: " << REWARD_TEST << " model_test_iteration: " << MODEL_TEST_ITERATION
         << " reward_test_iteration: " << REWARD_TEST_ITERATION << endl;

    char *id = env->getID();

    for (int i = 0; i < ANZ_EXPERIMENTS; i++) {
        *out << "Next Experiment" << endl;
        cout << "Experiment Nr: " << (i + 1) << endl;
        if (strcmp(id, "CRRat") == 0) {
            startCRRatExperiment(env, out);
        } else {
            startOneExperiment(env, out);
        }
    }

    delete[] id;
    delete out;
}

/**
 * Controls the execution of one experiment.
 */
void startOneExperiment(Environment *env, ofstream *out) {
    int time = 0, trial, exploitSteps[REWARD_TEST_ITERATION];

    for (trial = 0; trial < REWARD_TEST_ITERATION; trial++) {
        char *id = env->getID();
        if (strcmp(id, "MP") == 0)
            exploitSteps[trial] = 0;
        else
            exploitSteps[trial] = REWARD_TEST_ITERATION;
        delete[] id;
    }

    ClassifierList *population = new ClassifierList(env);
    cout << population;

    knowledge = 0;
    trial = 0;

    while ((REWARD_TEST || time <= MAX_STEPS) && (!REWARD_TEST || trial <= MAX_STEPS)) {
        time += startOneTrialExplore(population, env, time, out);
        if (REWARD_TEST) {
            exploitSteps[trial % REWARD_TEST_ITERATION] = startOneTrialExploit(population, env);
            if (trial % REWARD_TEST_ITERATION == 0) {
                writeRewardPerformance(population, exploitSteps, time, trial, out);
            }
        }
        trial++;
    }
    //printTestSortedClassifierList(population, env, out);
    //*out<<population<<endl;

    population->deleteClassifiers();
    delete population;
}

/**
 * Controls the execution of one exploration (learning) trial. The environment specifies when one trial ends.
 */
int startOneTrialExplore(ClassifierList *population, Environment *env, int time, ofstream *out) {
    ClassifierList *matchSet, *actionSet = 0;
    int steps;
    double rho0 = 0;

    Perception *situation = new Perception();
    Perception *previousSituation = new Perception();

    env->reset();
    env->getSituation(situation);

    Action *act = new Action();

//    for (steps = 0; !env->isReset() && (REWARD_TEST || time + steps <= MAX_STEPS) &&
//                    (REWARD_TEST || steps < MAX_TRIAL_STEPS); steps++) {

    for (steps = 0; !env->isReset() && (time + steps <= MAX_STEPS) && (steps < MAX_TRIAL_STEPS); steps++) {

        if (!REWARD_TEST && (time + steps) % MODEL_TEST_ITERATION == 0) {
            testModel(population, out, env, time + steps);
        }

        if (DO_MENTAL_ACTING_STEPS > 0)
            population->doOneStepMentalActing(DO_MENTAL_ACTING_STEPS);

        matchSet = new ClassifierList(population, situation);

        if (steps > 0) {
            //Learning in the last action set.
            actionSet->applyALP(previousSituation, act, situation, time + steps, population, matchSet);
            actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR());
            delete actionSet;
        }

        matchSet->chooseAction(act, population, situation);
        actionSet = new ClassifierList(matchSet, act);
        delete matchSet;

        rho0 = env->executeAction(act);

        previousSituation->setPerception(situation);
        env->getSituation(situation);

        if (env->isReset()) {
            //Learning in the current action set if end of trial.
            actionSet->applyALP(previousSituation, act, situation, time + steps, population, 0);
            actionSet->applyReinforcementLearning(rho0, 0);
        }
    }
    delete actionSet;
    delete situation;
    delete previousSituation;
    delete act;

    return steps;
}

/**
 * Executes on explotation trial. 
 * Here always the apparent best action (i.e. max(q*r)) is executed.
 */
int startOneTrialExploit(ClassifierList *population, Environment *env) {
    ClassifierList *matchSet, *actionSet;
    int steps;
    double rho0 = 0;

    Perception *situation = new Perception();

    env->reset();
    env->getSituation(situation);

    Action *act = new Action();

    for (steps = 0; !env->isReset() && steps < MAX_TRIAL_STEPS; steps++) {

        matchSet = new ClassifierList(population, situation);

        if (steps > 0) {
            //Reinforcement learning also during exploitation
            actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR());
            delete actionSet;
        }

        matchSet->chooseBestQRAction(act);
        actionSet = new ClassifierList(matchSet, act);
        delete matchSet;

        rho0 = env->executeAction(act);

        env->getSituation(situation);

        if (env->isReset()) {
            actionSet->applyReinforcementLearning(rho0, 0);
        }
    }
    delete actionSet;
    delete situation;

    char *id = env->getID();
    if (strcmp(id, "MP") == 0) {
        if (rho0 < 1)
            return 0;
    }
    delete[] id;

    return steps;
}

/**
 * This function is programmed to execute the colwill/rescorla rat experiments.
 * This is quite a hack and just suitable for the specific experiments:-)
 * Hereby, reset denotes when the testing phase starts.
 * during testing the returned reward denotes if the action was the better one!
 */
void startCRRatExperiment(Environment *env, ofstream *out) {
    ClassifierList *population = new ClassifierList(env);
    cout << population;

    ClassifierList *matchSet, *actionSet = 0;
    int time = 0;
    int steps = 0;
    double rho0 = 0;
    Perception *situation = new Perception();
    Perception *previousSituation = new Perception();
    Action *act = new Action();

    while (env->reset() == 1) {
        if (DO_MENTAL_ACTING_STEPS > 0)
            population->doOneStepMentalActing(DO_MENTAL_ACTING_STEPS);
        steps = 0;
        env->getSituation(situation);
        while (!env->isReset()) {
            matchSet = new ClassifierList(population, situation);

            if (steps > 0) {
                //Learning in the last action set.
                actionSet->applyALP(previousSituation, act, situation, time + steps, population, matchSet);
                actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR());
                delete actionSet;
            }

            matchSet->chooseAction(act, population, situation);
            actionSet = new ClassifierList(matchSet, act);
            delete matchSet;

            //cout<<situation<<"-"<<act<<endl;

            rho0 = env->executeAction(act);
            steps++;
            time++;

            previousSituation->setPerception(situation);
            env->getSituation(situation);

            if (env->isReset()) {
                //Learning in the current action set if end of trial.
                actionSet->applyALP(previousSituation, act, situation, time + steps, population, 0);
                actionSet->applyReinforcementLearning(rho0, 0);
            }
        }
        delete actionSet;
        actionSet = 0;
    }

    //*out<<population<<endl;

    int testTrial = 0;
    do {
        //do final phase here and record behavior
        //hereby we assume one-step problem but execute the set behavior
        if (DO_MENTAL_ACTING_STEPS > 0)
            population->doOneStepMentalActing(DO_MENTAL_ACTING_STEPS);
        steps = 0;
        env->getSituation(situation);
        testTrial++;
        while (!env->isReset()) {
            matchSet = new ClassifierList(population, situation);

            if (steps > 0) {
                //Learning in the last action set.
                actionSet->applyALP(previousSituation, act, situation, time + steps, population, matchSet);
                actionSet->applyReinforcementLearning(rho0, matchSet->getMaximumQR());
                delete actionSet;
            }
            matchSet->chooseAction(act, population, situation);
            actionSet = new ClassifierList(matchSet, act);
            delete matchSet;

            //*out<<situation<<"-"<<act<<endl;

            rho0 = env->executeAction(act);
            steps++;
            time++;

            previousSituation->setPerception(situation);
            env->getSituation(situation);

            if (env->isReset()) {
                //Learning in the current action set if end of trial.
                actionSet->applyALP(previousSituation, act, situation, time + steps, population, 0);
                actionSet->applyReinforcementLearning(0, 0);
            }
        }
        *out << testTrial << " " << rho0 << " " << population->getSize() << endl;

        delete actionSet;
        actionSet = 0;
    } while (env->reset() == 2);

    delete situation;
    delete previousSituation;
    delete act;

    //*out<<population<<endl;

    population->deleteClassifiers();
    delete population;
}


/**
 * Prints Classifiers that match in from the environment env requested test situations.
 * Prints to the stream out.
 */
void printTestSortedClassifierList(ClassifierList *list, Environment *env, ofstream *out) {
    env->doTesting();
    Perception *s1 = new Perception();
    Action *a = new Action();
    Perception *s2 = new Perception();

    while (env->getNextTest(s1, a, s2)) {
        ClassifierList *matchSet = new ClassifierList(list, s1);
        ClassifierList *actionSet = new ClassifierList(matchSet, a);
        *out << s1 << "-" << a << "-" << endl << s2 << endl << actionSet << endl;
        delete actionSet;
        delete matchSet;
    }
    env->endTesting();
    delete s1;
    delete a;
    delete s2;
}

/**
 * Tests the current environmental model of ACS2 and writes result to stream out.
 * Testing is done by interaction with the environment that provides the test triples.
 * The reliable list is searched for a classifier that matches, specifies the action, 
 * and anticipates correctly. The parameter knowledge is global and serves for monitoring 
 * purposes. 
 */
void testModel(ClassifierList *pop, ofstream *out, Environment *env, int time) {
    ClassifierList *relList = new ClassifierList(pop, THETA_R);
    double nrCorrect = 0, nrWrong = 0;
    env->doTesting();
    Perception *s1 = new Perception();
    Action *a = new Action();
    Perception *s2 = new Perception();

    while (env->getNextTest(s1, a, s2)) {
        //cout<<s1<<"-"<<a<<"-"<<endl<<s2<<endl;
        if (relList->existClassifier(s1, a, s2, THETA_R))
            nrCorrect++;
        else {
            nrWrong++;
            /*if(time>500000){
          cout<<s1<<"-"<<a<<"-"<<endl<<s2<<endl;
          }*/
        }
    }

    if (knowledge < nrCorrect * 100 / (nrCorrect + nrWrong)) {
        while (knowledge < nrCorrect * 100 / (nrCorrect + nrWrong))
            knowledge += 2;
        cout << "Knowlege greater than " << knowledge << "% at time " << time << endl;
    }

    cout << time << " " << nrCorrect << "-" << nrWrong << "=" << (nrCorrect * 100 / (nrCorrect + nrWrong)) << " "
         << pop->getSize() << " " << pop->getNumSize() << " " << relList->getSize() << " " << pop->getSpecificity()
         << endl;
    *out << time << " " << (nrCorrect * 100 / (nrCorrect + nrWrong)) << " " << pop->getSize() << " "
         << pop->getNumSize() << " " << relList->getSize() << " " << pop->getSpecificity() << endl;

    env->endTesting();
    delete relList;
    delete s1;
    delete a;
    delete s2;
}

/**
 * Used for randomizing the random number generator.
 */
void randomize(void) {
    int i;
    for (i = 0; i < time(NULL) % 1000; rand(), i++);
}

