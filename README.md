# Engineering thesis branch

This is the branch for my engineering thesis. It's going to be removed once the work is fully upstreamed.

Thesis title: "Extending the functionality of pyALCS library with PEE"

Michał Szymański
Wrocław, 2018

# Anticipatory Learning Classifier Systems in Python
Repository containing code implementation for various *Anticipatory Learning Classifier Systems* (ALCS).

[![Build Status](https://travis-ci.org/ParrotPrediction/pyalcs.svg?branch=master)](https://travis-ci.org/ParrotPrediction/pyalcs) [![Documentation Status](https://readthedocs.org/projects/pyalcs/badge/?version=latest)](https://pyalcs.readthedocs.io/en/latest/?badge=latest)

## Documentation
Documentation is available [here](https://pyalcs.readthedocs.io).

### Citation
If you want to use the library in your project please cite the following:

    @inbook{
        title = "Integrating Anticipatory Classifier Systems with OpenAI Gym",
        keywords = "Aniticipatory Learning Classifier Systems, OpenAI Gym",
        author = "Norbert Kozlowski, Olgierd Unold",
        year = "2018",
        doi = "10.1145/3205651.3208241",
        isbn = "978-1-4503-5764-7/18/07",
        booktitle = "Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '18)",
        publisher = "Association for Computing Machinery",
    }

Prior to PR please execute to check if standards are holding:

    make test


### PRIVATE NOTES

    cout << time << " " << nrCorrect << "-" << nrWrong << "=" << (nrCorrect * 100 / (nrCorrect + nrWrong)) << " "
         << pop->getSize() << " " << pop->getNumSize() << " " << relList->getSize() << " " << pop->getSpecificity()
         << endl;

in testModel() (acs2++.cc). Interpretation of the output:

    2600 40-60=40 2280 2331 41 0.678411
     |   |   |  |  |    |   |   \-- specificity
     |   |   |  |  |    |   \-- size of relList
     |   |   |  |  |    \-- population size, including numerosity
     |   |   |  |  \-- population size, excluding numerosity
     |   |   |  \-- percentage of correct (knowledge)
     |   |   \-- number of wrong
     |   \-- number of correct
     \-- time
