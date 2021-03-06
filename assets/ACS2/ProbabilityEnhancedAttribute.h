/*
/       ACS2 in C++
/	------------------------------------
/       choice of without/with GA, subsumption, PEEs
/
/     (c) by Martin V. Butz
/     University of Wuerzburg / University of Illinois at Urbana/Champaign
/     butz@illigal.ge.uiuc.edu
/     Last modified: 02-23-2001
/
/     list structure for one attribute in the effect part (essential for PEEs) header
*/


#ifndef _ProbCharList_h_
#define _ProbCharList_h_

using namespace std;

#include<iostream>
#include<fstream>

// Single element of Enhanced Effect.
// List of (character, probability) tuples.
class ProbabilityEnhancedAttribute {
public:
    ProbabilityEnhancedAttribute(char c) {
        p = 1.0;
        this->c = c;
        next = 0;
    }

    ProbabilityEnhancedAttribute(ProbabilityEnhancedAttribute *oldList);

    ~ProbabilityEnhancedAttribute() { delete next; }

    void insert(ProbabilityEnhancedAttribute *merger, double q1, double q2);

    void insert(char c, double q1, double q2);

    void insert(char c);

    int remove(char c);

    char getBestChar();

    int doesContain(char c);

    int isEnhanced() { if (next != 0) return 1; else return 0; }

    int isSimilar(ProbabilityEnhancedAttribute *list2);

    int increaseProbability(char ch, double updateRate);

    friend ostream &operator<<(ostream &out, ProbabilityEnhancedAttribute *pcl);

private:
    void adjustProbabilities();

    void adjustProbabilities(double probSum);

    ProbabilityEnhancedAttribute *next;
    double p;
    char c;
};

#endif
