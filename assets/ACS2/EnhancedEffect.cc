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
/     list structure for effect part
*/

#include<iostream>

#include"ProbabilityEnhancedAttribute.h"
#include"EnhancedEffect.h"

/**
 * Creates a new list with one item
 */
EnhancedEffect::EnhancedEffect(char chr, int pos) {
    first = new AttrWithPos(new ProbabilityEnhancedAttribute(chr), pos);
    act = first;
    size = 1;
}

/**
 * Creates a copy of the old ProbCharPosList oldList.
 */
EnhancedEffect::EnhancedEffect(EnhancedEffect *oldList) {
    AttrWithPos *cplp = oldList->first, *cplpNew;

    if (oldList->first != 0) {
        first = new AttrWithPos(new ProbabilityEnhancedAttribute(cplp->getItem()), cplp->p);
        cplpNew = first;
        while (cplp->next != 0) {
            cplp = cplp->next;
            cplpNew->next = new AttrWithPos(new ProbabilityEnhancedAttribute(cplp->getItem()), cplp->p);
            cplpNew = cplpNew->next;
        }
    } else {
        first = 0;
    }
    act = first;
    size = oldList->size;
}

/**
 * Inserts new list object with charcter c and position p in order.
 * Does not affect pointer act except if the first item is inserted.
 * @return If item was successfully inserted.
 */
int EnhancedEffect::insert(char chr, int pos) {
    AttrWithPos *cpip, *cpipl;

    //First item
    if (first == 0) {
        first = new AttrWithPos(new ProbabilityEnhancedAttribute(chr), pos);
        size = 1;
        act = first;
        return 1;
    }

    //Look for position in the ordered list
    for (cpip = first, cpipl = 0; cpip != 0; cpip = cpip->next) {
        if (cpip->p >= pos)
            break;
        cpipl = cpip;
    }
    if (cpip != 0 && cpip->p == pos) {
        cpip->item->insert(chr); //Item exists already!->enhance the effect part!
    } else {
        //Now insert at the determined position
        if (cpipl == 0) {
            first = new AttrWithPos(new ProbabilityEnhancedAttribute(chr), pos);
            first->next = cpip;
        } else {
            cpipl->next = new AttrWithPos(new ProbabilityEnhancedAttribute(chr), pos);
            cpipl->next->next = cpip;
        }
        size++;
    }
    return 1;
}

/**
 * Insert character chr at the empty position epos 
 * Does not affect pointer act except if the first item is inserted.
 */
int EnhancedEffect::insertAt(char chr, int epos) {
    AttrWithPos *cpip, *cpipl;
    int pos;

    //First item
    if (first == 0) {
        first = new AttrWithPos(new ProbabilityEnhancedAttribute(chr), epos);
        size = 1;
        act = first;
        return 1;
    }
    //Determine pos of empty position
    for (cpip = first, cpipl = 0, pos = -1; cpip != 0; cpip = cpip->next) {
        if (cpip->p - pos - 1 > epos)
            break;
        epos -= (cpip->p - pos - 1);
        pos = cpip->p;
        cpipl = cpip;
    }

    //Now insert!
    if (cpipl == 0) {
        first = new AttrWithPos(new ProbabilityEnhancedAttribute(chr), epos);
        first->next = cpip;
    } else {
        cpipl->next = new AttrWithPos(new ProbabilityEnhancedAttribute(chr), pos + epos + 1);
        cpipl->next->next = cpip;
    }
    size++;
    act = first;
    return 1;
}


/**
 * Removes item with key pos.
 */
int EnhancedEffect::remove(int pos) {
    AttrWithPos *cpip, *cpipl;
    for (cpip = first, cpipl = 0; cpip != 0; cpip = cpip->next) {
        if (cpip->p == pos)
            break;
        cpipl = cpip;
    }
    if (cpip == 0)//Item not found
        return 0;

    remove(cpip, cpipl);
    return 1;
}

/**
 * Removes nr'st item. (0 init)
 */
int EnhancedEffect::removeAt(int nr) {
    AttrWithPos *cpip, *cpipl;
    for (cpip = first, cpipl = 0; nr != 0 && cpip != 0; cpip = cpip->next) {
        nr--;
        cpipl = cpip;
    }

    if (cpip == 0)//Item not found
        return 0;

    remove(cpip, cpipl);
    return 1;
}

/**
 * Direct Remover with pointers.
 */
void EnhancedEffect::remove(AttrWithPos *cpip, AttrWithPos *cpipl) {
    if (cpipl == 0) {
        first = cpip->next;
    } else {
        cpipl->next = cpip->next;
    }
    cpip->next = 0;
    delete cpip;
    size--;
    act = first;
}

/**
 * Returns current ProbCharPosItem and sets act (current) to the next item in the list
 */
AttrWithPos *EnhancedEffect::getNextItem() {
    AttrWithPos *ret = act;
    if (act != 0)
        act = act->next;
    return ret;
}

/**
 * Returns item with key pos (if it exists), 0 otherwise
 */
AttrWithPos *EnhancedEffect::getItem(int pos) {
    for (AttrWithPos *item = first; item != 0; item = item->next)
        if (item->p == pos)
            return item;
    return 0;
}

