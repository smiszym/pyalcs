cmake_minimum_required(VERSION 3.7)
project(acs2)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        acs2.cc
        ACSConstants.h
        Action.cc
        Action.h
        CharCharPosList.cc
        CharCharPosList.h
        CharList.cc
        CharList.h
        CharPosList.cc
        CharPosList.h
        Classifier.cc
        Classifier.h
        ClassifierList.cc
        ClassifierList.h
        Condition.cc
        Condition.h
        Effect.cc
        Effect.h
        Environment.cc
        Environment.h
        MazeEnvironment.cc
        MazeEnvironment.h
        Perception.cc
        Perception.h
        PMark.cc
        PMark.h
        ProbabilityEnhancedAttribute.cc
        ProbabilityEnhancedAttribute.h
        EnhancedEffect.cc
        EnhancedEffect.h)

add_executable(acs2 ${SOURCE_FILES})
