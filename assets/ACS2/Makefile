.PHONY: all

HEADERS += ACSConstants.h
HEADERS += Action.h
HEADERS += CharCharPosList.h
HEADERS += CharList.h
HEADERS += CharPosList.h
HEADERS += Classifier.h
HEADERS += ClassifierList.h
HEADERS += Condition.h
HEADERS += Effect.h
HEADERS += EnhancedEffect.h
HEADERS += Environment.h
HEADERS += MazeEnvironment.h
HEADERS += Perception.h
HEADERS += PMark.h
HEADERS += ProbabilityEnhancedAttribute.h

OBJECTS += acs2.o
OBJECTS += Action.o
OBJECTS += CharCharPosList.o
OBJECTS += CharList.o
OBJECTS += CharPosList.o
OBJECTS += Classifier.o
OBJECTS += ClassifierList.o
OBJECTS += Condition.o
OBJECTS += Effect.o
OBJECTS += EnhancedEffect.o
OBJECTS += Environment.o
OBJECTS += MazeEnvironment.o
OBJECTS += Perception.o
OBJECTS += PMark.o
OBJECTS += ProbabilityEnhancedAttribute.o

all: acs2

acs2: $(OBJECTS)
	g++ -o $@ -g -Wall $^ -lm

%.o: %.cc $(HEADERS)
	g++ -c -o $@ -g -Wall $<

clean:
	rm -f *.o acs2
