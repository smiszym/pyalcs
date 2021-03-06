def calculate_performance(maze, population):
    """
    Analyzes all possible transition in maze environment and checks if there
    is a reliable classifier for it.
    :param maze: maze object
    :param population: list of classifiers
    :return: percentage of knowledge
    """
    transitions = maze.env.get_all_possible_transitions()

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = maze.env.maze.perception(*start)
        p1 = maze.env.maze.perception(*end)

        if any([True for cl in population
                if cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return {
        'knowledge': nr_correct / len(transitions) * 100.0
    }


def print_detailed_knowledge(maze, population):
    result = ""
    transitions = maze.env.get_all_possible_transitions()

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = maze.env.maze.perception(*start)
        p1 = maze.env.maze.perception(*end)

        result += "\n{}-{}-\n{}".format("".join(p0), action, "".join(p1))
        result += "\n"
        result += str(population.form_match_set(p0).form_action_set(action))
        result += "\n"

    return result
