# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:55:22 2020

@author: Marian
"""

import os
import random
from typing import Dict, Tuple, List

import numpy as np
from dwave.cloud import Client


def solveWithDwave(qubo, num_reads=100, solver_name="DW_2000Q_VFYC_6"):
    """

    :param qubo: The qubo to be solved
    :param num_reads: The amount of reads the D'Wave quantum annealer should do
    :param solver_name: The D'Wave Annealer to be used
    :return: num_reads Solutions of the Qubo
    """
    with Client.from_config() as client:
        solver = client.get_solver(solver_name)
        computation = solver.sample_qubo(qubo, num_reads=num_reads)
        result = computation.result()
        return result


def getHardwareAdjacency(solver_name="DW_2000Q_VFYC_6"):
    """

    :param solver_name: The D'Wave Annealer to be used
    :return: A graph with the Chimera structure of the D'Wave Annealer
    """
    with Client.from_config() as client:
        solver = client.get_solver(solver_name)
        return (solver.nodes, solver.undirected_edges)


def invertField(field):
    """

    :param field: A Quantum game of Life field
    :return: A Quantum game of Life field with living and dead cells flipped
    """
    inverted_field = np.zeros(field.shape)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if round(field[i, j], 10) == 0:
                inverted_field[i, j] = 1
            elif round(field[i, j], 10) == 1:
                inverted_field[i, j] = 0
            else:
                print("An Error occured: Some value in the fields is not valid!")
                raise ValueError
    return inverted_field


def checkIfFieldCompleteZero(field):
    """

    :param field: A Quantum game of Life field
    :return: true if all cells are dead, false in any other case
    """
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if round(field[i, j], 10) == 1:
                return False
    return True


def checkIfFieldCompleteOne(field):
    """

    :param field: A Quantum game of Life field
    :return: true if all cells are alive, false in any other case
    """
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if round(field[i, j], 10) == 0:
                return False
    return True


def intializeField(width: int, height: int, one_coverage=0.):
    """

    :param width: width of the Quantum game of Life field
    :param height: heigth of the Quantum game of Life field
    :param one_coverage: probability of a cell being alive on initialization
    :return: a Quantum game of Life field
    """
    field = np.zeros((height, width))
    if round(one_coverage, 10) > 0:
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if random.random() < one_coverage:
                    field[i, j] = 1
    return field


def makeStepOne(field, qubo_value, field_to_block: Dict[Tuple[int, int], List[int]], solver_name="DW_2000Q_VFYC_6",
                prob_of_fixing=1.,
                model_overpopulation=False, overpopulation_amnt=3):
    """
    In this method the ones are fixed with Probability prob_of_fixing
    and the rest is left free to mutate using the Qubo Value,
    Non neighbours are fixed to their values if they are zero.
    This is the growth Step as described in the paper.

    :param field: A quantum game of life field
    :param qubo_value: the qubo value with which live cells are connected to neighbouring dead cells
    :param field_to_block:  a mapping of field position to qubits in the corresponding chimera graph cluster
    :param solver_name: The D'Wave Annealer to be used
    :param prob_of_fixing: probability of fixing dead cells to remain dead if they have no alive neighbours
    :param model_overpopulation: Invert the qubo_value if the cell has more than overpopulation_amnt alive neighbours
    :param overpopulation_amnt: Amount of alive neighbouring cells after which the qubo_value should be inverted
    :return: the new quantum game of life field after the growth step
    """
    nodes, edges = getHardwareAdjacency(solver_name)
    qubo = {(i, j): 0.0 for i in range(len(nodes)) for j in range(len(nodes))}
    one_qubits = []
    # connect qubits in the Same Block
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for q_1 in field_to_block[(i, j)]:
                for q_2 in field_to_block[(i, j)]:
                    if ((q_1, q_2) in edges) or ((q_2, q_1) in edges):
                        qubo[(q_2, q_2)] += 100.0
                        qubo[(q_1, q_1)] += 100.0
                        qubo[(q_1, q_2)] += -200.0
    # Fix ones in place
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if field[i, j] == 1:
                for k in field_to_block[(i, j)]:
                    one_qubits.append(k)
                if random.random() < prob_of_fixing:
                    for k in field_to_block[(i, j)]:
                        qubo[(k, k)] += -100.
    used_qubits = set()
    # Set copling strength between connected nodes
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if field[i, j] == 1:
                amnt_used_qubits = 0
                for k_1 in range(-1, 2):
                    for k_2 in range(-1, 2):
                        pos_1 = i + k_1
                        pos_2 = j + k_2
                        if not (pos_1, pos_2) == (i, j):
                            if (pos_1 in range(0, field.shape[0])) and (pos_2 in range(0, field.shape[1])):
                                if field[pos_1, pos_2] == 1:
                                    amnt_used_qubits += 1
                for k_1 in range(-1, 2):
                    for k_2 in range(-1, 2):
                        pos_1 = i + k_1
                        pos_2 = j + k_2
                        if not (pos_1, pos_2) == (i, j):
                            if (pos_1 in range(0, field.shape[0])) and (pos_2 in range(0, field.shape[1])):
                                for q_1 in field_to_block[(i, j)]:
                                    for q_2 in field_to_block[(pos_1, pos_2)]:
                                        used_qubits.add(q_1)
                                        used_qubits.add(q_2)
                                        if (q_1, q_2) in edges:
                                            if not model_overpopulation:
                                                qubo[(q_1, q_2)] += qubo_value
                                            else:
                                                if amnt_used_qubits > overpopulation_amnt:
                                                    qubo[(q_1, q_2)] += -qubo_value
                                                    amnt_used_qubits += 1
                                                else:
                                                    qubo[(q_1, q_2)] += qubo_value
                                                    amnt_used_qubits += 1
    # Fix not connected zeros in place
    for i in range(len(nodes)):
        if i not in used_qubits:
            qubo[(i, i)] += 100.0
    new_field = np.zeros(field.shape)
    qubo = {k: v for k, v in qubo.items() if abs(v) > 0.0000001}
    # Get best solution
    result = solveWithDwave(qubo, 1, solver_name)['solutions'][0]
    # Update field from solution
    for i in range(new_field.shape[0]):
        for j in range(new_field.shape[1]):
            zero_votes = 0
            one_votes = 0
            for q in field_to_block[(i, j)]:
                if result[q] == 0:
                    zero_votes += 1
                else:
                    one_votes += 1
            if zero_votes < one_votes:
                new_field[i, j] = 1
            elif one_votes < zero_votes:
                new_field[i, j] = 0
            else:
                if random.random() < 0.5:
                    new_field[i, j] = 1
                else:
                    new_field[i, j] = 0
    return new_field


def makeStepZero(field, qubo_value, field_to_block: Dict[Tuple[int, int], List[int]], prob_of_fixing=1.,
                 solver_name="DW_2000Q_VFYC_6", model_overpopulation=False, overpopulation_amnt=3):
    """
        In this Method the zeros are fixed with Probability prob_of_fixing
        and the rest is free to mutate using the Qubo Value,
        Non neighbours are fixed to their values if they are one.
        This is the death Step as described in the paper.

        :param field: A quantum game of life field
        :param qubo_value: the qubo value with which live cells are connected to neighbouring alive cells
        :param field_to_block:  a mapping of field position to qubits in the corresponding chimera graph cluster
        :param solver_name: The D'Wave Annealer to be used
        :param prob_of_fixing: probability of fixing alive cells to remain dead if they have no dead neighbours
        :param model_overpopulation: Invert the qubo_value if the cell has more than overpopulation_amnt dead neighbours
        :param overpopulation_amnt: Amount of dead neighbouring cells after which the qubo_value should be inverted
        :return: the new quantum game of life field after the death step

    """
    inverted_field = invertField(field)
    new_inverted_field = makeStepOne(inverted_field, qubo_value, field_to_block, solver_name, prob_of_fixing,
                                     model_overpopulation=model_overpopulation, overpopulation_amnt=overpopulation_amnt)
    new_field = invertField(new_inverted_field)
    return new_field


def visualizeField(field, iteration, game_name, directory, many_games=False):
    """
        Default representation is a txt file
        Does not generate a new directory if the current one has the same name as
        the one given through the directory argument

    :param field: a quantum game of life field
    :param iteration: the current Qgol iteration in order to name the file correctly
    :param game_name: name of the game currently being played
    :param directory: output directory where the representation should be saved
    :param many_games: if a single game is being played, the directory should contain a single file
    :return: Nothing
    """
    if (not os.getcwd()[-len(directory):] == directory):
        if (not os.path.exists(directory)):
            os.mkdir(directory)
        else:
            if not many_games:
                print("This path already exists!")
                return
        os.chdir(directory)
    with open(game_name + str(iteration), mode="w") as f:
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                f.write(str(int(field[i, j])) + " ")
            f.write("\n")
    return


def playQGOL(max_iterations, qubo_value1, qubo_value0, game_name, directory,
             prob_of_fixing1=1., prob_of_fixing0=1., solver_name="DW_2000Q_VFYC_6", overpopulation_amnt=3,
             model_overpopulation=True):
    """

    Play the Quantum Game of Life for max_iterations Iterations and save each
    step played as a txt representation into the directory under the name game_name + str(current_iteration)

    :param max_iterations: The maximum amount of iterations the Qgol should be played
    :param qubo_value1: the qubo value for the growth phases
    :param qubo_value0: the qubo value for the death phases
    :param game_name: the name of the game in order to save the visualizations to a file
    :param directory: the output directory for the game
    :param prob_of_fixing1: the probability of fixing dead cells in the growth phase
    :param prob_of_fixing0: the probability of fixing dead alive in the death phase
    :param model_overpopulation: Invert the qubo_value if the cell has more than overpopulation_amnt alive neighbours
    :param overpopulation_amnt: Amount of alive neighbouring cells after which the qubo_value should be inverted
    :return: the new quantum game of life field after the growth step
    :return: None
    """
    num_qubits = len(getHardwareAdjacency(solver_name)[0])
    width, height = int(np.ceil(np.sqrt((num_qubits // 8)))), int(np.floor(np.sqrt((num_qubits // 8))))
    field_to_block = {}
    nodes, edges = getHardwareAdjacency(solver_name)
    for i in range(width):
        for j in range(height):
            field_to_block[(i, j)] = tuple(
                [(i * height + j) * 8 + k for k in range(8) if (i * height + j) * 8 + k in nodes])
    field = intializeField(height, width)

    # Custom start position
    # In this case a square starting position is given
    for i in range(3):
        for j in range(3):
            field[7 + i, 7 + j] = 1

    for iteration in range(max_iterations):
        visualizeField(field, 2 * iteration, game_name, directory, many_games=True)
        if checkIfFieldCompleteOne(field) or checkIfFieldCompleteZero(field):
            return
        field = makeStepOne(field, qubo_value1, field_to_block, solver_name, prob_of_fixing1,
                            model_overpopulation=model_overpopulation, overpopulation_amnt=overpopulation_amnt)
        visualizeField(field, 2 * iteration + 1, game_name, directory, many_games=True)
        if checkIfFieldCompleteOne(field) or checkIfFieldCompleteZero(field):
            return
        field = makeStepZero(field, qubo_value0, field_to_block, prob_of_fixing0, solver_name,
                             model_overpopulation=model_overpopulation, overpopulation_amnt=overpopulation_amnt)
    visualizeField(field, 2 * max_iterations, game_name, directory, many_games=True)
    return


if __name__ == "__main__":
    # Constants for playing the QGOL
    max_iterations = 100
    qubo_weigth_for_growth = -0.1
    qubo_weigth_for_death = -0.1
    game_name = "test-game-"
    game_directory = "square-initial"

    playQGOL(max_iterations, qubo_weigth_for_growth, qubo_weigth_for_death, game_name, game_directory)
