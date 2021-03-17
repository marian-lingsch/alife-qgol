# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:55:22 2020

@author: Marian
"""

from dwave.cloud import Client
import random
import os
import numpy as np
from typing import Dict, Tuple, List

def solveWithDwave(qubo, num_reads=100,solver_name="DW_2000Q_VFYC_6"):
    with Client.from_config() as client:
        solver = client.get_solver(solver_name)
        computation = solver.sample_qubo(qubo, num_reads=num_reads)
        result = computation.result()
        return result

def getHardwareAdjacency(solver_name="DW_2000Q_VFYC_6"):
    with Client.from_config() as client:
        solver = client.get_solver(solver_name)
        return (solver.nodes,solver.undirected_edges)


def invertField(field):
    inverted_field = np.zeros(field.shape)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if round(field[i,j],10) == 0:
                inverted_field[i,j] = 1
            elif round(field[i,j],10) == 1:
                inverted_field[i,j] = 0
            else:
                print("An Error occured: Some value in the fields is not valid!")
                raise ValueError
    return inverted_field

def checkIfFieldCompleteZero(field):
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if round(field[i,j],10) == 1:
                return False
    return True

def checkIfFieldCompleteOne(field):
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if round(field[i,j],10) == 0:
                return False
    return True

def intializeField(width:int, height:int, one_coverage=0.):
    field = np.zeros((height, width))
    if round(one_coverage, 10) > 0:
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if random.random() < one_coverage:
                    field[i,j] = 1
    return field

def makeStepOne(field, qubo_value, field_to_block:Dict[Tuple[int,int],List[int]], solver_name="DW_2000Q_VFYC_6", prob_of_fixing = 1.,
                model_overpopulation = False, overpopulation_amnt = 3):
    """
        In this method the ones are fixed with Probability prob_of_fixing
        and the rest is left free to mutate using the Qubo Value,
        Non neighbours are fixed to their values if they are zero
    """
    nodes, edges = getHardwareAdjacency(solver_name)
    qubo = {(i,j):0.0 for i in range(len(nodes)) for j in range(len(nodes))}
    one_qubits = []
    # connect qubits in the Same Block
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            for q_1 in field_to_block[(i,j)]:
                for q_2 in field_to_block[(i,j)]:
                    if ((q_1,q_2) in edges) or ((q_2,q_1) in edges):
                        qubo[(q_2,q_2)] += 100.0
                        qubo[(q_1,q_1)] += 100.0
                        qubo[(q_1,q_2)] += -200.0
    # Fix ones in place
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if field[i,j] == 1:
                for k in field_to_block[(i,j)]:
                    one_qubits.append(k)
                if random.random() < prob_of_fixing:
                    for k in field_to_block[(i,j)]:
                        qubo[(k,k)] += -100.
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
            qubo[(i,i)] += 100.0
    new_field = np.zeros(field.shape)
    qubo = {k: v for k, v in qubo.items() if abs(v) > 0.0000001}
    result = solveWithDwave(qubo,1,solver_name)['solutions'][0]
    for i in range(new_field.shape[0]):
        for j in range(new_field.shape[1]):
            zero_votes = 0
            one_votes = 0
            for q in field_to_block[(i,j)]:
                if result[q] == 0:
                    zero_votes += 1
                else:
                    one_votes += 1
            if zero_votes < one_votes:
                new_field[i,j] = 1
            elif one_votes < zero_votes:
                 new_field[i,j] = 0
            else:
                if random.random() < 0.5:
                    new_field[i,j] = 1
                else:
                    new_field[i,j] = 0
    return new_field

def makeStepZero(field, qubo_value, field_to_block:Dict[Tuple[int,int],List[int]],prob_of_fixing = 1. ,solver_name="DW_2000Q_VFYC_6", model_overpopulation = False):
    """
        In this Method the zeros are fixed with Probability prob_of_fixing
        and the rest is free to mutate using the Qubo Value,
        Non neighbours are fixed to their values if they are one
    """
    inverted_field = invertField(field)
    new_inverted_field = makeStepOne(inverted_field, qubo_value, field_to_block,solver_name,prob_of_fixing, model_overpopulation=model_overpopulation)
    new_field = invertField(new_inverted_field)
    return new_field

def visualizeField(field,iteration, game_name,directory,many_games=False,as_image=False):
    """
        Default representation is a txt file
        Does not generate a new directory if the current one has the same name as
        the one given through the directory argument
    """
    if (not os.getcwd()[-len(directory):] == directory):
        if (not os.path.exists(directory)):
            os.mkdir(directory)
        else:
            if not many_games:
                print("This path already exists!")
                return
        os.chdir(directory)
    if not as_image:
        with open(game_name + str(iteration),mode="w") as f:
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    f.write(str(int(field[i,j])) + " ")
                f.write("\n")
    else:
        return


def playQGOL(max_iterations, qubo_value1, qubo_value0 , game_name, directory,
             prob_of_fixing1=1.,prob_of_fixing0=1., solver_name="DW_2000Q_VFYC_6", model_overpopulation=True):
    num_qubits = len(getHardwareAdjacency(solver_name)[0])
    width, height = int(np.ceil(np.sqrt((num_qubits//8)))), int(np.floor(np.sqrt((num_qubits//8))))
    field_to_block = {}
    nodes, edges = getHardwareAdjacency(solver_name)
    for i in range(width):
        for j in range(height):
            field_to_block[(i,j)] = tuple([(i*height + j)*8 + k for k in range(8) if (i*height + j)*8 + k in nodes])
    field = intializeField(height, width)

    # Custom start position
    # In this case a square starting position is given
    for i in range(3):
        for j in range(3):
            field[7 + i, 7 + j] = 1

    for iteration in range(max_iterations):
        visualizeField(field, 2*iteration, game_name, directory, many_games=True)
        if checkIfFieldCompleteOne(field) or checkIfFieldCompleteZero(field):
            return
        field = makeStepOne(field,qubo_value1,field_to_block,solver_name,prob_of_fixing1, model_overpopulation = model_overpopulation)
        visualizeField(field, 2*iteration + 1, game_name, directory, many_games=True)
        if checkIfFieldCompleteOne(field) or checkIfFieldCompleteZero(field):
            return
        field = makeStepZero(field,qubo_value0,field_to_block,prob_of_fixing0,solver_name, model_overpopulation = model_overpopulation)
    visualizeField(field, 2*max_iterations, game_name, directory, many_games=True)
    return

if __name__ == "__main__":
    max_iterations = 100
    qubo_weigth_for_growth = -0.1
    qubo_weigth_for_death = -0.1
    game_name = "test-game-"
    game_directory = "square-initial"

    playQGOL(max_iterations,qubo_weigth_for_growth, qubo_weigth_for_death,game_name,game_directory)




