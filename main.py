import numpy as np
from bitstring import Bits, BitArray
from sympy.combinatorics.graycode import gray_to_bin, bin_to_gray, random_bitstring
import math
import random
import matplotlib.pyplot as plt
import time


def f(t):
    return (1.5*t + 0.9) * math.sin(math.pi*t + 1.1)


def gray_to_uint(code):
    return Bits(bin=gray_to_bin(code)).uint


def pairs(iterable, n):
    size = len(iterable)
    count = 0

    while True:
        ids = np.random.permutation(size).tolist() + np.random.permutation(size).tolist()
        ids_len = size << 1
        for i in range(0, ids_len, 2):
            yield (iterable[ids[i]], iterable[ids[i+1]])
            count += 1
            if count >= n:
                return


class Individual:

    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness

    def num(self):
        return gray_to_uint(self.chromosome.bin)


class GA:

    def __init__(self, f, target, solution_area, accuracy, population_size=8, crossover_prob=0.7, mutation_prob=0.01,
                 generations_max=100):
        self.f = f
        self.target = target
        self.solution_area = solution_area
        self.accuracy = accuracy
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations_max = generations_max
        self.chromosome_length = self.calc_chromosome_length()

    def run(self):
        axes = plt.gca()
        axes.set_xlim(self.solution_area[0], self.solution_area[1])
        x_data = np.arange(self.solution_area[0], self.solution_area[1], 0.05)
        vfunc = np.vectorize(f, otypes=[float])
        y_data = vfunc(x_data)
        axes.plot(x_data, y_data, 'r-')
        solutions, = axes.plot([], [], 'x')
        plt.draw()

        population = self.init_population()

        for generation in range(self.generations_max):
            if self.target == 'max':
                population = sorted(population, key=lambda indiv: indiv.fitness, reverse=True)
            else:
                population = sorted(population, key=lambda indiv: indiv.fitness)

            solutions.set_xdata([self.segment_value(indiv.num()) for indiv in population])
            solutions.set_ydata([indiv.fitness for indiv in population])
            plt.draw()
            plt.pause(1e-17)
            time.sleep(0.1)

            if generation == self.generations_max - 1:
                result = self.segment_value(population[0].num()), population[0].fitness
                print("result: {0}".format(result))
                plt.plot(result[0], result[1], 'X')
                plt.show()
                return result

            parents = self.selection(population)
            children = self.crossover(parents)
            children = self.mutation(children)
            population = self.generate_new_population(population, children)

    def calc_chromosome_length(self):
        n_segments = (self.solution_area[1] - self.solution_area[0]) * (10 ** self.accuracy)
        return n_segments.bit_length()

    def rand_chromosome(self):
        return Bits(bin=random_bitstring(self.chromosome_length))

    def segment_value(self, segment_num):
        return (segment_num * (self.solution_area[1] - self.solution_area[0]) / (2 ** self.chromosome_length - 1)) \
               + self.solution_area[0]

    def calc_fitness(self, chromosome):
        num = gray_to_uint(chromosome.bin)
        x = self.segment_value(num)
        return self.f(x)

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = self.rand_chromosome()
            fitness = self.calc_fitness(chromosome)
            population.append(Individual(chromosome=chromosome,
                                         fitness=fitness))

        return population

    def selection(self, population):
        return self.tournament(population)
        # chose selection method
        # return self.ranging(population)
        # return self.roulette(population)

    def roulette(self, population):
        fitness = np.array([indiv.fitness for indiv in population])
        min_val = fitness.min()
        positive_fitness = fitness + (0 - min_val)
        fitness_sum = positive_fitness.sum()
        cells_probs = positive_fitness / fitness_sum

        ids = np.random.choice(self.population_size, size=self.population_size // 2, p=cells_probs, replace=False)
        return [population[ind] for ind in ids]

    def ranging(self, population):
        a = random.uniform(1, 2)
        b = 2 - a
        N = self.population_size
        i = np.arange(self.population_size)
        probs = (1/N) * (a-(a-b)*i/(N-1))
        ids = np.random.choice(N, size=N//2, p=probs, replace=False)
        return [population[ind] for ind in ids]

    def tournament(self, population):
        m = random.randint(2, self.population_size-1)
        parents = []
        for i in range(self.population_size // 2):
            participants_ids = random.sample(range(self.population_size), m)
            parents.append(population[min(participants_ids)])
        return parents

    def crossover(self, parents):
        children = []
        for parent1, parent2 in pairs(parents, self.population_size):
            crossover_point = random.randint(1, self.population_size-2)
            chromosome = parent1.chromosome[:crossover_point]+parent2.chromosome[crossover_point:]
            fitness = self.calc_fitness(chromosome)
            children.append(Individual(chromosome=chromosome, fitness=fitness))

        return children

    def mutation(self, children):
        for i in range(len(children)):
            if random.random() < self.mutation_prob:
                pos = random.randint(0, self.chromosome_length-1)
                chromosome = BitArray(children[i].chromosome)
                chromosome.invert(pos)
                fitness = self.calc_fitness(chromosome)
                children[i] = Individual(chromosome=chromosome, fitness=fitness)
        return children

    def generate_new_population(self, population, children):
        # uncomment to try elitism
        # new_population = population[:2] + children
        # return new_population[:self.population_size]
        return children


if __name__ == "__main__":
    ga = GA(f=f,
            target='max',
            solution_area=(0, 5),
            accuracy=3,
            population_size=16,
            generations_max=100)
    ga.run()
