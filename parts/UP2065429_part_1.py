import random as rd
import numpy as np
import math

def posi(step,pos): #moves the position of the agent, used in a loop in the find_fitness functionto help find the fitness of the chromosome
    up=[0,0]
    right=[0,1]
    down=[1,0]
    left=[1,1]
    if step==up: pos[1]+=1
    elif step==right: pos[0]+=1
    elif step==down: pos[1]-=1
    elif step==left: pos[0]-=1

    return pos

def fitness_f(pos,goal): #finds how far off the agent is from the goal
    return math.dist(goal, pos)

def find_fitness(bob,goal): #moves the agent, then returns (number of genes in each chromosome) minus the fitness function
    i=0
    steps=[]
    temp2=[]
    pos=[0,0]
    #
    
    while i<len(bob): #iterates through each gene in the chromosome
        #
        temp2=[bob[i], bob[i+1]] #stores the next two genes in the chromosome
        pos=posi(temp2,pos) #calls the posi function to move the agent
        f=32-fitness_f(pos,goal) #finds the fitness of the chromosome
        #
        #
        #
        #
        #
        i+=2
    return f #returns the final fitness value of the chromosome

def print_fpop(f_pop): #prints out the population
    for indexp in f_pop:
        print(indexp)
    
def mating_crossover(parent_a,parent_b): #creates a child where the first few numbers are the start of parent a, and the last few are the end of parent b
    offspring=[]
    cut_point=rd.randint(1, len(parent_a) -1) #decides where parent a ends and parent b starts in relation to the offspring
    #
    offspring.append(parent_a[:cut_point] + parent_b[cut_point:]) #adds the offspring to a list
    offspring.append(parent_b[:cut_point] + parent_a[cut_point:])
    return offspring

def mutate(chromo):  
    for idx in range(len(chromo)): #runs through each gene in the chromosome
        if rd.random() < 0.05:  #0.05 means there is a 5% chance of each gene mutating
            chromo = chromo[:idx] + [1-chromo[idx]] + chromo[idx + 1:] #if the gene does mutate, then it is flipped eg 1 becomes a zero and vice versa
    return chromo #returns the now mutated chromosome

def Roulette_wheel(pop,fitness): #this function selects the parents for the next generation
    parents=[]
    fitotal=sum(fitness) #sum of all fitnesses
    normalized=[x/fitotal for x in fitness] #fitness of each chromosome divided by the total fitness of all chromosomes

    pop_size=len(pop)


    parents = []
    for _ in range(pop_size): #tournament style selection
        random_tourney = [np.random.randint(0, pop_size) for _ in range(16)] #chooses 16 random chromosomes
        best_fitness = 0
        winner = []
        for i in random_tourney: #finds the chromosome with the highest fitness in the tournament
            if normalized[i] >= best_fitness:
                best_fitness = normalized[i]
                winner = pop[i]
        parents.append(winner) #adds the winner to the list of parents



    return parents #returns the list of parents
    

psize=128 #population size
ch=32 #amount of genes in each chromosome
fgoal=[-16,0] #end goal (to get to the end goal the agent must have a chromosome consisting of all 1s)
generations = 0 #keeps track of how many generations there have been

#generates 32 chromosomes, all of which are the same as Pi which we are given in the question
pop = []
for n in range(psize):
    pop.append([1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,1,0,1,0])
solution_found = False

#we could also use the function below to generate the initial population randomly, but we will not use it in this case (note that the code will still work fine if you generate the pop this way)
def i_pop(size, chromosome):
    pop=[]
    for inx in range(size):
        pop.append(rd.choices(range(2), k=chromosome))
    return pop
#pop=i_pop(psize,ch)  #if you uncomment this line, then you can use this function to generate the initial population randomly

# keeps running until the optimal solution is found
while 1:

    #finds and prints the fitness of each chromosome
    print('population & corresponding fitness')
    fitall=[find_fitness(indi,fgoal) for indi in pop]
    pop_fit=list(zip(pop,fitall))
    print_fpop(pop_fit)


    #checks if optimal solution has been found
    for m in range(len(fitall)): 
        if fitall[m] == 32: #if the fitness of a chromosome is 32, then it must be the optimal solution (all 1s)
            print('OPTIMAL SOLUTION FOUND!')
            print(pop[m])
            print(fitall[m])
            print('generations: ' + str(generations))
            solution_found = True
            break
    if solution_found == True:
        break

    #finds the parents for the next generation by calling the Roulette_wheel function
    parents_p=Roulette_wheel(pop,fitall)


    print('parents')
    print_fpop(parents_p)


    print('offspring')
    generations += 1
    off = []
    a = 0
    while a < psize: #runs through each pair of parents
        off += mating_crossover(parents_p[a],parents_p[a+1]) #creates 2 children
        a += 2
    print_fpop(off)


    print('apply mutation to the offspring')
    pop = []
    for b in range(psize): #applies mutation to each child
        pop.append(mutate(off[b]))
    print_fpop(pop)

