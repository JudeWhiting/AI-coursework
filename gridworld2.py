import random as rd
import numpy as np
import math

def i_pop(size, chromosome): #initialize population
    pop=[]
    for inx in range(size):
        pop.append(rd.choices(range(2), k=chromosome))
        #      
    return pop

def posi(step,pos): #moves the position of the agent
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

def find_fitness(bob,goal): #moves the agent, then returns 10 - the fitness function
    i=0
    steps=[]
    temp2=[]
    pos=[0,0]
    #
    
    while i<len(bob):
        #
        temp2=[bob[i], bob[i+1]]
        pos=posi(temp2,pos)
        f=12-fitness_f(pos,goal)
        #
        #
        #
        #
        #
        i+=2
    return f

def print_fpop(f_pop): #prints out the population
    for indexp in f_pop:
        print(indexp)
    
def mating_crossover(parent_a,parent_b): #creates a child where the first few numbers are the start of parent a, and the last few are the end of parent b
    offspring=[]
    cut_point=rd.randint(1, len(parent_a) -1) #decides where parent a ends and parent b starts in relation to the offspring
    #
    offspring.append(parent_a[:cut_point] + parent_b[cut_point:])
    offspring.append(parent_b[:cut_point] + parent_a[cut_point:])
    return offspring

def mutate(chromo):  
    for idx in range(len(chromo)):
        if rd.random() < 0.1:  #this is quite high, usually it should be 0.1   #this means there's a 3/10 chance for the gene to mutate
            chromo = chromo[:idx] + [1-chromo[idx]] + chromo[idx + 1:] #if the gene does mutate, then it is flipped eg 1 becomes a zero and vice versa
    return chromo

def Roulette_wheel(pop,fitness):
    parents=[]
    fitotal=sum(fitness)
    normalized=[x/fitotal for x in fitness] #fitness of each chromosome divided by the total fitness of all chromosomes

    print('normalized fitness')
    print_fpop(normalized)
    f_cumulative=[]
    index=0
    for n_value in normalized:
        index+=n_value
        f_cumulative.append(index)

    pop_size=len(pop)
    print('cumulative fitness')
    print_fpop(f_cumulative) #cumulative fitness is used to achieve proportional selection later on
    for index2 in range(pop_size): #adds 8 parents
        rand_n=rd.uniform(0,1)
        individual_n=0
        for fitvalue in f_cumulative: #adds a chromosome to the parent list
            if(rand_n<=fitvalue):
                parents.append(pop[individual_n])
                break
            individual_n+=1
    return parents #same amount as initial population
    

psize=8 #population size
ch=12 #amount of genes in each chromosome
fgoal=[3,3]


pop=i_pop(psize,ch)
solution_found = False




for count in range(1000):


    print('population')
    print_fpop(pop)


    print('population & corresponding fitness')
    fitall=[find_fitness(indi,fgoal) for indi in pop]
    pop_fit=list(zip(pop,fitall))
    print_fpop(pop_fit)


    for m in range(len(fitall)): #checks if optimal solution has been found
        if fitall[m] == 12:
            print('OPTIMAL SOLUTION FOUND!')
            print(pop[m])
            print(fitall[m])
            solution_found = True
            break
    if solution_found == True:
        break


    parents_p=Roulette_wheel(pop,fitall) #selects parents & prints nf & cf


    print('parents')
    print_fpop(parents_p)


    print('offspring')
    off = []
    a = 0
    while a < 8:
        off += mating_crossover(parents_p[a],parents_p[a+1]) #creates 2 children
        a += 2
    print_fpop(off)


    print('apply mutation to the offspring')
    pop = []
    for b in range(8):
        pop.append(mutate(off[b]))
    print_fpop(pop)
