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
        f=32-fitness_f(pos,goal)
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
        if rd.random() < 0.05:  #this is quite high, usually it should be 0.1   #this means there's a 3/10 chance for the gene to mutate
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

#
    best_fitness = [0,0]
    best_parent = [0,0]
    for i in range(pop_size):
        if fitness[i] > best_fitness[0]:

            best_fitness[1] = best_fitness[0]
            best_fitness[0] = fitness[i]

            best_parent[1] = best_parent[0]
            best_parent[0] = pop[i]

    if best_parent[1] == 0:
        best_parent[1] = best_parent[0]

    parents = []
    for i in range(int(pop_size/2)):
        parents.append(best_parent[0])
        parents.append(best_parent[1])
#
        
#
    parents = []
    
    for _ in range(len(pop)):
        random_tourney = [np.random.randint(0, pop_size) for _ in range(16)]
        winner = max(fitness[random_tourney])
        parents.append(winner)



    return parents #same amount as initial population
    

psize=128 #population size
ch=32 #amount of genes in each chromosome
fgoal=[-16,0]
generations = 0

#pop=i_pop(psize,ch)


#generates 8 chromosomes, all of which are the same as Pi which we are given in the question
pop = []
for n in range(psize):
    pop.append([1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,1,0,1,0])
solution_found = False



# keeps running until the optimal solution is found
while 1:


    print('population')
    print_fpop(pop)

    #finds and prints the fitness of each chromosome
    print('population & corresponding fitness')
    fitall=[find_fitness(indi,fgoal) for indi in pop]
    pop_fit=list(zip(pop,fitall))
    print_fpop(pop_fit)


    #checks if optimal solution has been found
    for m in range(len(fitall)): 
        if fitall[m] == 32:
            print('OPTIMAL SOLUTION FOUND!')
            print(pop[m])
            print(fitall[m])
            print('generations: ' + str(generations))
            solution_found = True
            break
    if solution_found == True:
        break

    #selects the parents for the next generation by finding the parent with the highest fitness and duplicating it eight times
    parents_p=Roulette_wheel(pop,fitall)


    print('parents')
    print_fpop(parents_p)


    print('offspring')
    generations += 1
    off = []
    a = 0
    while a < psize:
        off += mating_crossover(parents_p[a],parents_p[a+1]) #creates 2 children
        a += 2
    print_fpop(off)


    print('apply mutation to the offspring')
    pop = []
    for b in range(psize):
        pop.append(mutate(off[b]))
    print_fpop(pop)

