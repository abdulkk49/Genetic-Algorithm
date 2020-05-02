"""
To minimize the function f(x) = a + 2b + 3c + 4d -30 using genetic algorithm.
"""

import numpy as np
#============ STEP 2 : Evaluation ==============
def eval_obj(chromosome, num_chroms): 
    # define fitness objective
    fitness_obj = abs(30 - chromosome[:,0] - 2*chromosome[:,1] -3*chromosome[:,2] -
                        4*chromosome[:,3] )
    return fitness_obj

#============ STEP 3 : Selection ==============
def select_roulette(chromosome, num_chroms, fitness_obj): 
    # calculate fitness
    fitness =  1/(1 + fitness_obj)
   
    # total fitness
    total_fitness = fitness.sum()
    
    # calculate probability
    probability = fitness/total_fitness
    
    # calculate cumulative probablity
    cum_sum = np.cumsum(probability)
    
    # np.random.seed(8) 
    # generate rabdom numbers for selection
    rand_nums = np.random.random((num_chroms))
   
    # roulette selection
    chromosome_new = np.zeros((num_chroms,4))
    for i in range(rand_nums.shape[0]):
        for j in range(chromosome.shape[0]):
            if rand_nums[i]  < cum_sum[j]:
                chromosome_new[i,:] = chromosome[j,:]
                break

    chromosome = chromosome_new
    
    return chromosome

#============ STEP 4 : Crossover ==============
def crossover(chromosome, num_chroms):
    # np.random.seed(4)
    randm = np.random.random((num_chroms))

    # defining crossover Rate
    pc = 0.25
    selector = randm < pc
    
    # selecting cross chromosomes
    cross_chromosome = chromosome[[(i == True) for i in selector]]

    len_cross_chrom = len(cross_chromosome)

    # calculating cross values
    cross_values = np.random.randint(1,3,len_cross_chrom)
    
    # copying the chromosome values for calculations
    copy_chromosome = np.zeros(cross_chromosome.shape)
    for i in range(cross_chromosome.shape[0]):
        copy_chromosome[i , :] = cross_chromosome[i , :]
    
    # cross-over calculation
    if len_cross_chrom == 1:
        cross_chromosome = cross_chromosome
    else :
        for i in range(len_cross_chrom):
            cross_val = cross_values[i]
            if i == len_cross_chrom - 1 :
                cross_chromosome[i , cross_val:] = copy_chromosome[0 , cross_val:]
            else :
                cross_chromosome[i , cross_val:] = copy_chromosome[i+1 , cross_val:]

    # copying over the crossed chromosomes to original matrix
    index_chromosome = 0
    index_newchromosome = 0
    for i in selector :
        if i == True :
            chromosome[index_chromosome, :] = cross_chromosome[index_newchromosome, :]
            index_newchromosome = index_newchromosome + 1
        index_chromosome = index_chromosome + 1
    
    return chromosome
    
#============ STEP 5 : Mutation ==============
def mutate(chromosome, num_chroms):
    # calculating the total no. of generations
    m ,n = chromosome.shape[0] ,chromosome.shape[1]
    total_gen = m*n

    # mutation rate = pm
    pm = 0.1
    no_of_mutations = int(np.round(pm * total_gen))
    
    # calculating the Generation number
    gen_num = np.random.randint(0,total_gen - 1, no_of_mutations)
    
    # calculatng a number that can replace
    replacing_num = np.random.randint(0,30, no_of_mutations)
    
    # Generating a random number which can replace the selected chromosome to be mutated   
    for i in range(no_of_mutations):
        a = gen_num[i]
        row = a//4
        col = a%4
        chromosome[row , col] = replacing_num[i]
    
    return chromosome

# Selecting the most frequent row
def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


# ===== STEP 1 : Initialization ==============
def main():
    # number of chromosomes in our population
    num_chroms = 6

    # Initialization of chromosomes 6 * 4 matrix
    # Assuming that each of a,b,c,d lie between 0 and 30
    chromosome = np.random.randint(0,30 ,(num_chroms,4))
    print("chromosomes :",chromosome)
    iteration = 0
    # run for 200 iterations
    while iteration <  200 :
        fitness_obj = eval_obj(chromosome, num_chroms)
        chromosome = select_roulette(chromosome, num_chroms, fitness_obj)
        chromosome = crossover(chromosome, num_chroms)
        chromosome = mutate(chromosome, num_chroms)
        iteration = iteration + 1
    
    print("Final Output : ", chromosome)
    print(np.squeeze(mode_rows(chromosome)))

# RUN THE PROGRAM
if __name__ == "__main__":
    main()
