'''

Copyright 2020 Ryan (Mohammad) Solgi

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

###############################################################################
###############################################################################
###############################################################################
from tqdm import tqdm
from typing import Optional

import numpy as np
import sys
np.random.seed(1)
###############################################################################
###############################################################################
###############################################################################

class geneticalgorithm():

    '''  Genetic Algorithm (Elitist version) for Python

    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.



    Implementation and output:

        methods:
                run(): implements the genetic algorithm

        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations

    '''
    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=10,\
                 algorithm_parameters=None,
                 progress_callback=None):


        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param dimension <integer> - the number of decision variables

        @param variable_type <string> - 'bool' if all variables are Boolean;
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)

        @param variable_boundaries <numpy array/None> - Default None; leave it
        None if variable_type is 'bool'; otherwise provide an array of tuples
        of length two as boundaries for each variable;
        the length of the array must be equal dimension. For example,
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first
        and upper boundary 200 for second variable where dimension is 2.

        @param variable_type_mixed <numpy array/None> - Default None; leave it
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first
        variable is integer but the second one is real the input is:
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1]
        in variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.

        @param function_timeout <float> - if the given function does not provide
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function.

        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of
            successive iterations without improvement. If None it is ineffective

        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm

        '''
        self.__name__=geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"
        if progress_callback is None:
            self.progress_callback = lambda x, y, z: ()
        self.progress_callback = progress_callback
        if algorithm_parameters is None:
            algorithm_parameters = {'max_num_iteration': None,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.5,\
                                       'surviving_parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':30}

        self.f = function
        #############################################################
        #dimension

        self.chromosome_size=int(dimension)

        #############################################################
        # input variable type

        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
       #############################################################
        # input variables' type (MIXED)

        if variable_type_mixed is None:

            if variable_type=='real':
                self.var_type=np.array([['real']] * self.chromosome_size)
            else:
                self.var_type=np.array([['int']] * self.chromosome_size)


        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"
            assert (len(variable_type_mixed) == self.chromosome_size), \
            "\n variable_type must have a length equal dimension."

            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"


            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries


        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':

            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"

            assert (len(variable_boundaries) == self.chromosome_size),\
            "\n variable_boundaries must have a length equal dimension"


            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two."
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]] * self.chromosome_size)

        #############################################################
        #Timeout
        self.funtimeout=float(function_timeout)

        #############################################################
        # input algorithm's parameters

        self.param=algorithm_parameters

        self.population_size=int(self.param['population_size'])

        assert (self.param['surviving_parents_portion']<=1\
                and self.param['surviving_parents_portion']>=0),\
        "parents_portion must be in range [0,1]"

        self.selected_parents_number=int(self.param['surviving_parents_portion'] * self.population_size)
        trl= self.population_size - self.selected_parents_number
        if trl % 2 != 0:
            self.selected_parents_number+=1

        self.prob_mut=self.param['mutation_probability']

        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"


        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "crossover_probability must be in range [0,1]"

        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"

        trl= self.population_size * self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.elite_individuals_number=1
        else:
            self.elite_individuals_number=int(trl)

        assert(self.selected_parents_number >= self.elite_individuals_number), \
        "\n number of parents must be greater than number of elits"

        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0, self.chromosome_size):
                if self.var_type[i]=='int':
                    self.iterate+= (self.var_bound[i][1]-self.var_bound[i][0]) * self.chromosome_size * (100 / self.population_size)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100 / self.population_size)
            self.iterate=int(self.iterate)
            if (self.iterate*self.population_size)>10000000:
                self.iterate=10000000/self.population_size
        else:
            self.iterate=int(self.param['max_num_iteration'])

        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"


        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.max_iterations_without_improvement= self.iterate + 1
        else:
            self.max_iterations_without_improvement=int(self.param['max_iteration_without_improv'])

        self.parent_selection_algorithm = 'fitness_proportionate'



    def rank_population(self, population):
        for individual in population:
            individual[self.chromosome_size] = self.f(individual[:-1])
        population_scores = population[:, self.chromosome_size]
        ##re-organize population by increasing score (minimization problem, hence best first)
        return population[population_scores.argsort()]


    def mating_probabilities(self, population_scores):
        minobj = population_scores[0]
        if minobj < 0:
            normobj = population_scores + abs(minobj)
        else:
            normobj = population_scores.copy()

        maxnorm = np.amax(normobj)
        normobj = maxnorm - normobj + 1
        # Calculate probability #of what????

        sum_normobj = np.sum(normobj)
        prob = normobj / sum_normobj
        cumprob = np.cumsum(prob)
        return cumprob

    def select_parents(self, population):
        parents = np.array([np.zeros(self.chromosome_size + 1)] * self.selected_parents_number)
        parents[0:self.elite_individuals_number] = population[0:self.elite_individuals_number].copy()
        cumulative_mating_probabilities = self.mating_probabilities(population[:, self.chromosome_size])
        for k in range(self.elite_individuals_number, self.selected_parents_number): #fitness proportionate
            index = np.searchsorted(cumulative_mating_probabilities, np.random.random())
            parents[k] = population[index].copy()

        return parents

    def pick_parent_pair(self, parent_count, ef_par):
        parent_1_index = np.random.randint(0, parent_count)
        parent_2_index = np.random.randint(0, parent_count)
        parent1 = ef_par[parent_1_index, : self.chromosome_size].copy()
        parent2 = ef_par[parent_2_index, : self.chromosome_size].copy()
        return parent1, parent2

    def generate_childrens(self, ef_par, par_count):
        for k in range(self.selected_parents_number, self.population_size, 2):
            parent1, parent2 = self.pick_parent_pair(par_count, ef_par)
            childs = self.cross(parent1, parent2, self.c_type)
            child1 = self.random_mutation(childs[0])
            yield child1
            child2 = self.halfway_gene_mutation(childs[1], parent1, parent2)
            yield child2


    def run(self):
        self.integer_genes_indexes = np.where(self.var_type == 'int')
        self.real_genes_indexes = np.where(self.var_type == 'real')

        population = np.array([np.zeros(self.chromosome_size + 1)] * self.population_size)
        for individual in population:
            for i in self.integer_genes_indexes[0]:
                individual[i] = np.random.randint(self.var_bound[i][0], self.var_bound[i][1]+1)
            for i in self.real_genes_indexes[0]:
                individual[i] = self.var_bound[i][0]+np.random.random()*(self.var_bound[i][1]-self.var_bound[i][0])

        fittest_individual = population[0][:-1].copy() #any will do.
        min_score = sys.float_info.max

        counter = 0
        for t in tqdm(range(0, self.iterate)):
            population = self.rank_population(population)

            if population[0 , self.chromosome_size] < min_score:
                counter = 0
                min_score = population[0,self.chromosome_size].copy()
                fittest_individual = population [0,: self.chromosome_size].copy()
                self.progress_callback(t, fittest_individual, min_score)
            else:
                counter +=1
                if counter > self.max_iterations_without_improvement:
                    sys.stdout.write('\nWarning: GA is terminated due to the' + \
                                     ' maximum number of iterations without improvement was met!')
                    break

            # Select parents
            parents = self.select_parents(population)
            ef_par_list = np.array([False] * self.selected_parents_number)


            ### what is this doing?
            par_count=0
            while par_count==0:
                for k in range(0, self.selected_parents_number):
                    if np.random.random() <= self.prob_cross: #selected for crossover.
                        ef_par_list[k]=True
                        par_count += 1

            ef_par = parents[ef_par_list]
            #New generation
            population[0:self.selected_parents_number] = parents
            for child, position in zip(self.generate_childrens(ef_par, par_count), range(self.selected_parents_number, self.population_size)):
                population[position] = np.append(child, 0)




        self.output_dict={'variable': fittest_individual, 'function':\
                          min_score}
        show=' '*100
        sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (fittest_individual))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (min_score))
        sys.stdout.flush()
        return fittest_individual, min_score


    def cross(self,x,y,c_type):

        ofs1=x.copy()
        ofs2=y.copy()
        if c_type=='one_point':
            ran=np.random.randint(0, self.chromosome_size)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()

        if c_type=='two_point':

            ran1=np.random.randint(0, self.chromosome_size)
            ran2=np.random.randint(ran1, self.chromosome_size)

            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()

        if c_type=='uniform':

            for i in range(0, self.chromosome_size):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy()

        return np.array([ofs1,ofs2])


    def random_mutation(self, x):

        for i in self.integer_genes_indexes[0]:
            ran=np.random.random()
            if ran < self.prob_mut:

                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)



        for i in self.real_genes_indexes[0]:
            ran=np.random.random()
            if ran < self.prob_mut:

               x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])

        return x

    def halfway_gene_mutation(self, x, p1, p2):
        for i in self.integer_genes_indexes[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)

        for i in self.real_genes_indexes[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])
        return x