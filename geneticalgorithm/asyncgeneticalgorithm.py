import asyncio

from geneticalgorithm import geneticalgorithm


class asyncgeneticalgorithm(geneticalgorithm):

    def __init__(self, *, max_concurrent_tasks, function, dimension, **kwargs):
        geneticalgorithm.__init__(self, function, dimension, **kwargs)
        self.semaphore = asyncio.Semaphore(value=max_concurrent_tasks or self.population_size)

    def rank_population(self, population):
        population = asyncio.get_event_loop().run_until_complete(self._async_rank_population(population))
        population_scores = population[:, self.chromosome_size]
        return population[population_scores.argsort()]

    async def _async_rank_population(self, population):
        tasks = [asyncio.create_task(self.evaluate_one(individual)) for individual in population]
        await asyncio.gather(*tasks)
        return population

    async def evaluate_one(self, individual):
        async with self.semaphore:
            individual[self.chromosome_size] = await self.f(individual[:-1])

