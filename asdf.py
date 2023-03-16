import asyncio

def square(x):
    return x*x


async def compute_f(value):
    # Compute function f on value
    return square(value)

async def process_values(values):
    tasks = []
    for value in values:
        tasks.append(asyncio.create_task(compute_f(value)))
    results = await asyncio.gather(*tasks)
    return results





# Example usage
values = [1, 2, 3, 4, 5]
results = asyncio.run(process_values(values))

print(results)