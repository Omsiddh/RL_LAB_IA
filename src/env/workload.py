import random


def generate_workload(seed=None, process_count_range=(4, 6),
                      burst_time_range=(1, 10), arrival_time_range=(0, 8)):

    settings = [
        ("process_count_range", process_count_range, 1),
        ("burst_time_range", burst_time_range, 1),
        ("arrival_time_range", arrival_time_range, 0),
    ]
    for name, bounds, minimum in settings:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(name + " must contain two integer bounds")
        low, high = bounds
        if type(low) is not int or type(high) is not int:
            raise ValueError(name + " must contain integer bounds")
        if low < minimum or high < low:
            raise ValueError(name + " contains invalid bounds")

    generator = random.Random(seed)
    count = generator.randint(process_count_range[0], process_count_range[1])
    workload = []
    for index in range(count):
        pid = "P" + str(index + 1)
        arrival = generator.randint(arrival_time_range[0], arrival_time_range[1])
        burst = generator.randint(burst_time_range[0], burst_time_range[1])
        workload.append((pid, arrival, burst))
    return tuple(workload)
