import concurrent.futures
import time
import multiprocessing

def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     secs = [5, 4, 3, 2, 1]
#     results = executor.map(do_something, secs)

#     # for result in results:
#     #     print(result)



if __name__ == '__main__':
    start = time.perf_counter()
    # do_something(1)
    # do_something(1)
    processes = []
    for _ in range(2):
        p = multiprocessing.Process(target=do_something, args=[1])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} second(s)')