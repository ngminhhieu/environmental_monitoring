import concurrent.futures
import time
import multiprocessing

def do_something(i, num, return_dict):
    return_dict[i] = num * 2
    return_dict[i+50] = num*3

if __name__ == '__main__':
    start = time.perf_counter()
    # do_something(1)
    # do_something(1)
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in range(2):
        p = multiprocessing.Process(target=do_something, args=[i, 3+i, return_dict])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
    finish = time.perf_counter()
    print(return_dict[1+50])

    print(f'Finished in {round(finish-start, 2)} second(s)')

    from shutil import copyfile
    copyfile('test.py', 'test2.py')