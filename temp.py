# -*- coding:utf-8 -*-
import time
import threading

start = time.clock()


def worker(m):
    print('worker', m)
    time.sleep(1)
    return


if __name__ == "__main__":
    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=worker, args=(i,)))
    for t in threads:
        t.start()
        # t.join()  #阻塞子线程

    t.join()   #阻塞父线程

    end = time.clock()
    print("finished: %.3fs" % (end - start))
