# -*- coding:utf-8 -*-
import time
import threading

start = time.clock()

class Worker():
    def __init__(self):
        self.train_is_alive = 0


    def worker1(self, info):
        print('Start ' + info)
        self.train_is_alive = 1
        time.sleep(10)
        self.train_is_alive = 0
        return


    def worker2(self, info):
        print('Start ' + info)
        while(self.train_is_alive == 1):
            print('Train is alive')
            time.sleep(1)
        print('Train is not alive')
        return


if __name__ == "__main__":
    threads = []
    worker = Worker()
    threads.append(threading.Thread(target=worker.worker1, args=('worker1',)))
    threads.append(threading.Thread(target=worker.worker2, args=('worker2',)))
    print('threads1 isAlive:', threads[0].is_alive())
    print('threads2 isAlive:', threads[0].is_alive())
    for t in threads:
        t.start()
    print('threads1 isAlive:', threads[0].is_alive())
    print('threads2 isAlive:', threads[0].is_alive())

    t.join()

    end = time.clock()
    print("finished: %.3fs" % (end - start))

    time.sleep(1)
    print('threads1 isAlive:', threads[0].is_alive())
    print('threads2 isAlive:', threads[0].is_alive())
