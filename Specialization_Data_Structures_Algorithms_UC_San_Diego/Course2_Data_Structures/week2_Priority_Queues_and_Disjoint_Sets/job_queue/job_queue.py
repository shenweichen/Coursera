# python3
import heapq
import sys
class thread:
    def __init__(self,id,t=0):
        self.id = id
        self.next_free_time = t
    def addtime(self,t):
        self.next_free_time += t
    def __lt__(self,other):
        if(self.next_free_time != other.next_free_time):
            return self.next_free_time < other.next_free_time
        else:
            return self.id < other.id
class JobQueue:
    def read_data(self):
        self.num_workers, m = map(int, sys.stdin.readline().split())
        # in Python3 input() can replace sys.stdin.readline()
        self.jobs =list(map(int, sys.stdin.readline().split()))
        assert m == len(self.jobs)

    def write_response(self):
        for i in range(len(self.jobs)):
          print(self.assigned_workers[i], self.start_times[i]) 

    def assign_jobs(self):
        # TODO: replace this code with a faster algorithm.
        self.assigned_workers = [None] * len(self.jobs)
        self.start_times = [None] * len(self.jobs)
        # next_free_time = [0] * self.num_workers
        q = []#using priority queue
        for i in range(0,self.num_workers):
            q.append(thread(i))
        heapq.heapify(q)#build the heap
        for i in range(0,len(self.jobs)):
            duration = self.jobs[i]
            td = heapq.heappop(q)
            self.assigned_workers[i] = td.id
            self.start_times[i] = td.next_free_time
            td.next_free_time += duration
            heapq.heappush(q,td)
        # for i in range(len(self.jobs)):
        #   next_worker = 0
        #   for j in range(self.num_workers):
        #     if next_free_time[j] < next_free_time[next_worker]:
        #       next_worker = j
        #   self.assigned_workers[i] = next_worker
        #   self.start_times[i] = next_free_time[next_worker]
        #   next_free_time[next_worker] += self.jobs[i]

    def solve(self):
        self.read_data()
        self.assign_jobs()
        self.write_response()

if __name__ == '__main__':
    job_queue = JobQueue()
    job_queue.solve()

