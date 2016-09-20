import java.io.*;
import java.util.StringTokenizer;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;

public class JobQueue {
    private int numWorkers;
    private int[] jobs;

    private int[] assignedWorker;
    private long[] startTime;

    private FastScanner in;
    private PrintWriter out;

    public static void main(String[] args) throws IOException {
        new JobQueue().solve();
    }

    private void readData() throws IOException {
        numWorkers = in.nextInt();
        int m = in.nextInt();
        jobs = new int[m];
        for (int i = 0; i < m; ++i) {
            jobs[i] = in.nextInt();
        }
    }

    private void writeResponse() {
        for (int i = 0; i < jobs.length; ++i) {
            out.println(assignedWorker[i] + " " + startTime[i]);
        }
    }

    private void assignJobs() {
        // TODO: replace this code with a faster algorithm.
        assignedWorker = new int[jobs.length];
        startTime = new long[jobs.length];
        //long[] nextFreeTime = new long[numWorkers];
        Queue<thread> q = new PriorityQueue<thread>(11,lt);
        for (int i = 0; i < numWorkers; i++) {
            q.add (new thread(i));
        }
        for(int i = 0;i<jobs.length;i++){
            int duration = jobs[i];
            thread td = q.poll();
            assignedWorker[i] = td.id;
            startTime[i] = td.next_free_time;
            td.next_free_time+=duration;
            q.add(td);
        }
        /*for (int i = 0; i < jobs.length; i++) {
            int duration = jobs[i];
            int bestWorker = 0;
            for (int j = 0; j < numWorkers; ++j) {
                if (nextFreeTime[j] < nextFreeTime[bestWorker])
                    bestWorker = j;
            }
            assignedWorker[i] = bestWorker;
            startTime[i] = nextFreeTime[bestWorker];
            nextFreeTime[bestWorker] += duration;
        }*/
    }

    public void solve() throws IOException {
        in = new FastScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
        readData();
        assignJobs();
        writeResponse();
        out.close();
    }

    static class FastScanner {
        private BufferedReader reader;
        private StringTokenizer tokenizer;

        public FastScanner() {
            reader = new BufferedReader(new InputStreamReader(System.in));
            tokenizer = null;
        }

        public String next() throws IOException {
            while (tokenizer == null || !tokenizer.hasMoreTokens()) {
                tokenizer = new StringTokenizer(reader.readLine());
            }
            return tokenizer.nextToken();
        }

        public int nextInt() throws IOException {
            return Integer.parseInt(next());
        }
    }
    class thread{
        public int id;
        public long next_free_time;
        thread(int id,long t ){
            this.id = id;
            this.next_free_time = t;
        }
        thread(int id){
            this.id = id;
            this.next_free_time = 0;
        }
        public void addtime(long t){
            this.next_free_time += t;
        }

    }
    Comparator<thread> lt = new Comparator<thread>() {
        @Override
        public int compare(thread thread, thread t1) {
            if(thread.next_free_time == t1.next_free_time){
                if(thread.id < t1.id)
                    return -1;
                else if(thread.id > t1.id)
                    return 1;
                else
                    return 0;
            }
            else if(thread.next_free_time < t1.next_free_time)
                return -1;
            else if(thread.next_free_time > t1.next_free_time)
                return 1;
            else
                return 0;

        }
    };

}
