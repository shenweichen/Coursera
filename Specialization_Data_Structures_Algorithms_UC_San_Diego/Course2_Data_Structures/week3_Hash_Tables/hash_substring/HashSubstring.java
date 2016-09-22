import java.io.*;
import java.util.*;

public class HashSubstring {

    private static FastScanner in;
    private static PrintWriter out;



    private static Data readInput() throws IOException {
        String pattern = in.next();
        String text = in.next();
        return new Data(pattern, text);
    }

    private static void printOccurrences(List<Integer> ans) throws IOException {
        for (Integer cur : ans) {
            out.print(cur);
            out.print(" ");
        }
    }

    static long hash_func(String s, int prime, int multiplier){
        long hash = 0;
        for(int i = s.length()-1;i>-1;i--) {
            hash = (hash * multiplier + s.charAt(i)) % prime;
        }

        return hash;
    }
    static ArrayList<Long> PrecomputeHashes(String T, int lenP, int p, int x){
        int lenT = T.length();
        ArrayList<Long> H = new ArrayList<>(lenT - lenP +1);
        for(int i = 0;i<lenT-lenP+1;i++)
            H.add(new Long(0));
        String S = T.substring(lenT-lenP);
        H.set(lenT-lenP,hash_func(S,p,x)) ;
        long y = 1;
        for(int i = 0;i<lenP;i++)
            y = (y*x)%p;
        for(int i = lenT - lenP-1;i>-1;i--) {
            H.set(i, ((x * H.get(i + 1) + T.charAt(i ) - y * T.charAt(i + lenP)) % p + p) %
                    p);
        }
        return H;
    }

    static boolean AreEqual(String T, String P, int start){
        for(int i = start;i<start + P.length();i++)
            if(T.charAt(i) != P.charAt(i-start))
                return false;
        return true;
    }

    private static List<Integer> RabinKarp(Data input){
        String T = new String(input.text);
        String P = new String(input.pattern);
        int p = (int) 1e9 + 7;
        Random rand = new Random();
        int x = rand.nextInt (p-1)  +1;
        ArrayList<Integer> result = new ArrayList<>();
        long pHash = hash_func(P,p,x);
        ArrayList<Long> H = PrecomputeHashes(T,P.length(),p,x);
        for(int i = 0;i<T.length() - P.length()+1;i++){
            if(pHash != H.get(i)) {
                continue;
            }
            if(AreEqual(T,P,i)) {
                result.add(i);
            }
        }
        return result;
    }
    private static List<Integer> getOccurrences(Data input) {
        String s = input.pattern, t = input.text;
        int m = s.length(), n = t.length();
        List<Integer> occurrences = new ArrayList<Integer>();
        for (int i = 0; i + m <= n; ++i) {
	    boolean equal = true;
	    for (int j = 0; j < m; ++j) {
		if (s.charAt(j) != t.charAt(i + j)) {
		     equal = false;
 		    break;
		}
	    }
            if (equal)
                occurrences.add(i);
	}
        return occurrences;
    }

    static class Data {
        String pattern;
        String text;
        public Data(String pattern, String text) {
            this.pattern = pattern;
            this.text = text;
        }
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
    public static void main(String[] args) throws IOException {
        in = new FastScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
        //printOccurrences(getOccurrences(readInput()));
        printOccurrences(RabinKarp(readInput()));

        out.close();
    }

}

