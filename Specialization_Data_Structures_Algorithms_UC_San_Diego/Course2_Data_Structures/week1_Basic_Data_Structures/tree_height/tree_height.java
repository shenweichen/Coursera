import java.util.*;
import java.io.*;
public class tree_height {
    class FastScanner {
		StringTokenizer tok = new StringTokenizer("");
		BufferedReader in;

		FastScanner() {
			in = new BufferedReader(new InputStreamReader(System.in));
		}

		String next() throws IOException {
			while (!tok.hasMoreElements())
				tok = new StringTokenizer(in.readLine());
			return tok.nextToken();
		}
		int nextInt() throws IOException {
			return Integer.parseInt(next());
		}
	}

	public class TreeHeight {
		int n;
		//int parent[];
		ArrayList<ArrayList<Integer>> children;
		int root;
		
		void read() throws IOException {
			FastScanner in = new FastScanner();
			n = in.nextInt();
			children = new ArrayList<ArrayList<Integer>>(n);
			for(int i = 0;i< n;i++)
				children.add(new ArrayList<Integer>());
			//parent = new int[n];
			for (int i = 0; i < n; i++) {

				int parent = in.nextInt();
				if(parent == -1)
					root = i;
				else {

					//System.out.println(children.get(parent).size());
					children.get(parent).add(i);
					//System.out.println("go");
				}
			}
		}
		int compute(int root){
			int max = 0;
			for (int i = 0;i < children.get(root).size();i++){
				max = Math.max(max,compute(children.get(root).get(i)));
			}
			return max + 1;
		}
		int computeHeight() {
                        // Replace this code with a faster implementation
			/*int maxHeight = 0;
			for (int vertex = 0; vertex < n; vertex++) {
				int height = 0;
				for (int i = vertex; i != -1; i = parent[i])
					height++;
				maxHeight = Math.max(maxHeight, height);
			}
			return maxHeight;
			*/
			return compute(root);
		}
	}

	static public void main(String[] args) throws IOException {
            new Thread(null, new Runnable() {
                    public void run() {
                        try {
                            new tree_height().run();
                        } catch (IOException e) {
                        }
                    }
                }, "1", 1 << 26).start();
	}
	public void run() throws IOException {
		TreeHeight tree = new TreeHeight();
		tree.read();
		System.out.println(tree.computeHeight());
	}
}
