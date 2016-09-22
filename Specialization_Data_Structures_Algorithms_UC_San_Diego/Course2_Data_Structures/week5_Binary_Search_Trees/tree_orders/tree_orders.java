import java.util.*;
import java.io.*;

public class tree_orders {
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

	public class TreeOrders {
		int n;
		int root;
		int[] key, left, right;
		int[] isRoot;
		
		void read() throws IOException {
			FastScanner in = new FastScanner();
			n = in.nextInt();
			key = new int[n];
			left = new int[n];
			right = new int[n];
			isRoot = new int[n];
			for (int i = 0; i < n; i++) { 
				key[i] = in.nextInt();
				left[i] = in.nextInt();
				right[i] = in.nextInt();
				if (left[i] != -1)
					isRoot[i] = 1;
				if (right[i] != -1)
					isRoot[i] = 1;
			}
			Arrays.sort(isRoot);
			root = isRoot[0];
		}
		void iter_inorder(int root, ArrayList<Integer> r) { // LNR 迭代中序遍历
			Stack<Integer> st = new Stack<>();
			int x = root;
			while (true) {
				while (x != -1) { //寻找最左下结点，沿途路径结点入栈
					st.push(x);
					x = left[x];
				}
				if (st.empty())
					break;
				x = st.peek();        //
				r.add(key[x]); //访问
				st.pop();
				x = right[x]; //转向右子树，若右子树为空，则下次循环会弹出栈顶元素继续访问
			}
		}
		void iter_preorder(int root, ArrayList<Integer> r) { // NLR 迭代先序遍历
			Stack<Integer> st = new Stack<>();
			int x = root;
			while (true) {
				while (x != -1) {      //左孩子不空持续向左下访问
					r.add(key[x]); //访问
					if (right[x] != -1)  //若右孩子不空
						st.add(right[x]); //右子树根入栈
					x = left[x];         //访问下一个左孩子
				}
				if (st.empty())
					break;
				x = st.peek();
				st.pop();
			}
		}
		void iter_postorder(int root, ArrayList<Integer> r) { // LRN  该算法还有问题。
			Stack<Integer> st = new Stack<>();
			int x = root;
			if (x != -1)
				st.push(x);
			while (!st.empty()) {
				if (left[st.peek()] != x &&
						right[st.peek()] != x) { //若栈顶非当前结点之父（则其必为右兄），
					//此时需在以其右兄为根的子树中寻找HLVFL
					while ((x = st.peek()) != -1) {
						if (left[x] != -1) { //尽可能向左
							if (right[x] != -1)
								st.push(right[x]);
							st.push(left[x]);
						} else
							st.push(right[x]);
					}
					st.pop(); //返回之前，弹出栈顶空元素
				}
				x = st.peek();
				r.add(key[x]); //访问元素
				st.pop();
			}
		}

		List<Integer> inOrder() {
			ArrayList<Integer> result = new ArrayList<Integer>();
                        // Finish the implementation
                        // You may need to add a new recursive method to do that
            iter_inorder(root,result);
			return result;
		}

		List<Integer> preOrder() {
			ArrayList<Integer> result = new ArrayList<Integer>();
                        // Finish the implementation
                        // You may need to add a new recursive method to do that
            iter_preorder(root,result);
			return result;
		}

		List<Integer> postOrder() {
			ArrayList<Integer> result = new ArrayList<Integer>();
                        // Finish the implementation
                        // You may need to add a new recursive method to do that
            iter_postorder(root,result);
			return result;
		}
	}

	static public void main(String[] args) throws IOException {
            new Thread(null, new Runnable() {
                    public void run() {
                        try {
                            new tree_orders().run();
                        } catch (IOException e) {
                        }
                    }
                }, "1", 1 << 26).start();
	}

	public void print(List<Integer> x) {
		for (Integer a : x) {
			System.out.print(a + " ");
		}
		System.out.println();
	}

	public void run() throws IOException {
		TreeOrders tree = new TreeOrders();
		tree.read();
		print(tree.inOrder());
		print(tree.preOrder());
		print(tree.postOrder());
	}
}
