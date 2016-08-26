#include <cstdio>

// Splay tree implementation

// Vertex of a splay tree
struct Vertex {
  int key;
  // Sum of all the keys in the subtree - remember to update
  // it after each operation that changes the tree.
  long long sum;
  Vertex *left;
  Vertex *right;
  Vertex *parent;

  Vertex(int key, long long sum, Vertex *left, Vertex *right, Vertex *parent)
      : key(key), sum(sum), left(left), right(right), parent(parent) {}
}; //构造函数

void update(Vertex *v) {
  if (v == NULL)
    return;
  v->sum = v->key + (v->left != NULL ? v->left->sum : 0ll) +
           (v->right != NULL ? v->right->sum : 0ll);
  if (v->left != NULL) {
    v->left->parent = v;
  }
  if (v->right != NULL) {
    v->right->parent = v;
  }
}

void small_rotation(Vertex *v) { //单旋操作
  Vertex *parent = v->parent;
  if (parent == NULL) { // v是根节点，直接返回
    return;
  }
  Vertex *grandparent = v->parent->parent;
  if (parent->left == v) { // v是左子结点
    Vertex *m = v->right;
    v->right = parent;
    parent->left = m;
  } else {
    Vertex *m = v->left;
    v->left = parent;
    parent->right = m;
  }
  update(parent);
  update(v);
  v->parent = grandparent;
  if (grandparent != NULL) {
    if (grandparent->left == parent) {
      grandparent->left = v;
    } else {
      grandparent->right = v;
    }
  }
}

void big_rotation(Vertex *v) { //双旋操作
  if (v->parent->left == v && v->parent->parent->left == v->parent) {
    // Zig-zig
    small_rotation(v->parent);
    small_rotation(v);
  } else if (v->parent->right == v && v->parent->parent->right == v->parent) {
    // Zig-zig
    small_rotation(v->parent);
    small_rotation(v);
  } else {
    // Zig-zag
    small_rotation(v);
    small_rotation(v);
  }
}

// Makes splay of the given vertex and makes
// it the new root.
void splay(Vertex *&root, Vertex *v) {
  if (v == NULL)
    return;
  while (v->parent != NULL) {
    if (v->parent->parent == NULL) { // v的父节点是根节点，只需单旋一次即可推出
      small_rotation(v);
      break;
    }
    big_rotation(v); // g-p-v结构，不断进行双旋
  }
  root = v;
}

// Searches for the given key in the tree with the given root
// and calls splay for the deepest visited node after that.
// If found, returns a pointer to the node with the given key.
// Otherwise, returns a pointer to the node with the smallest
// bigger key (next value in the order).
// If the key is bigger than all keys in the tree,
// returns NULL.
Vertex *find(Vertex *&root, int key) {
  Vertex *v = root;
  Vertex *last = root;
  Vertex *next = NULL;
  while (v != NULL) { // v是当前查找结点
    if (v->key >= key &&
        (next == NULL ||
         v->key < next->key)) { //这里的next上一次最小的大于ley的结点
      next = v;                 //记录比key值大的最小的结点
    }
    last = v;
    if (v->key == key) { //找到key，推出循环
      break;
    }
    if (v->key < key) { // v.key<key，在v的右子树查找
      v = v->right;
    } else { // key>v.key，在v的左子树查找
      v = v->left;
    }
  }
  splay(root, last); //查找后将最后查找结点调整至根
  return next;
}

void split(Vertex *root, int key, Vertex *&left, Vertex *&right) {
  right = find(root, key);
  splay(root, right); //将待拆分结点调整至树根
  if (right == NULL) {
    left = root;
    return;
  }
  left = right->left; //待拆分结点归并至右子树
  right->left = NULL;
  if (left != NULL) {
    left->parent = NULL; //左子树断链
  }
  update(left); //调整树高和结构
  update(right);
}

Vertex *merge(Vertex *left, Vertex *right) {
  if (left == NULL)
    return right;
  if (right == NULL)
    return left;
  Vertex *min_right = right;
  while (min_right->left != NULL) {
    min_right = min_right->left; //找到right中最小的元素
  }
  splay(right, min_right); //将最小key结点调整至根结点
  right->left = left;
  update(right);
  return right;
}

// Code that uses splay tree to solve the problem

Vertex *root = NULL;

void insert(int x) {
  Vertex *left = NULL;
  Vertex *right = NULL;
  Vertex *new_vertex = NULL;
  split(root, x, left, right);
  if (right == NULL || right->key != x) {
    new_vertex = new Vertex(x, x, NULL, NULL, NULL); //树中无x,则新建结点存储x
  }
  root = merge(merge(left, new_vertex), right);
}

bool find(int x) {
  // Implement find yourself
  Vertex *ans = find(root, x);
  if (ans != NULL && ans->key == x)
    return true;
  else
    return false;
}
void erase(int x) {
  // Implement erase yourself
  if (!root || find(x) == false) //若树空或目标关键吗不存在，则无法删除
    return;
  Vertex *w = root;         //经find后，待查找结点已被伸展至树根
  if (root->left == NULL) { //若无左子树，直接删除
    root = root->right;
    if (root)
      root->parent = NULL;

  } else if (root->right == NULL) { //若无右子树，也直接删除
    root = root->left;
    if (root)
      root->parent = NULL;
  } else { //左右子树同时存在
    Vertex *lt = root->left;
    lt->parent = NULL;
    root->left = NULL; //暂时将左子树切除
    root = root->right;
    root->parent = NULL; //只保留右子树
    bool temp = find(w->key); //以原树根为目标，做一次（必定失败的）查找
    // 至此，右子树中最小节点必伸展至根，且（因无雷同节点）其左子树必空，亍是
    root->left = lt; //只需将原左子树接回原位即可
    lt->parent = root;
  }
  delete w; //释放结点
  if (root) //若树非空，更新规模
    update(root);
}
long long sum(int from, int to) {
  Vertex *left = NULL;
  Vertex *middle = NULL;
  Vertex *right = NULL;
  split(root, from, left, middle);
  split(middle, to + 1, middle, right);
  long long ans = 0;
  // Complete the implementation of sum
  if (middle) //若不存在范围内的结点
    ans = middle->sum;
  middle = merge(middle, right);
  root = merge(left, middle);
  return ans;
}

const int MODULO = 1000000001;

int main() {
  int n;
  scanf("%d", &n);
  int last_sum_result = 0;
  for (int i = 0; i < n; i++) {
    char buffer[10];
    scanf("%s", buffer);
    char type = buffer[0];
    switch (type) {
    case '+': {
      int x;
      scanf("%d", &x);
      insert((x + last_sum_result) % MODULO);
    } break;
    case '-': {
      int x;
      scanf("%d", &x);
      erase((x + last_sum_result) % MODULO);
    } break;
    case '?': {
      int x;
      scanf("%d", &x);
      printf(find((x + last_sum_result) % MODULO) ? "Found\n" : "Not found\n");
    } break;
    case 's': {
      int l, r;
      scanf("%d %d", &l, &r);
      long long res =
          sum((l + last_sum_result) % MODULO, (r + last_sum_result) % MODULO);
      printf("%lld\n", res);
      last_sum_result = int(res % MODULO);
    }
    }
  }
  return 0;
}
