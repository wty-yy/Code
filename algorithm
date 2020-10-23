#### 并查集 Union_Find
```c++
int fa[N];
int getfa(int x) {
    if (fa[x] == x) return x;
    fa[x] = getfa(fa[x]);
    return fa[x];
}
```
#### 生成树 Kruskal
```c++
struct Edge{int u, v, w;} e[M];
bool cmp(Edge a, Edge b) {return a.w > b.w;}
int tot;
void kruskal() {
    sort(e + 1, e + 1 + tot, cmp);
    for (int i = 1; i <= tot; i++) {
        int u = e[i].u, v = e[i].v;
        double w = e[i].w;
        if (w < 10 || w > 1000) continue;
        if (getfa(u) != getfa(v)) {
            ans += w * 100;
            fa[getfa(u)] = getfa(v);
        }
    }
}
```
### LCA
#### 倍增
```c++
//1<<18 = 262144
//1<<19 = 524288
//如果总的点数为1e6就选择19就好了
//1<<20 = 1048576
int dep[N], pos[N][20];
inline void dfs(int u, int fa) {
    for (int i = 1; i <= 18; i++) {
        if (dep[u] < (1 << i)) break;
        pos[u][i] = pos[pos[u][i - 1]][i - 1];
    }
    for (int i = head[u]; i; i = e[i].nt) {
        int v = e[i].b;
        if (v == fa) continue;
        pos[v][0] = u;
        dep[v] = dep[u] + 1;
        dfs(v, u);
    }
}
int lca(int x, int y) {
    if (dep[x] < dep[y]) swap(x, y);
    int t = dep[x] - dep[y];
    for (int i = 0; i <= 18 && t; i++, t >>= 1) if (t & 1) x = pos[x][i];
    if (x == y) return x;
    for (int i = 18; i >= 0; i--) {
        if (pos[x][i] != pos[y][i]) {
            x = pos[x][i];
            y = pos[y][i];
        }
    }
    return pos[x][0];
}
```
#### Tarjan
```c++
int a_num, ahead[N], ans[N];
bool vis[N];
struct Ask {int b, nt, id;} A[M];
void aask(int u, int v, int id) {
    A[++a_num].b = v;
    A[a_num].nt = ahead[u];
    A[a_num].id = id;
    ahead[u] = a_num;
}
void tarjan(int u, int prt) {
    for (int i = head[u]; i; i = e[i].nt) {
        int v = e[i].b;
        if (v == prt) continue;
        tarjan(v, u);
        vis[v] = 1;
        fa[v] = getfa(u);
    }
    for (int i = ahead[u]; i; i = A[i].nt) {
        int v = A[i].b;
        if (vis[v]) ans[A[i].id] = getfa(v);
    }
}
```
#### 树链剖分
详细树链剖分代码在下文
通过跳转重链的顶端，最终在一条链上的时候，位于dep小的节点就是LCA
### ST表-RMQ
```c++
//与倍增原理相似，只需要把点权左移就好了
int n, mx[N][16], lg[N];
void query(int x, int y) {
    --x; //所有的点权向左释放到边权上，所以x左移一个
    int t = lg[y - x]; x += (1 << t);
    printf("%d\n", max(mx[x][t], mx[y][t])); //利用并集O(1)求解

}
void make_ST() {
    read(n);
    for (int i = 1; i <= n; i++) {
        lg[i] = lg[i - 1] + (1 << (lg[i - 1] + 1) == i); //预处理,log2(i)向下取整
        read(mx[i][0]);
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= 15; j++) {
            if (i < (1 << j)) break;
            mx[i][j] = max(mx[i][j - 1], mx[i - (1 << (j - 1))][j - 1]);
        }
    }
}
```
### 树状数组
#### 基础
```c++
int c[N];
void update(int x, int t) {
    for (; x <= n; x += x & (-x)) c[x] += t;
}
int query(int x) {
    int ret = 0;
    for (; x; x -= x & (-x)) ret += c[x];
    return ret;
}
```
#### 求逆序对
```c++
///hsh[]为离散后的对应的编号
int inversion() {
	for (int i = 1; i <= n; i++) {
		update(hsh[i], 1);
		ans += i - query(hsh[i]);
	} 
}
```
#### 区间修改-区间查询(差分)

令A[]为原数组c1[]为差分数组$c1[i]=A[i]-A[i-1](A[0]=0)$，所以$A[j]=\sum\limits_{i=1}^jc1[i]$，对c1查询能很容易做到单点查询，修改区间[l,r]只需将$c1[l]+1,c1[r+1]-1$就行。

区间查询（统计下每个$c1[j]$出现了多少次）$\sum\limits_{i=1}^{p}\sum\limits_{j=1}^ic1[j]=c1[1]\cdot p+c1[2]\cdot (p-1)+c1[3]\cdot (p-2)\cdots\cdots=\sum\limits_{i=1}^pc1[i]\cdot (p-i+1)$

$=\sum\limits_{i=1}^pc1[i]\cdot (p+1)-\sum\limits_{i=1}^pc1[i]\cdot i$由于$(p+1)$为常数，只需要维护$c1[i]$和$c2[i]=c1[i]\cdot i$的树状数组即可

```c++
void update(int x, int t) {
    for (int i = x; i <= n; i += i & (-i)) c1[i] += t, c2[i] += t * x;
}
int query(int x) {
    int ret = 0;
    for (int i = x; i; i -= i & (-i)) ret += c1[i] * (x + 1) - c2[i];
    return ret;
}
void build() {
    int mem = 0;
    for (int i = 1; i <= n; i++) update(i, val[i] - mem), mem = val[i];
}
//更新区间[l,r]
update(l, k); update(r + 1, -k);
//求区间和[l,r]
query(r) - query(l - 1)
```

### 二维树状数组
#### 单点修改，区间查询

query(x,y)能求出左上角为(1,1)右下角为(x,y)矩形内数值之和，记为$sum[x][y]$

```c++
void update(int x, int y, int t) {
    for (int i = x; i <= n; i += i & (-i))
    for (int j = y; j <= m; j += j & (-j))
        c[i][j] += t;
}
int query(int x, int y) {
    int ret = 0;
    for (int i = x; i; i -= i & (-i))
    for (int j = y; j; j -= j & (-j))
        ret += c[i][j];
    return ret;
}
(a,b)为左上角(c,d)为右下角，ans=query(c,d)-query(a-1,d)-query(c,b-1)+query(a-1,b-1)
```
#### 区间修改，单点查询（差分）
令A为原数组，c为差分数组，直接求解$sum[x][y]=sum[x-1][y]+sum[x][y-1]-sum[x-1][y-1]+A[x][y]$

若要使$\sum\limits_{i=1}^x\sum\limits_{j=1}^yc[i][j]=A[x][y]​$则$c[x][y]=A[x][y]-(A[x-1][y]+A[x][y-1]-A[x-1][y-1])​$

所以$\sum\limits_{i=1}^x\sum\limits_{j=1}^yc[i][j]=sum[x][y]-(sum[x-1][y]+sum[x][y-1]-sum[x-1][y-1])=A[x][y]$
```c++
区间修改方法
初始时
0	0	0	0
0	0	0	0
0	0	0	0
0	0	0	0
差分数组修改方法
0	0	0	0
0	k	0  -k
0	0	0	0
0  -k   0   k
原数组变化效果
0	0	0	0
0	k	k	0
0	k	k	0
0	0	0	0
左上角为(x,y)右下角为(a,b)的矩形区域都加上k
update(x, y, k);
update(x, b + 1, -k);
update(a + 1, y, -k);
update(a + 1, b + 1, k);
```
#### 区间修改，区间查询
和查询一维区间一样，$sum[x][y]=\sum\limits_{k=1}^x\sum\limits_{h=1}^y\sum\limits_{i=1}^k\sum\limits_{j=1}^hc[i][j]$，和一维一样，可以统计一下每个$c[i][j]$各出现了多少次

$=c[1][1]\cdot xy+c[1][2]\cdot x(y-1)+c[1][3]\cdot x(y-2)\cdots+c[2][1]\cdot (x-1)y+c[3][1]\cdot (x-2)y\cdots$

$=\sum\limits_{i=1}^x\sum\limits_{j=1}^yc[i][j]\cdot (x-i+1)(y-j+1)=\sum\limits_{i=1}^x\sum\limits_{j=1}^yc[i][j]\cdot ((x+1)(y+1)-i(y+1)-j(x+1)+ij)$

所以需要维护四个树状数组分别是$c1[i][j]=c[i][j],c2[i][j]=c[i][j]\cdot i,c3[i][j]=c[i][j]\cdot j,c4[i][j]=c[i][j]\cdot ij$
```c++
void update(int x, int y, int t) {
    for (int i = x; i <= n; i += i & (-i))
    for (int j = y; j <= m; j += j & (-j)) {
        c1[i][j] += t;
        c2[i][j] += t * x;
        c3[i][j] += t * y;
        c4[i][j] += t * x * y;
    }
}
int query(int x, int y) {
    int ret = 0;
    for (int i = x; i; i -= i & (-i))
    for (int j = y; j; j -= j & (-j))
        ret += c1[i][j] * (x + 1) * (y + 1) - c2[i][j] * (y + 1) - c3[i][j] * (x + 1) + c4[i][j];
    return ret;
}
//更新和查询方法和上述两篇代码一样
```

### 线段树

#### 基础
```c++
//以区间加，区间查询为例
struct node {int l, r, tag, sum;};
void build(int p, int l, int r) {
	//重置新开的点
    c[p].l = l, c[p].r = r;
    c[p].sum = 0, c[p].tag = 0;
    if (l == r) {read(c[p].sum); return;}
    int m = (l + r) >> 1;
    build(p << 1, l, m);
    build(p << 1 | 1, m + 1, r);
    //标记上传
    pushup(p);
}
void add(int p, int l, int r, int k) {
    if (c[p].l == l && c[p].r == r) {
    	//更新该区间
        c[p].sum += (r - l + 1) * k;
        c[p].tag += k;
        return;
    }
    //标记下移
    pushdown(p);
    //分区段讨论
    int m = (c[p].l + c[p].r) >> 1;
    if (r <= m) add(p << 1, l, r, k);
    else if (l > m) add(p << 1 | 1, l, r, k);
    else {
        add(p << 1, l, m, k);
        add(p << 1 | 1, m + 1, r, k);
    }
    pushup(p);
}
//此处还有一种不用讨论的写法
void add(int p, int l, int r, int k) {
    int L = c[p].l, R = c[p].r;
    if (L > r || R < r) return;//在此处进行判断目标区间是否超出当前节点的区间范围
    if (L >= l && R <= r) {
        c[p].sum += (R - L + 1) * k;
        c[p].addv += k;
        return;
    }
    pushdown(p);
    add(p << 1, l, r, k);
    add(p << 1 | 1, l, r, k);
    pushup(p);
}
//还可以这样写
void add(int p, int l, int r, int k) {
    int L = c[p].l, R = c[p].r;
    if (L >= l && R <= r) {
        c[p].sum += (R - L + 1) * k;
        c[p].addv += k;
        return;
    }
    pushdown(p);
    int m = (c[p].l + c[p].r) >> 1;
    if (l <= m) add(p << 1, l, r, k);//这说明左儿子中一定有目标区间的一部分
    if (r > m) add(p << 1 | 1, l, r, k);//同理这说明右儿子中有目标区间的一部分
    pushup(p);
}
int query(int p, int l, int r) {
    if (c[p].l == l && c[p].r == r) return c[p].sum;
    pushdown(p);
    int m = (c[p].l + c[p].r) >> 1;
    if (r <= m) return query(p << 1, l, r);
    else if (l > m) return query(p << 1 | 1, l, r);
    else return query(p << 1, l, m) + query(p << 1 | 1, m + 1, r);
}
```
#### 可持久化线段树（主席树）
求数列某一区间的第k大/小值，先进行离散化，再对数列从左往右依次向线段树中插入数值，每个点对应一个线段树，利用动态开点线段树将优化内存，求某个[l,r]中间数值的树就是用第r个减去第l-1个线段树就是他们中间的值。图解：<https://blog.csdn.net/bestFy/article/details/78650360>
```c++
int build(int l, int r) {
    int p = ++cnt; c[p].sum = 0;
    if (l < r) {
        c[p].l = build(l, mid);
        c[p].r = build(mid + 1, r);
    }
    return p;
}
int update(int pre, int l, int r, int x) {
    int p = ++cnt;
    c[p].l = c[pre].l, c[p].r = c[pre].r, c[p].sum = c[pre].sum + 1;
    if (l < r) {
        if (x <= mid) c[p].l = update(c[pre].l, l, mid, x);
        else c[p].r = update(c[pre].r, mid + 1, r, x);
    }
    return p;
}
int query(int u, int v, int l, int r, int k) {
    if (l == r) return l;
    int x = c[c[v].l].sum - c[c[u].l].sum;
    if (x >= k) return query(c[u].l, c[v].l, l, mid, k);
    else return query(c[u].r, c[v].r, mid + 1, r, k - x);
}
T[0] = build(1, tot);//tot为离散后的元素个数
for (int i = 1; i <= m; i++) T[i] = update(T[i - 1], 1, tot, hsh[i]);
ans = A[query(T[l - 1], T[r], 1, tot, k)].w;//query求出的是离散后的编号，还要对应回原数值
```
#### 扫描线
##### 求面积并
思路：https://blog.csdn.net/xianpingping/article/details/83032798
注意：内存的计算，有N个矩形，则横坐标的值最多可能有2N个，所以线段树开8N，还注意pushup()操作中，叶子节点可能会导致越界访问。
```c++
const int N = 1e5 + 10;
int n, ans;
int H[N << 1], tot;
struct Node {int l, r, h, fg;} A[N << 1];
bool cmp(Node a, Node b) {return a.h < b.h;}
struct Tree{int l, r, sum, len;};
struct Seg {
    Tree c[N << 3];
    void pushup(int p) {
        if (c[p].sum) c[p].len = H[c[p].r + 1] - H[c[p].l];
        else if (c[p].l == c[p].r) c[p].len = 0;
        //如果p为叶子节点，那么p最大可能是8N(2N*4)。如果再*2的话，就会导致越界访问
        else c[p].len = c[p << 1].len + c[p << 1 | 1].len;
    }
    void build(int p, int l, int r) {
        c[p].l = l, c[p].r = r, c[p].sum = c[p].len = 0;
        if (l == r) return;
        build(p << 1, l, mid);
        build(p << 1 | 1, mid + 1, r);
    }
    void add(int p, int l, int r, int k) {
        int ll = c[p].l, rr = c[p].r;
        if (r >= rr && l <= ll) {
            c[p].sum += k;
            pushup(p);
            return;
        }
        if (l <= mid) add(p << 1, l, r, k);
        if (r > mid) add(p << 1 | 1, l, r, k);
        pushup(p);
    }
}seg;
signed main() {
    read(n);
    for (int i = 1; i <= n; i++) {
        int x1, y1, x2, y2; read(x1), read(y1), read(x2), read(y2);
        A[i * 2 - 1] = (Node){x1, x2, y1, 1};
        A[i * 2] = (Node){x1, x2, y2, -1};
        H[i * 2 - 1] = x1, H[i * 2] = x2;
    }
    n <<= 1;
    sort(H + 1, H + n + 1);
    sort(A + 1, A + n + 1, cmp);
    tot = unique(H + 1, H + n + 1) - H - 1;
    seg.build(1, 1, tot - 1);
    for (int i = 1; i < n; i++) {
    	int l = lower_bound(H + 1, H + 1 + tot, A[i].l) - H;
    	//线段树中区间[l,r]代表的实际区间为[H[l],H[r+1]]，所以此处r要-1
    	int r = lower_bound(H + 1, H + 1 + tot, A[i].r) - H - 1;
        //不用加入第n条扫描线，因为第n条扫描线上方没有面积了
        seg.add(1, l, r, A[i].fg);
        ans += seg.c[1].len * (A[i + 1].h - A[i].h);
    }
    printf("%lld", ans);
    return 0;
}
```
### 最短路
#### dijkstra
解决非负权图的最短路，贪心思路，从每次从距离出发点最近的点向外延伸，更新其他点的距离，直到所有点都被访问过
```c++
struct Node {
    int id, dis;
    Node(int id, int dis) {this->id = id, this->dis = dis;}
    friend bool operator < (Node a, Node b) {return a.dis > b.dis;}
};
priority_queue<Node> q;
int dis[N];
bool vis[N];
void dijkstra(int st) {
    for (int i = 1; i <= n; i++) dis[i] = MAX;
    dis[st] = 0;
    q.push(Node(st, 0));
    while (!q.empty()) {
        Node t = q.top(); q.pop();
        int u = t.id;
        if (vis[u]) continue;
        vis[u] = 1;
        for (int i = head[u]; i; i = e[i].nt) {
            int v = e[i].b, w = e[i].w;
            if (dis[v] - dis[u] > w) {
                dis[v] = dis[u] + w;
                q.push(Node(v, dis[v]));
            }
        }
    }
}
```
#### SPFA
与dijkstra不同的是，spfa用的是普通队列，每个点可能进入队列多次，vis数组用于记录该点是否在队列之中，其他大致一样。
```c++
struct Node {
    int id, dis;
    Node(int id, int dis) {this->id = id, this->dis = dis;}
};
queue<Node> q;
int dis[N];
bool vis[N];
void spfa(int st) {
    for (int i = 1; i <= n; i++) dis[i] = MAX;
    dis[st] = 0;
    q.push(Node(st, 0));
    while (!q.empty()) {
        Node t = q.front(); q.pop();
        int u = t.id;
        vis[u] = 0;//从队列中弹出
        for (int i = head[u]; i; i = e[i].nt) {
            int v = e[i].b, w = e[i].w;
            if (dis[v] - dis[u] > w) {
                dis[v] = dis[u] + w;
                if (!vis[v]) {//如果不在队列中
                    vis[v] = 1;
                    q.push(Node(v, dis[v]));
                }
            }
        }
    }
}
```
#### Floyd
用于计算完全图中任意两点间的距离，复杂度$O(n^3)$，利用中间点k来作为$i，j$两个点距离的桥梁$dis[i][j]=min(dis[i][k]+dis[k][j])$
对k的理解：如果k=1时只用了编号为1的点作为中转，如果k=2时则用了1,2两个点作为中转……其实就是：从i号点到j号点只经过前k号点的最短路径
练习：[P1119 灾后重建](https://www.luogu.com.cn/problem/P1119)
```c++
int dis[N][N];
void reset() {
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            if (i == j) dis[i][j] = 0;
            else dis[i][j] = INF;    
}
void floyd() {
    for (int k = 1; k <= n; k++)
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                Min(dis[i][j], dis[i][k] + dis[k][j]);
}
```
#### K短路
##### A*算法
在BFS搜索中，第一次到达终点的是最短路径，那么第k次到达终点的就是k短路了。
如果直接暴力BFS，搜索范围非常大，要通过剪枝处理。因此使用启发式函数f（预估最有希望到达终点的点），$f=x+h$（x为当前点实际走的距离，h为从当前点到达终点预估的距离），每次使用f值最小的点就可做到剪枝的效果了。
如何求出h呢？用预处理的方法，通过反向建图，从终点反向跑出到每个点的最短路（dis数组）这个就是h。
下一步只需要将**起点到当前点的距离与当前点到终点的最短距离的和**作为关键字放入优先队列中，这样每次弹出的就是最有希望到达终点的那个点。
https://blog.csdn.net/qq_40772692/article/details/82530467
https://www.cnblogs.com/Paul-Guderian/p/6812255.html

```c++
struct Kth {
    int id, use;
    Kth(int id, int use) {this->id = id, this->use = use;}
    friend bool operator < (Kth a, Kth b) {return a.use + dis[a.id] > b.use + dis[b.id];}
};
int bfs(int st, int en, int k) {
    priority_queue<Kth> q;
    q.push(Kth(st, 0));
    while (!q.empty()) {
        Kth t = q.top(); q.pop();
        int u = t.id;
        if (u == en) if (--k == 0) return t.use;
        for (int i = head[u]; i; i = e[i].nt) {
            int v = e[i].b, w = e[i].w;
            q.push(Kth(v, t.use + w));
        }
    }
    return -1;
}
//用spfa跑反向最短路比dijkstra使用的空间小很多
```
### Tarjan算法
#### 定义
**强联通**：在一个有向图中，任意两个点可以互相到达。
**割点集合**：在一个无向图中，如果一个顶点集合$V$，删除顶点集合$V$以及$V$中顶点相连的所有边后，原图不再连通，此点集$V$称为割点集合。
**点连通度**：在一个无向图中，最小割点集合中的顶点数。
**割边集合**：在一个无向图中，如何一个边集合，删除这个边集合后，原图不再连通，这个集合就是割边集合。
**边连通度**：在一个无向图中，最小割边集合中的边数。
**双连通图**：一个无向图中，点连通度大于1，则该图是点双连通，边连通度大于1，则该图是边双连通，它们都被简称为双连通或重连通（即删去任意一点后原图仍然连通），但不完全等价。
**割点**：当且仅当该无向图的点连通度为1时，割点集合的唯一元素被称为割点，一个图中可能有多个割点。
**桥**：当且仅当该无向图的边连通度为1时，割边集合的唯一元素被称为桥，一个图中可能有多个桥。
两个猜想：两个割点之间一定是桥，桥的两个端点一定是割点。**两个猜想都是错的！**
如下图，左图两个割点之间不是桥，右图中一个桥两边不是割点。
<img src="D:\yy\program\Typora\Photos\Tarjan.jpg" style="zoom:10%">

#### 强联通分量（缩点)
先通过Tarjan算法计算出一个强连通分量，然后都给他们染上相同的颜色，再通过枚举两两点之间它们的颜色是否一样，如果不同就连起来，就可以构建新图B了（A是原图）
图解：http://keyblog.cn/article-72.html
```c++
//Edge------------------------------------------------------------------
struct Edge {int b, nt;};
struct Union {
    int head[N], e_num;
    Edge e[M];
    void aedge(int u, int v) {
        e[++e_num].b = v;
        e[e_num].nt = head[u];
        head[u] = e_num;
    }
}A, B;
//----------------------------------------------------------------------
int cnt, idx[N], sz[N];//染色

//tarjan所需变量
int tot, dfn[N], low[N];
int st[N], top;//栈
bool vis[N];//判断是否在栈中

void tarjan(int u) {
    dfn[u] = low[u] = ++tot;
    vis[u] = 1;
    st[++top] = u;
    for (int i = A.head[u]; i; i = A.e[i].nt) {
        int v = A.e[i].b;
        if (!dfn[v]) {tarjan(v); Min(low[u], low[v]);}
        else if (vis[v]) Min(low[u], dfn[v]);
        //此处我认为可以写成Min(low[u], low[v]);
    }
    if (low[u] != dfn[u]) return;
    ++cnt;//新的一种颜色
    do {
        idx[st[top]] = cnt;//标记颜色
        vis[st[top]] = 0;//从栈中弹出
        sz[cnt] += val[st[top]];//点权之和
    } while(st[top--] != u);
}
//主函数中的操作-----------------------------------------------------------
for (int i = 1; i <= n; i++) if (!dfn[i]) tarjan(i);
//构造新图B
for (int u = 1; u <= n; u++) {
    for (int i = A.head[u]; i; i = A.e[i].nt) {
        int v = A.e[i].b;
        if (idx[v] != idx[u]) B.aedge(idx[u], idx[v]);
    }
}

```
#### 割点
判断u是割点的条件，满足下述两个条件之一：
1、u为搜索树的树根，且u有多于一个子树。
2、u不为树根，且满足u为v在搜索树中的父亲，并且$dfn[u]\le low[v]$。删去u后v的子树无法到达u的祖先。
https://zhuanlan.zhihu.com/p/101923309

```c++
int tot, dfn[N], low[N], rt;
bool cut[N];
void tarjan(int u) {
    dfn[u] = low[u] = ++tot;
    int cnt = 0;//记录子树个数
    for (int i = head[u]; i; i = e[i].nt) {
        int v = e[i].b;
        if (!dfn[v]) {
            ++cnt;//子树+1
            tarjan(v);
            Min(low[u], low[v]);
            //对应上述两个判断条件
            if ((rt == u && cnt > 1) || (rt != u && dfn[u] <= low[v])) cut[u] = 1;
        //由于是无向图，不同讨论是否在栈中
        } else Min(low[u], dfn[v]);
    }
}
for (int i = 1; i <= n; i++) if (!dfn[i]) rt = i, tarjan(i);
```
##### 割点周围连通图的个数
简单说就是去掉割点后，会产生几个连通图
cnt代表u点去掉后连通子图的个数，如果时u=rt时，则rt一定会在u去掉后的一个连通子图中，所以cnt开始为1，每次$dfn[u]<=low[v]$表示v到达最上方的点就是u，则v下方的一系列点都在u去掉后的一个连通子图中。
[电力](https://loj.ac/problem/10103)

```c++
void tarjan(int u) {
    dfn[u] = low[u] = ++tot;
    int cnt = u != rt;//如果不是rt的话，rt则一定在u点去掉后的一个连通子图中，则至少有1个
    for (int i = head[u]; i; i = e[i].nt) {
        int v = e[i].b;
        if (!dfn[v]) {
            tarjan(v), Min(low[u], low[v]);
            if (dfn[u] <= low[v]) mem[u] = ++cnt;//当v到达的最小的点是u，则v属于一个连通子图中
        }
        else Min(low[u], dfn[v]);
    }
}
```
于是我们又发现一个新的更好的方法判断割点。
```c++
//修改下这个判断
if (dfn[u] <= low[v] && ++cnt > 1) cut[u] = 1;
//很好理解，如果u点去掉后，连通子图个数为2个以上，则u就是一个割点
```
#### 桥
判断一条无向边(u,v)是桥，当且仅当(u,v)是树枝边，且满足$dfn[u]<low[v]$（因为v要到达u的节点则需经过(u,v)这条边，所以删去这条边，图不连通）注意：不要走相同的道路。
注意：桥和割点判断条件的位置是不相同的
https://zhuanlan.zhihu.com/p/101923309

```c++
int tot, dfn[N], low[N];
bool cut[M << 1], vis[M << 1];
void tarjan(int u) {
    dfn[u] = low[u] = ++tot;
    for (int i = head[u]; i; i = e[i].nt) if (!vis[i]) {
        vis[i] = vis[i ^ 1] = 1;
        int v = e[i].b;
        if (!dfn[v]) {
            tarjan(v);
            Min(low[u], low[v]);
        //由于是无向图，不用讨论是否在栈中
        } else Min(low[u], dfn[v]);
        //判断是否是桥
        if (dfn[u] < low[v]) cut[i] = cut[i ^ 1] = 1;
    }
}
for (int i = 1; i <= n; i++) if (!dfn[i]) rt = i, tarjan(i);
```
#### 2-SAT
https://www.luogu.com.cn/problem/solution/P4782
建图方法：建立有向图，使连接边均为必要条件。
例：a和a’，b和b‘中都有且仅有一个变量为true（a,a',b,b'均为bool型变量）
eg1：条件：a和b'不能同时成立。建边：a$\rightarrow$b，b'$\rightarrow$a’（a如果成立的话，则b'不能成立，则b一定要成立，另一条边同理）
eg2：条件：a和b'至少成立一个。建边：a'$\rightarrow$b'，b$\rightarrow$a（如果a没有成立的话，a'一定要成立，则b'也一定要成立，另一条边同理）
如果a和a’在同一个强连通分量中，这说明a和a'必须要同时成立，这是不可能的，所以无解。
反之，a和a'如果有前后关系的，如a$\rightarrow$a'，则此时a'为true，a为false，因为如果a为true则a'也要为true。所以用tarjan中的染色序号，序号小的就是箭头右边的，大的就是左边的，十分容易判断出谁是true谁是false。又因为a和a'可能没有任何的联系(没有边连接)，所以它们中间可以任意取值，就产生了多解。
```c++
//根据题意建好有向边
for (int i = 1; i <= n; i++)
    if (idx[i] == idx[get(i)]) { //get(x)用于获取x'的序号
        printf("IMPOSSIBLE");
        return 0;
    }
for (int i = 1; i <= n; i++) {
    if (idx[i] < idx[get(i)]) printf("%d\n", i);
    else printf("%d\n", get(i));
}

```
### 树链剖分
[树链剖分](https://www.cnblogs.com/ivanovcraft/p/9019090.html)目的是把一个整体的树，差分成很多条重链与轻链，重链与轻链上的dfs序是连续的，所以可以做到区间修改(用其他数据结构很容易维护，线段树，树状数组……)，从而降低时间复杂度理论复杂度为$O(nlog^2n)$

#### 预处理
预处理包含，两次dfs，从树根出发。
dfs1：计算子树大小sz[]，求出重孩子节点son[]，记录深度dep[]，记录父亲节点prt[]。
dfs2：做出重链，记录重链中的顶点（重链中深度最小的就是最上方的一个节点）top[]，树上每个节点在线段树中对应的编号idx[]，反对应编号H[]。

```c++
int id;
int sz[N], dep[N], prt[N], son[N], top[N], idx[N], H[N];
void dfs1(int u) {
    sz[u] = 1;
    for (int i = head[u]; i; i = e[i].nt) {
        int v = e[i].b;
        if (v == prt[u]) continue;
        prt[v] = u; dep[v] = dep[u] + 1;
        dfs1(v); sz[u] += sz[v];
        if (!son[u] || sz[v] > sz[son[u]]) son[u] = v;
    }
}
void dfs2(int u, int chain) {
    top[u] = chain;
    idx[u] = ++id;
    H[id] = u;
    if (son[u]) dfs2(son[u], chain);
    for (int i = head[u]; i; i = e[i].nt) {
        int v = e[i].b;
        if (v == prt[u] || v == son[u]) continue;
        dfs2(v, v);
    }
}
```
#### 树链操作
查询或修改，原理大致一样，通过跳该节点top节点的prt来确保节点上移的复杂度为$O(log_2)$，每次选择深度较深的节点上移，直到两个节点到达同一条重链上来为止。
```c++
//以查询为例，其他操作如LCA，修改……都差不多。
//线段树中查询函数，p为根节点，查询区间为[l,r]
struct Seg {int query(int p, int l, int r){...}} seg;
int query(int x, int y) {
	int ret = 0;
	while (top[x] != top[y]) {//不在同一条重链上面
    	if (dep[top[x]] < dep[top[y]]) swap(x, y);
    	//比较重链顶点的深度，因为即将要跳到较深的一个重链顶点的父亲节点上。
    	ret += seg.query(1, idx[top[x]], idx[x]);
    	x = prt[top[x]];
	}
	//最终两个节点一定会在同一条重链上面，再加上两者之间的数值即可
	if (dep[x] < dep[y]) swap(x, y);
	ret += seg.query(1, idx[y], idx[x]);
	return ret;
}
```
#### 边权处理
由于树链剖分动态处理的是点权，为了处理边权只需将**边权下放**，处理两点之间的时候，注意**不处理LCA的点权**（因为，LCA的点权LCA和prt[LCA]之间的边权，它是不可能在两点路径上的一条边）
#### 动态求解两点之间的桥的个数
[[AHOI2005] 航线规划](https://www.luogu.com.cn/record/list?user=76226)
问题：求无向图中两点之间的个数，有两钟操作：删一条边和询问两点之间最短路径上桥的个数。
思路：看到桥就想到tarjan求桥的方法，但是删边操作不好处理可能会增加桥的数量。**反向思考：加边/点变为删边/点**，加边的操作只会减少桥的数量，而且减少的就是两端点之间的所有的桥。所以，可以先缩点建图变为树形结构，每个边权值赋值为1，再进行树链剖分，每次连接两点，两点之间的边权值全部变为0，查询直接求出两点之间边权值之和。答案倒序输出即可。
### 后缀自动机SAM
hihoCoder
[SAM基本概念](http://hihocoder.com/problemset/problem/1441)
[SAM实现方法](http://hihocoder.com/problemset/problem/1445)
[陈立杰PPT](https://max.book118.com/html/2016/1007/57498384.shtm)
```c++
struct SAM {
    int cnt, last, ch[N][26], mx[N], prt[N];
    int sz[N];//统计该节点right集合大小，就是在母串中出现的次数
    SAM() {cnt = last = 1;}//记得初始化
    void add(int c) {
        int p = last, np = ++cnt; last = np;
        mx[np] = mx[p] + 1; sz[np] = 1;
        for (; p && !ch[p][c]; p = prt[p]) ch[p][c] = np;
        if (!p) prt[np] = 1;
        else {
            int q = ch[p][c];
            if (mx[q] == mx[p] + 1) prt[np] = q;
            else {
                int nq = ++cnt; mx[nq] = mx[p] + 1;
                prt[nq] = prt[q]; prt[q] = prt[np] = nq;
                memcpy(ch[nq], ch[q], sizeof(ch[q]));
                for (; ch[p][c] == q; p = prt[p]) ch[p][c] = nq;
            }
        }
    }
    //进行拓扑排序
    void dp() {
        for (int i = cnt; i; i--) {
            sz[prt[c[i]]] += sz[c[i]];
        }
    }
} sam;
```
#### 桶排序（简化拓扑排序）
```c++
//桶排序，按照mx[]的从小到大排序在c[]中
int t[N], c[N];
void tsort() {
    for (int i = 1; i <= cnt; i++) t[mx[i]]++;//入桶
    for (int i = 1; i <= cnt; i++) t[i] += t[i - 1];//记录排名
    for (int i = 1; i <= cnt; i++) c[t[mx[i]]--] = i;//排名对应序号
    for (int i = 1; i <= cnt; i++) sz[prt[c[i]]] += sz[c[i]];//对sz数组进行累加，求出一个节点中的endpos出现次数，即拓扑排序后的DP
}
```
#### 倍增
在parent树上进行倍增，可以快速$O(logn)$求出子串属于SAM中的哪个节点
```c++
int pos[N][20];
void init() {
   	for (int i = 1; i <= cnt; i++) pos[i][0] = prt[i];
   	for (int j = 1; j < 20; j++)//从小到大
   	for (int i = 1; i <= cnt; i++) pos[i][j] = pos[pos[i][j - 1]][j - 1];
}
int find(int p, int len) {//p为当前子串右端点在SAM上对应的位置
	//从大到小
    for (int i = 19; i >= 0; i--) if (mxl[pos[p][i]] >= len) p = pos[p][i];
}
```
#### 广义SAM
将多个字符串插入到SAM中去，让SAM能对多个字符串同时处理，处理出多个字符串的共同与不同之处。
构造方法：在每次新加入一个串的时候将last=1；同时注意不要插入相同的节点产生多余和错误，判断节点是否重复的方法：因为每次之加入一个字符，那么一定会有$mxl[np]=mxl[p]+1$，如果$ch[last][c]$有值存在，并且$mxl[ch[last][c]]=mxl[last]+1$于是就可以直接转移$last=ch[last][c]$。
```c++
void add(int c) {
    if (ch[last][c] && mxl[ch[last][c]] == mxl[last] + 1) {
        last = ch[last][c];
        return;
    }
    ///...其他部分与标程一致...
}
void init() {
    last = 1;
    for (int i = 1; i <= len; i++) add(s[i] - 'a')；
}

```
### 回文算法
#### Manacher
利用已有的大的回文串，当大的回文串中包含有小的回文串时候，计算左侧和右侧会重复计算，浪费时间，通过对称的性质，通过DP的思路将右侧的回文串对称到左侧去，从而降低时间复杂度。
注意各个变量的初始值和对字符串进行的预处理操作。
```c++
char A[N << 1];
int len, rad[N << 1];//rad[i]表示(以i为对称中心的最长回文串的长度+1)/2
//len保存处理后的总数组长度
void init() {
    char c = getchar();
    //对0做一个特殊的标记，保证不会与后面的符号重复
    A[0] = '~'; A[++len] = '#';//'#'作为分隔符号
    while (c < 'a' || c > 'z') c = getchar();
    while (c >= 'a' && c <= 'z') {A[++len] = c; A[++len] = '#'; c = getchar();}
}
void Manacher() {
    init();
    for (int i = 1, r = 0, mid = 0; i <= len; i++) {
    	//如果i在已经处理过的长度之内，则可以对称找到现有可以处理到的最长长度rad[i]，但绝对不可能处理到还未处理过的长度(就是r的右侧)
        if (i <= r) rad[i] = min(rad[(mid << 1) - i], r - i + 1);
        //开始尝试向外拓展
        while (A[i - rad[i]] == A[i + rad[i]]) ++rad[i];
        //如果当前拓展的长度大于了已有的扫描范围，就更新右端端点r和对称中心mid
        if (i + rad[i] > r) r = i + rad[i] - 1, mid = i;
    }
}
```
### 数学
#### 本原解
$x^2+y^2=z^2$且$(x,y)=(y,z)=(x,z)=1$，求不定方程的正整数解。
全部解均可表示成：$z=r^2+s^2,y=2rs,x=r^2-s^2,其中r>s>0,(r,s)=1,2\nmid (r+s)$
且有本原解的性质：x，y一奇一偶即$2\nmid (x+y)$，z为奇数且$6\mid xy$（用模证明）
例题：[ Streaming_4_noip_day2 距离统计](https://blog.csdn.net/weixin_44627639/article/details/109209451)

