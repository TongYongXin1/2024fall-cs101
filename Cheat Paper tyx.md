M, N = map(int, input().split())    x=tuple/list(map(int, input().split()))        list=[?]*n

input().strip() 去除两端的空格和换行符           print(f"{name} is {age} years old.")   

split()拆开  'abcd'-'a''b'c'd' 默认空格 可以是其他分隔符

c = [[0] * m for _ in range(n)]     生成n*m空矩阵     

max(max(row) for row in matrix)矩阵最大值  

points = sorted([(matrix[i][j], i, j) for i in range(rows) for j in range(cols)]) 排序

range(start,stop,step)  start到stop-1 start默认0    list(range())       切片相同（有首无尾）

matrix = [list(map(int, input().split())) for _ in range()]   读取矩阵        

sz=[int(d) for d in str(number)]
    if number==sum(d**3 for d in sz)

 print(" ".join(map(str,list)))   " "里是希望的连接符    \n换行

ord('A')=65  chr(65)='A'     char.upper/lower/isdigit/isalpha/isupper()

numbers = [x for x in range(10) if]  squares = [x**2 for x in range(10)]

numbers.sort(reverse=True) 默认升序   **`sorted()`**：返回一个新的排序后的列表，原列表不变    if ... in list

list.remove(num)   del list[index]    list.pop(index)  list.append(num)  list.insert(index,num)    list.index(x) 返回索引    

dict={key:value,}    squares = {x: x**2 for x in range(5)}

可以使用 `.keys()`、`.values()` 和 `.items()` 方法来获取字典的所有键、值和键值对   value=dict[key]

反向字典 reverse_person = {value: key for key, value in person.items()}   dict=dict(zip(list_key,list_value))

for index, value in enumerate(fruits,start=1):         list(enumerate())  dict(enumerate())

**排队  如果两名同学相邻，并且他们的身高之差不超过 D，那么老师就能交换他俩的顺序   字典序最小的所有同学（从左到右）身高序列是什么？

```
N, D = map(int, input().split())
h = [int(input()) for _ in range(N)]
used = [0] * N
while 0 in used:
    free = []
    for i in range(N):
        if used[i]:
            continue
        if not free:
            minv = h[i]
            maxv = h[i]
        else:
            if h[i] > maxv:
                maxv = h[i]
            if h[i] < minv:
                minv = h[i]

        if maxv - minv > 2*D:
            break

        if (h[i] + D >= maxv and h[i] - D <= minv):
            free.append(h[i])
            used[i] = 1
    free.sort()
    for v in free:
        print(v)
```

#### 遍历每一行（使用索引）：

```
for i in range(len(matrix)):
```

#### 遍历每一列（使用索引）：

```
for j in range(len(matrix[0])):  
    column = [matrix[i][j] for i in range(len(matrix))]
```

埃氏筛

```
def sieve_of_eratosthenes(max_n):
    is_prime = [True] * (max_n + 1)
    p = 2
    while (p * p <= max_n):
        if (is_prime[p]):
            for i in range(p * p, max_n + 1, p):
                is_prime[i] = False
        p += 1
    return [p for p in range(2, max_n + 1) if is_prime[p]]
```

from functools import cmp_to_key

def compare_by(a,b)

return -1    a排在b前

return 1     a排在b后

sorted_words = sorted(words, key=cmp_to_key(compare_by_length))

indexs.sort(key=lambda x: (x[0], x[1]))

:.nf          保留n位小数

#### 最大整数

```python
def compare(a,b):
    x,y=a+b,b+a
    if int(x)<int(y):
        return True
    elif int(x)==int(y) and len(a)>len(b):
        return True
    return False

def bubble_sort(lst):
    n=len(lst)
    for i in range(n):
        for j in range(i+1,n):
            if compare(lst[i],lst[j]):
                lst[i],lst[j]=lst[j],lst[i]
    return lst

m=int(input())
n=int(input())
lst=list(map(str,input().split()))
bubble_sort(lst)
l=[]
for i in range(n):
    l.append(len(lst[i]))
dp=[["" for _ in range(m+1)] for _ in range(n)]
if l[n-1]<=m:
    for j in range(l[n-1],m+1):
        dp[n-1][j]=lst[n-1]
for i in range(n-2,-1,-1):
    for j in range(m+1):
        dp[i][j]=dp[i+1][j]
        if j>=l[i]:
            if dp[i+1][j]=="":
                dp[i][j]=lst[i]
            else:
                s=max(int(lst[i]+dp[i+1][j-l[i]]),int(dp[i+1][j]))
                s=str(s)
                dp[i][j]=s
print(dp[0][m])


from functools import cmp_to_key
def cmp(x, y): # -1:不交换，x在y前；1:交换，y在x前
    if x + y > y + x: return -1
    return 1
m = int(input())
n = int(input())
s = input().split()
s = sorted(s, key = cmp_to_key(cmp))
ans = [""] + ["!"] * m
for i in range(n):
    for j in range(m, len(s[i]) - 1, -1):
        if ans[j - len(s[i])] != "!":
            if ans[j] == "!" or ans[j - len(s[i])] + s[i] > ans[j]: ans[j] = ans[j - len(s[i])] + s[i]
for j in range(m, -1, -1):
    if ans[j] != "!":
        print(ans[j])
        break
```



#### **安装雷达**

```python
import math
def Radar_Installation(n, d, islands):
    distances = []
    for x, y in islands:
        if y > d:  
            return -1
        left = x - math.sqrt(d**2 - y**2)
        right = x + math.sqrt(d**2 - y**2)
        distances.append((left, right))
    distances.sort(key=lambda x: x[1])
    count = 0
    current_end = -float('inf')  
    for distance in distances:
        left, right = distance
        if current_end < left:
            current_end = right
            count += 1           
    return count
case_number = 1
while True:
    line = input().strip()
    if line == "0 0":
        break
    n, d = map(int, line.split())
    islands = []   
    for _ in range(n):
        x, y = map(int, input().strip().split())
        islands.append((x, y))
    input()
    result = Radar_Installation(n, d, islands)
    print(f"Case {case_number}: {result}")
    case_number += 1
```

#### 全排列 递归,dfs

```python
def dfs(n, path, used, res):
    if len(path) == n:
        res.append(path[:])
        return
    for i in range(1, n+1):
        if not used[i]:
            used[i] = True
            path.append(i)
            dfs(n, path, used, res)
            path.pop()
            used[i] = False

def print_permutations(n):
    res = []
    dfs(n, [], [False]*(n+1), res)
    for perm in sorted(res):
        print(' '.join(map(str, perm)))

nums = []
while True:
    num = int(input())
    if num == 0:
        break
    nums.append(num)

for num in nums:
    print_permutations(num)
```

    def quanpailie(arr, li):
        if not arr:
            print(" ".join(map(str, li)))
            return
    for i in range(len(arr)):
        first = arr[i]
        remaining = arr[:i] + arr[i+1:]
        quanpailie(remaining, li + [first])
        n = int(input())
    arr = list(range(1, n + 1)) 
    quanpailie(arr, [])


​    
​    
    def a(s):
        if len(s) == 3:
            list1.append(s)
            return
        for i in range(1, 4):
            if all(str(i) != s[j] for j in range(len(s))):
                a(s + str(i))
    
    a('')
    samples = int(input())
    for k in range(samples):
        print(list1[int(input()) - 1])

##### 八皇后

```python
# 02754 八皇后, http://cs101.openjudge.cn/practice/02754/
list1 = []

def queen(s):
    if len(s) == 8:
        list1.append(s)
        return
    for i in range(1, 9):
        if all(str(i) != s[j] and abs(len(s) - j) != abs(i - int(s[j])) for j in range(len(s))):
            queen(s + str(i))

queen('')
samples = int(input())
for k in range(samples):
    print(list1[int(input()) - 1])

```

```
积木
def can_spell_word(word, blocks):
    used = [False] * 4    
    def backtrack(index):
        if index == len(word):
            return True
        target_char = word[index]        
        for i in range(4):
            if not used[i] and target_char in blocks[i]:
                used[i] = True
                if backtrack(index + 1):
                    return True
                used[i] = False        
        return False
    return backtrack(0)
def main():
    N = int(input())
    blocks = [input().strip() for _ in range(4)]  
    words = [input().strip() for _ in range(N)] 
    for word in words:
        if can_spell_word(word, blocks):
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()


from collections import defaultdict
from itertools import permutations

a = defaultdict(int)
b = defaultdict(int)
c = defaultdict(int)
d = defaultdict(int)
n = int(input())

for i in input():
    a[i] += 1
for i in input():
    b[i] += 1
for i in input():
    c[i] += 1
for i in input():
    d[i] += 1

dicts = [a, b, c, d]

def check(word):
    for perm in permutations(dicts, len(word)):
        for i, d in enumerate(perm):
            if word[i] not in d:
                break
        else:
            return 'YES'
    else:
        return 'NO'

for _ in range(n):
    word = input()
    print(check(word))

```



### DP问题

##### 拦截导弹

```python
def lanjiedaodan(n, h):
    dp = [1] * n      
    for i in range(1, n):
        for j in range(i):
            if h[j] >= h[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
n = int(input())  
h = list(map(int, input().split()))  
print(lanjiedaodan(n, h))
```

##### 小偷背包 0-1背包 有上限 取或不取

```python
def stealbag(N, B, values, weights):
    dp = [0] * (B + 1)   
    for i in range(N):
        for j in range(B, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    
    return dp[B]

N, B = map(int, input().split())
values = list(map(int, input().split()))
weights = list(map(int, input().split()))
print(stealbag(N, B, values, weights))
```

##### 剪彩带

```python
def Cut_Ribbon(n, a, b, c):
	dp = [-1] * (n + 1) 
	dp[0] = 0 
	for i in range(1, n + 1):
		if i >= a and dp[i - a] != -1:
			dp[i] = max(dp[i], dp[i - a] + 1)
		if i >= b and dp[i - b] != -1:
			dp[i] = max(dp[i], dp[i - b] + 1)
		if i >= c and dp[i - c] != -1:
			dp[i] = max(dp[i], dp[i - c] + 1)
	return dp[n]
n, a, b, c = map(int, input().split())
print(Cut_Ribbon(n, a, b, c))
```

##### 零钱兑换

```python
n, m = map(int, input().strip().split())
coins = list(map(int, input().strip().split()))
dp = [float('inf')] * (m + 1)
dp[0] = 0
for coin in coins:
    for j in range(coin, m + 1):
        dp[j] = min(dp[j], dp[j - coin] + 1)
print(dp[m] if dp[m] != float('inf') else -1)
```

##### boredom

```python
def Boredom(n, arr):
    max_val = max(arr)
    count = [0] * (max_val + 1)
    for num in arr:
        count[num] += 1
    dp = [0] * (max_val + 1)
    dp[1] = count[1]  
    for i in range(2, max_val + 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + i * count[i])
    return dp[max_val]
n = int(input())
arr = list(map(int, input().split()))
print(Boredom(n, arr))
```

##### 跳台阶/不同路径

```python
def count_steps(n):
    dp = [0] * (n + 1)    
    dp[0] = 1
    dp[1] = 1  
    for i in range(2, n + 1):
        dp[i] = sum(dp[j] for j in range(i))     
    return dp[n]
n = int(input())
print(count_steps(n))

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n for _ in range(m)] 
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]  
        return dp[m-1][n-1]
```

##### 回文字符

```python
def min_operations(s):
    n = len(s)
    dp = [[0]*n for _ in range(n)]
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = min(dp[i+1][j], dp[i][j-1], dp[i+1][j-1]) + 1
    return dp[0][n-1]

s = input().strip()
print(min_operations(s))

def longestPalindrome(self, s: str) -> str:   
        n = len(s)
        max_len = 0
        for i in range(2 * n + 1):
            if i % 2 == 0:
                left, right = i // 2, i // 2
            else:
                left, right = i // 2, i // 2 + 1
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            if right - left - 1 > max_len:
                max_len = right - left - 1
                result = s[left + 1: left + max_len + 1]
        return result

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]  
        start = 0  
        max_length = 1  
        for i in range(n):
            dp[i][i] = True
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_length = 2
        for length in range(3, n + 1): 
            for i in range(n - length + 1):
                j = i + length - 1  
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    if length > max_length:
                        start = i
                        max_length = length
        return s[start:start + max_length]
```

#### 最长公共子序列

```python
for i in range(len(A)):
    for j in range(len(B)):
        if A[i] == B[j]:
            dp[i][j] = dp[i-1][j-1]+1
        else:
            dp[i][j] = max(dp[i-1][j],dp[i][j-1])
```

#### 最长单调子序列

```python
dp = [1]*n
for i in range(1,n):
    for j in range(i):
        if A[j]<A[i]:
            dp[i] = max(dp[i],dp[j]+1)
ans = sum(dp)
```

**多重背包**

最简单的思路是将多个同样的物品看成多个不同的物品，从而化为0-1背包。稍作优化：可以改善拆分方式，譬如将m个1拆成x_1,x_2,……,x_t个1，只需要这些x_i中取若干个的和能组合出1至m即可。最高效的拆分方式是尽可能拆成2的幂，也就是所谓“二进制优化”

```python
dp = [0]*T
for i in range(n):
    all_num = nums[i]
    k = 1
    while all_num>0:
        use_num = min(k,all_num) #处理最后剩不足2的幂的情形
        for t in range(T,use_num*time[i]-1,-1):
            dp[t] = max(dp[t-use_num*time[i]]+use_num*value[i],dp[t])
        k *= 2
        all_num -= use_nume
```

双DP  土豪购物

```python
a = list(map(int, input().split(',')))
dp1 = [0] * len(a);
dp2 = [0] * len(a)
dp1[0] = a[0];
dp2[0] = a[0]
for i in range(1, len(a)):
    dp1[i] = max(dp1[i - 1] + a[i], a[i])
    dp2[i] = max(dp1[i - 1], dp2[i - 1] + a[i], a[i])
print(max(dp2))
```

### 堆

```
import heapq
lst = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
heapq.heapify(lst)
print(lst)  # 输出堆结构：[1, 1, 2, 3, 3, 9, 4, 6, 5, 5, 5]
heapq.heappop(heap)
弹出堆中的最小元素，并且保持堆的特性
heap = [1, 3, 5, 7, 9]
min_item = heapq.heappop(heap)
print(min_item)  # 输出：1
print(heap)      # 输出：[3, 7, 5, 9]
heapq.heappush(heap, item)
将一个元素 item 添加到堆中
heap = [1, 3, 5, 7, 9]
heapq.heappush(heap, 4)
print(heap)  # 输出：[1, 3, 4, 7, 9, 5]
heapq.nlargest(n, iterable, key=None)
返回可迭代对象中前 n 个最大元素
heapq.nsmallest(n, iterable, key=None)
返回可迭代对象中前 n 个最小元素，默认是从小到大排序
```

##### 打怪兽

```python
import heapq
from collections import defaultdict
def monster(n_cases, cases):
    results = []    
    for case in cases:
        n, m, b = case[0]
        skills = case[1]        
        skills_time = defaultdict(list)
        for ti, xi in skills:
            skills_time[ti].append(xi)
        for ti in sorted(skills_time.keys()):
            max_damages = heapq.nlargest(m, skills_time[ti])
            total_damage = sum(max_damages)
            b -= total_damage           
            if b <= 0:
                results.append(ti)
                break
        else:
            results.append("alive")
    return results

```











##### 螺旋矩阵

```
 def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: return []
        l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
        while True:
            for i in range(l, r + 1): res.append(matrix[t][i]) # left to right
            t += 1
            if t > b: break
            for i in range(t, b + 1): res.append(matrix[i][r]) # top to bottom
            r -= 1
            if l > r: break
            for i in range(r, l - 1, -1): res.append(matrix[b][i]) # right to left
            b -= 1
            if t > b: break
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # bottom to top
            l += 1
            if l > r: break
        return res

```

## DFS

#### 迷宫问题

问能否走到出口、输出可行路径、输出连通分支数、输出连通块大小等。这种问题应用经典的图搜索DFS即可解决，不需要回溯，**每个点只需要搜到一次**，所以**不需要撤销标记**。

```python
directions = [...]
...
visited = [[False]*n for _ in range(m)]
def dfs(x,y):
    global area
    if vis[x][y]:
        return
    visited[x][y] = True
    ...
    for dx,dy in directions:
        nx,ny=x+dx,y+dy
        if 0<=nx<m and 0<=ny<n and vis[nx][ny] and ...:
            dfs(nx,ny)
#此处还可以在dfs前不用标记vis，在for循环里：
#vis[nx][ny]=1
#dfs(nx,ny)
#vis[nx][ny]=0
#这样自身形成回溯
for i in range(m):
    for j in range(n):
        if not visited[i][j]:
            dfs(i,j)
```

#### 有回溯

##### 马走日

```python
directions = [(-2, -1), (-2, 1), (2, -1), (2, 1),
              (-1, -2), (-1, 2), (1, -2), (1, 2)]
def dfs(x, y, n, m, visited, step):
    if step == n * m:
        return 1

    total_paths = 0
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
            visited[nx][ny] = True  
            total_paths += dfs(nx, ny, n, m, visited, step + 1)  
            visited[nx][ny] = False  
    return total_paths

T = int(input())  
for _ in range(T):
    n, m, x, y = map(int, input().split())  
    visited = [[False] * m for _ in range(n)]  
    visited[x][y] = True  
    print(dfs(x, y, n, m, visited, 1))  
```

##### 最大权值路径

```
def dfs(x, y, now_value):
    global max_value, opt_path
    # 如果到达右下角，更新最大权值和最优路径
    if x == n - 1 and y == m - 1:
        if now_value > max_value:
            max_value = now_value
            opt_path = temp_path[:]
        return
    
    # 标记当前位置为已访问
    visited[x][y] = True
    
    # 尝试向四个方向移动
    for dx, dy in directions:
        next_x, next_y = x + dx, y + dy
        if 0 <= next_x < n and 0 <= next_y < m and not visited[next_x][next_y]:
            next_value = now_value + maze[next_x][next_y]
            temp_path.append((next_x, next_y))
            dfs(next_x, next_y, next_value)
            temp_path.pop()  # 回溯
    
    # 取消当前位置的访问标记
    visited[x][y] = False

# 读取输入
n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

# 初始化变量
max_value = float('-inf')
opt_path = []
temp_path = [(0, 0)]
visited = [[False] * m for _ in range(n)]
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 从左上角开始DFS搜索
dfs(0, 0, maze[0][0])

# 输出最优路径
for x, y in opt_path:
    print(x + 1, y + 1)
```

#### 无回溯

##### 最大连通面积

```python
T = int(input().strip())
data = []
for _ in range(T):
    N, M = map(int, input().strip().split())
    grid = [input().strip() for _ in range(N)]
    data.append((N, M, grid))
directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def find_largest_region_area(N, M, grid):
    visited = [[False] * M for _ in range(N)]  
    max_area = 0  

    def dfs(x, y):
        area = 1  
        visited[x][y] = True
        for dx, dy in directions:  
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] == 'W' and not visited[nx][ny]:
                area += dfs(nx, ny)  
        return area
    for i in range(N):
        for j in range(M):
            if grid[i][j] == 'W' and not visited[i][j]:
                max_area = max(max_area, dfs(i, j))  

    return max_area

for t in range(T):
    N, M, grid = data[t]
    print(find_largest_region_area(N, M, grid))
```

##### 受祝福平方

```python
import math
def is_square(num):
    if num <= 0:
        return False
    root = int(math.sqrt(num))
    return root * root == num

def dfs(index, A, n):
    if index == n: 
        return True    
    for i in range(index + 1, n + 1):  
        num = int(A[index:i])  
        if is_square(num) and dfs(i, A, n):  
            return True   
    return False

def is_blessed_id(A):
    A = str(A)  
    n = len(A)
    return "Yes" if dfs(0, A, n) else "No"
A = int(input())
print(is_blessed_id(A))
```

##### 水淹七军

```python
import sys

sys.setrecursionlimit(300000)
input = sys.stdin.read


# 判断坐标是否有效
def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n


# 深度优先搜索模拟水流
def dfs(x, y, water_height_value, m, n, h, water_height):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    for i in range(4):
        nx, ny = x + dx[i], y + dy[i]
        if is_valid(nx, ny, m, n) and h[nx][ny] < water_height_value:
            if water_height[nx][ny] < water_height_value:
                water_height[x][y] = water_height_value
                dfs(nx, ny, water_height_value, m, n, h, water_height)


# 主函数
def main():
    data = input().split()  # 快速读取所有输入数据
    idx = 0
    k = int(data[idx])
    idx += 1
    results = []

    for _ in range(k):
        m, n = map(int, data[idx:idx + 2])
        idx += 2
        h = []
        for i in range(m):
            h.append(list(map(int, data[idx:idx + n])))
            idx += n
        water_height = [[0] * n for _ in range(m)]

        i, j = map(int, data[idx:idx + 2])
        idx += 2
        i, j = i - 1, j - 1

        p = int(data[idx])
        idx += 1

        for _ in range(p):
            x, y = map(int, data[idx:idx + 2])
            idx += 2
            x, y = x - 1, y - 1
            if h[x][y] <= h[i][j]:
                continue

            dfs(x, y, h[x][y], m, n, h, water_height)

        results.append("Yes" if water_height[i][j] > 0 else "No")

    sys.stdout.write("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
```

决战双十一

```
result = float("inf")
n, m = map(int, input().split())
store_prices = [input().split() for _ in range(n)]  
coupons = [input().split() for _ in range(m)]  
def dfs(store_index, total_price, store_purchase):
    global result   
    if store_index == n:
        coupon_discount = 0
        for i in range(m):
            max_coupon = 0
            for coupon in coupons[i]:
                a, b = map(int, coupon.split('-'))
                if store_purchase[i] >= a:
                    max_coupon = max(max_coupon, b)
            coupon_discount += max_coupon
        final_price = total_price - (total_price // 300) * 50 - coupon_discount
        result = min(result, final_price)
        return
    for item in store_prices[store_index]:
        idx, p = map(int, item.split(':'))
        store_purchase[idx - 1] += p  
        dfs(store_index + 1, total_price + p, store_purchase)  
        store_purchase[idx - 1] -= p 
dfs(0, 0, [0] * m)
print(result)
```

滑雪

```python
rows, cols = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(rows)]
points = sorted([(matrix[i][j], i, j) for i in range(rows) for j in range(cols)])
dp = [[1] * cols for _ in range(rows)]
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
longest_path = 1
for height, x, y in points:
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and matrix[nx][ny] < height:
            dp[x][y] = max(dp[x][y], dp[nx][ny] + 1)
    longest_path = max(longest_path, dp[x][y])

print(longest_path)

r, c = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(r)]
dp = [[0 for _ in range(c)] for _ in range(r)]


def dfs(x, y):
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    for i in range(4):
        nx, ny = x + dx[i], y + dy[i]
        if 0 <= nx < r and 0 <= ny < c and matrix[x][y] > matrix[nx][ny]:
            if dp[nx][ny] == 0:
                dfs(nx, ny)
            dp[x][y] = max(dp[x][y], dp[nx][ny] + 1)
    if dp[x][y] == 0:
        dp[x][y] = 1


ans = 0
for o in range(r):
    for j in range(c):
        if not dp[o][j]:
            dfs(o, j)
        ans = max(ans, dp[o][j])
print(ans)
```

## BFS 最短路径

```
from collections import deque
空的 deque：
dq = deque()
dq.append(x)： 向 deque 的右端添加元素
dq.appendleft(x)： 向 deque 的左端添加元素
dq.pop()： 从 deque 的右端移除并返回一个元素
dq.popleft()： 从 deque 的左端移除并返回一个元素
count(x)： 统计元素 x 在 deque 中出现的次数
```

模板：

```python
from collections import deque
directions=[...]
queue=deque([(x0,y0)])
visited=set()
visited.add((x0,y0))
while queue:
	x,y=queue.popleft()
    for dx,dy in directions:
		nx,ny=x+dx,y+dy
        if -1<nx<n and -1<ny<m and ... and ma[nx][ny] not in visited:
            visited.add((nx,ny))
            queue.append((nx,ny))
            ...
#对while这一块，有时需要这样写：
while queue:
    for _ in range(len(queue)):
        x,y=queue.popleft()
    	for dx,dy in directions:
		nx,ny=x+dx,y+dy
        if -1<nx<n and -1<ny<m and ... and ma[nx][ny] not in visited:
            visited.add((nx,ny))
            queue.append((nx,ny))
            ...
```

##### 寻宝

```python
from collections import deque
m, n = map(int, input().split())
treasure_map = [list(map(int, input().split())) for _ in range(m)]
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def find_treasure():
    q = deque([(0, 0, 0)])  # (x, y, steps)
    visited = [[False] * n for _ in range(m)]  
    visited[0][0] = True

    while q:
        x, y, steps = q.popleft()
        if treasure_map[x][y] == 1:
            return steps
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and treasure_map[nx][ny] != 2:
                visited[nx][ny] = True
                q.append((nx, ny, steps + 1))  
    return "NO"

result = find_treasure()
print(result)
```

##### 跳房子

```python
from collections import deque
#import heapq

def bfs(s, e):
    q = deque()
    q.append((0, s, ''))
    vis = set()
    vis.add(s)
    # q = []
    #heapq.heappush(q, (0, s, ''))

    while q:
        step, pos, path = q.popleft()
        #step, pos, path = heapq.heappop(q)
        if pos == e:
            return step, path

        if pos * 3 not in vis:
            q.append((step+1, pos*3, path+'H'))
            vis.add(pos*3)
            #heapq.heappush(q, (step+1, pos*3, path+'H'))
        if int(pos // 2) not in vis:
            vis.add(int(pos//2))
            q.append((step+1, int(pos//2), path+'O'))
            #heapq.heappush(q, (step+1, int(pos//2), path+'O'))

while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    step, path = bfs(n, m)
    print(step)
    print(path)
```

##### 小游戏

```python
from collections import deque
def bfs(start, end, grid, h, w):
    queue = deque([start])
    in_queue = set()
    dirs = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    ans = []
    while queue:
        x, y, d_i_r, seg = queue.popleft()
        if (x, y) == end:
            ans.append(seg)
            break
        for i, (dx, dy) in enumerate(dirs):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h + 2 and 0 <= ny < w + 2 and ((nx, ny, i) not in in_queue):
                new_dir = i
                new_seg = seg if new_dir == d_i_r else seg + 1
                if (nx, ny) == end:
                    ans.append(new_seg)
                    continue
                if grid[nx][ny] != 'X':
                    in_queue.add((nx, ny, i))
                    queue.append((nx, ny, new_dir, new_seg))
    if len(ans) == 0:
        return -1
    else:
        return min(ans)
board_num = 1
while True:
    w, h = map(int, input().split())
    if w == h == 0:
        break
    grid = [' ' * (w + 2)] + [' ' + input() + ' ' for _ in range(h)] + [' ' * (w +2)]
    print(f"Board #{board_num}:")
    pair_num = 1
    while True:
        y1, x1, y2, x2 = map(int, input().split())
        if x1 == y1 == x2 == y2 == 0:
            break
        start = (x1, y1, -1, 0)
        end = (x2, y2)
        seg = bfs(start, end, grid, h, w)
        if seg == -1:
            print(f"Pair {pair_num}: impossible.")
        else:
            print(f"Pair {pair_num}: {seg} segments.")
        pair_num += 1
    print()
    board_num += 1
```

传送门（每次移动只能向上下左右移动一格，且只能移动到平地或传送点上。当位于传送点时，可以选择传

送到另一个 2 处（传送不计入步数），也可以选择不传送。求从迷宫左上角到右下角的最小步数。）——注意映

射		

```
from collections import deque
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
def canVisit(x, y, n, m, maze, in_queue):
	return 0 <= x < n and 0 <= y < m and (maze[x][y] == 0 or maze[x][y] == 2) and (x, y)
not in in_queue
def BFS(start_x, start_y, n, m, maze, transMap):
	q = deque([(0, start_x, start_y)]) # (step, x, y)
	in_queue = {(start_x, start_y)}
	while q:
		step, x, y = q.popleft()
		if x == n - 1 and y == m - 1:
			return step
		for i in range(MAXD):
			next_x = x + dx[i]
			next_y = y + dy[i]
			if canVisit(next_x, next_y, n, m, maze, in_queue):
				in_queue.add((next_x, next_y))
				q.append((step + 1, next_x, next_y))
				if maze[next_x][next_y] == 2:
					trans_position = transMap.get((next_x, next_y))
					if trans_position:
						in_queue.add(trans_position)
						q.append((step + 1, trans_position[0], trans_position[1]))
	return -1
n, m = map(int, input().split())
maze = []
transMap = {}
transVector = []
for i in range(n):
	row = list(map(int, input().split()))
	maze.append(row)
	if 2 in row:
		for j, val in enumerate(row):
			if val == 2:
				transVector.append((i, j))
	if len(transVector) == 2:
		transMap[transVector[0]] = transVector[1]
		transMap[transVector[1]] = transVector[0]
		transVector = [] # 清空 transVector 以便处理下一对传送点
if transVector:
	print("Error: Unpaired teleportation point found.")
	exit(1)
step = BFS(0, 0, n, m, maze, transMap)
print(step)
```

```python
孤岛最短距离

from collections import deque
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def dfs(x, y, grid, n, queue):
    grid[x][y] = 2  
    queue.append((x, y))  
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
            dfs(nx, ny, grid, n, queue)

def bfs(grid, n, queue):
    distance = 0
    while queue:
        for _ in range(len(queue)):
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    if grid[nx][ny] == 1:  
                        return distance
                    elif grid[nx][ny] == 0:
                        grid[nx][ny] = 2  
                        queue.append((nx, ny))
        distance += 1
    return distance

def main():
    n = int(input())  
    grid = [list(map(int, input().strip())) for _ in range(n)]  
    queue = deque()  
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1: 
                dfs(i, j, grid, n, queue)
                return bfs(grid, n, queue)  

if __name__ == "__main__":
    print(main())
```

有权最小值  Dijkstra

```python
import heapq
def dijkstra(m, n, Map, start, end):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if Map[start[0]][start[1]] == "#" or Map[end[0]][end[1]] == "#":
        return "NO"
    INF = float('inf')
    cost = [[INF] * n for _ in range(m)]
    cost[start[0]][start[1]] = 0
    pq = [(0, start[0], start[1])]   
    while pq:
        current_cost, x, y = heapq.heappop(pq)
        if (x, y) == end:
            return current_cost
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and Map[nx][ny] != "#":
                new_cost = current_cost + abs(int(Map[x][y]) - int(Map[nx][ny]))              
                if new_cost < cost[nx][ny]:
                    cost[nx][ny] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))
    return "NO"

def main():
    m, n, p = map(int, input().split())
    Map = []
    for _ in range(m):
        Map.append(input().split())
    for _ in range(p):
        start_x, start_y, end_x, end_y = map(int, input().split())
        start = (start_x, start_y)
        end = (end_x, end_y)
        result = dijkstra(m, n, Map, start, end)
        print(result)
if __name__ == "__main__":
    main()
```

变换迷宫

```python
from collections import deque
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def bfs(R, C, K, Map, start, end):
    queue = deque([(start[0], start[1], 0)])
    visited = [[[False] * K for _ in range(C)] for _ in range(R)]
    visited[start[0]][start[1]][0] = True
    
    while queue:
        x, y, time = queue.popleft()
        if (x, y) == end:
            return time
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            next_time = (time + 1) % K  
            if 0 <= nx < R and 0 <= ny < C and not visited[nx][ny][next_time]:
                if Map[nx][ny] != '#':
                    visited[nx][ny][next_time] = True
                    queue.append((nx, ny, time + 1))
                elif (time + 1) % K == 0:
                    visited[nx][ny][next_time] = True
                    queue.append((nx, ny, time + 1))
    
    return "Oop!"

def main():
    T = int(input()) 
    for _ in range(T):
        R, C, K = map(int, input().split()) 
        Map = [input().strip() for _ in range(R)]  
        start = None
        end = None
        for i in range(R):
            for j in range(C):
                if Map[i][j] == 'S':
                    start = (i, j)
                elif Map[i][j] == 'E':
                    end = (i, j)        
        result = bfs(R, C, K, Map, start, end)
        print(result)
if __name__ == "__main__":
    main()
```

螃蟹拆蘑菇   两个点

```python
from collections import deque

# 定义四个方向：右、下、左、上
dire = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def bfs(a, x1, y1, x2, y2):
    visit = set()  # 使用集合来避免重复访问
    queue = deque([(x1, y1, x2, y2)])
    visit.add((x1, y1, x2, y2))  # 初始点加入访问集合

    while queue:
        xa, ya, xb, yb = queue.popleft()
        # 遍历四个方向
        for xi, yi in dire:
            # 计算新位置
            nx1, ny1 = xa + xi, ya + yi
            nx2, ny2 = xb + xi, yb + yi

            # 判断新位置是否合法
            if 0 <= nx1 < a and 0 <= ny1 < a and 0 <= nx2 < a and 0 <= ny2 < a:
                if (nx1, ny1, nx2, ny2) not in visit and Matrix[nx1][ny1] != 1 and Matrix[nx2][ny2] != 1:
                    # 加入队列并标记访问
                    queue.append((nx1, ny1, nx2, ny2))
                    visit.add((nx1, ny1, nx2, ny2))
                    # 检查是否到达目标
                    if Matrix[nx1][ny1] == 9 or Matrix[nx2][ny2] == 9:
                        return True
    return False

# 读取输入
a = int(input())
Matrix = [list(map(int, input().split())) for _ in range(a)]

# 找到第一个和第二个 '5' 的位置
x1, y1, x2, y2 = -1, -1, -1, -1
found_first = False

for i in range(a):
    for j in range(a):
        if Matrix[i][j] == 5:
            if not found_first:
                x1, y1 = i, j
                Matrix[i][j] = 0  # 标记为已访问
                found_first = True
            else:
                x2, y2 = i, j
                Matrix[i][j] = 0  # 标记为已访问
                break
    if x2 != -1:  # 如果第二个 5 已经找到
        break

# 运行 BFS 检查是否可以从 (x1, y1) 到 (x2, y2)
check = bfs(a, x1, y1, x2, y2)
print('yes' if check else 'no')
```
