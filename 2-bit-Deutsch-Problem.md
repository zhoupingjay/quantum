
# 用谷歌Cirq模拟双比特Deutsch问题


## 谷歌量子计算框架Cirq
Cirq是谷歌开发的开源量子计算框架，它让开发者能够方便的用Python搭建量子电路。您可能会问，我没有量子计算机，如何运行我设计的电路呢？别急，Cirq内置了模拟器，您在自己的电脑上就可以模拟您设计的量子电路。如果想在真实的量子计算机上运行，Cirq也提供了相应的接口。

Cirq的源码：https://github.com/quantumlib/Cirq

Cirq的安装，最简单的方法是用pip：

```
python -m pip install --upgrade pip
python -m pip install cirq
```

安装完成，您就可以在电脑上模拟量子计算了！

接下来进入实战，我们要用Cirq来模拟双比特Deutsch问题。

## 回顾一下双比特Deutsch问题

回顾一下双比特Deutsch问题：有人给你一个函数f，不告诉你它内部是怎么运作的，只知道它输入2个比特，输出1个比特，并且f有可能是常量函数（无论什么输入，输出总是0或者总是1），也有可能是平衡函数（对所有可能的输入，一半的情况下输出0，另一半情况输出1）。我们只能通过对f的查询，来判断f的性质。在上文我们讨论过，这样的函数f总共有8种可能（2个常量函数，6个平衡函数）。所以，提问的人实际上是从这8个可能的f函数中，随机抽一个给我们来判断。

<img src="./images/2-bit deutsch.svg" width="400" />

我们知道量子计算里，每个这样的函数f，都可以包装成相应的可逆变换Uf。所以在量子计算环境下，提问的人实际上是把这8种可能的函数f包装成8个Uf，从里面随机抽一个作为黑盒给我们，让我们判断相应的函数f的性质。

所以要模拟双比特Deutsch问题，我们首先要从提问者的角度，把这8个可能的Uf准备出来。

**我们从一个简单的例子出发：**

假如有这样一个平衡函数函数f，它在输入是00或01的情况下输出0，其余情况下输出1，我们把它和相应的Uf输出列在下面的表里：

| 输入 (x0x1) | $$f(x_0x_1)$$ | $$y\oplus f(x_0x_1)$$ |
|------------|---------------|---------------------|
| 00 | 0 | y  |
| 01 | 0 | y |
| 10 | 1 | y取反  |
| 11 | 1 | y取反 |

从上面这个表中可以看出，Uf的第三个输出端，其状态取决于第一个输入：

如果x0=1，那么在第三个输出端输出，反之如果x0=1，那么就输出。这在量子电路里，就是一个简单的CNOT门：CNOT(x0, y)。所以对于上面这个表里的f函数，包装成相应的Uf内部就是一个简单的CNOT门实现。

<img src="./images/2-bit deutsch 2.svg" width="240" />

**以此类推，对每一个函数f，我们都可以用量子电路来实现相应的Uf。**

我们把这些Uf放在一个池子里，模拟的时候从里面随机选一个，作为提问者的问题。

## 用Cirq模拟双比特Deutsch问题

### 首先我们要导入Cirq包：


```python
import cirq
import random
```

### 然后生成这个Uf池子：


```python
q0, q1, q2 = cirq.LineQubit.range(3)
constant = [[], [cirq.X(q2)]]
balanced = [[cirq.CNOT(q0, q2)],
            [cirq.CNOT(q1, q2)],
            [cirq.CNOT(q0, q2), cirq.CNOT(q1, q2)],
            [cirq.CNOT(q0, q2), cirq.X(q2)],
            [cirq.CNOT(q1, q2), cirq.X(q2)],
            [cirq.CNOT(q0, q2), cirq.CNOT(q1, q2), cirq.X(q2)]]

Uf_pool = [
    [],            # 常量函数: f(x0x1) = 0
    [cirq.X(q2)],  # 常量函数: f(x0x1) = 1
    
    [cirq.CNOT(q0, q2)], # 平衡函数: f(00/01)=0, f(10/11)=1
    [cirq.CNOT(q1, q2)], # 平衡函数
    [cirq.CNOT(q0, q2), cirq.CNOT(q1, q2)], # 平衡函数
    [cirq.CNOT(q0, q2), cirq.X(q2)],        # 平衡函数
    [cirq.CNOT(q1, q2), cirq.X(q2)],        # 平衡函数
    [cirq.CNOT(q0, q2), cirq.CNOT(q1, q2), cirq.X(q2)] # 平衡函数
]
Uf_attributes = [
    "constant",
    "constant",
    "balanced",
    "balanced",
    "balanced",
    "balanced",
    "balanced",
    "balanced",
]
```

### 根据Uf生成量子电路

接下来，我们要写一个函数，根据提问者给出的Uf，生成相应的量子电路。

<img src="./images/2-bit deutsch 3.svg" width="400" />


```python
def generate_two_bit_deutsch(Uf):
    # 默认初始输入都是|0>, 所以q2先要用X变成|1>
    circuit = cirq.Circuit([cirq.X(q2)])
    circuit.append([cirq.H(q0), cirq.H(q1), cirq.H(q2)],
                   strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    
    # 接入提问者给的Uf
    circuit.append(Uf, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    
    # 输出端再加上H和M门
    circuit.append([cirq.H(q0), cirq.H(q1)],
                  strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    circuit.append([cirq.measure(q0), cirq.measure(q1)],
                  strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit
```

试试这个函数生成的量子电路：


```python
# 用Uf_pool[4]生成电路，并打印出来
print(generate_two_bit_deutsch(Uf_pool[4]))
```

    0: ───────H───@───────H───M───
                  │
    1: ───────H───┼───@───H───M───
                  │   │
    2: ───X───H───X───X───────────


两边H门之间的就是提问者给的Uf。


```python
# 测试一下，把池子里所有的电路都打印出来
for pick in range(0,8):
    Uf = Uf_pool[pick]
    Uf_attr = Uf_attributes[pick]
    circuit = generate_two_bit_deutsch(Uf)
    print("Circuit for Uf_pool[{}]: {}".format(pick, Uf))
    print(circuit)
```

### 模拟提问和回答的过程

接下来我们来模拟提问和回答的过程：
- 提问者随机选一个Uf给我们
- 我们用这个Uf生成量子电路
- 用模拟器运行生成的量子电路
- 根据电路运行结果，判断函数f的性质

模拟过程代码如下：


```python
# 提问者随机选一个Uf给我们
random.seed()
pick = random.randint(0, 7)
Uf = Uf_pool[pick]
Uf_attr = Uf_attributes[pick]
print("Random pick Uf[{}]: {}".format(pick, Uf))

# 我们用这个Uf生成量子电路
circuit = generate_two_bit_deutsch(Uf)
print("Generated circuit")
print(circuit)

simulator = cirq.Simulator()
result = simulator.run(circuit)
print(result)

# simulator 返回的是一个pandas DataFrame
if result.data.loc[0][0] == 0 and result.data.loc[0][1] == 0:
    print("Our result:    constant")
    print("Actual result: {}".format(Uf_attr))
else:
    print("Our result:    balanced")
    print("Actual result: {}".format(Uf_attr))
```

    Random pick Uf[1]: [cirq.X.on(cirq.LineQubit(2))]
    Generated circuit
    0: ───────H───────H───M───
    
    1: ───────H───────H───M───
    
    2: ───X───H───X───────────
    0=0
    1=0
    Our result:    constant
    Actual result: constant


多跑几次，可以看到无论提问者抽哪个Uf，我们的量子电路总能判断出正确的函数性质。而且更重要的是，**我们只需要对函数f进行一次查询！**
