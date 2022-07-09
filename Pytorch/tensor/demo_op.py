import torch

## tensor基本运算
## 带下划线会修改变量本身

class Basic():
    def __init__(self) -> None:
        self.a = torch.rand(2, 3)
        self.b = torch.rand(2, 3)
        print('a=', a)
        print('b=', b)
        print('======================================')

    def add(self):
        print(self.a + self.b)
        print(self.a.add(self.b))
        print(torch.add(self.a, self.b))
        print(self.a.add_(self.b))
        print('a=', self.a)

    def Sub(self):
        print(self.a-self.b)
        print(torch.sub(self.a, self.b))
        print(self.a.sub(self.b))
        print(self.a.sub_(b))
        print('a =', self.a )

    def mul(self):
        print(self.a * self.b)
        print(torch.mul(self.a, self.b))
        print(self.a.mul(self.b))
        print(self.a.mul_(self.b))
        print('a=', self.a)

    def div(self):
        print(self.a/self.b)
        print(torch.div(self.a,self.b))
        print(self.a.div(self.b))
        print(self.a.div_(self.b))
        print('a=', self.a)

## matrix operation
class Matrix():
    def __init__(self) -> None:
        self.a = torch.ones(2, 1)
        self.b = torch.ones(1, 2)

    def mat_op(self):
        print(self.a @ self.b)
        print(self.a.matmul(self.b))
        print(torch.matmul(self.a, self.b))
        print(torch.mm(self.a, self.b))
        print(self.a.mm(self.b))

## 高维tensor运算
## 保证最后两个维度可以进行矩阵计算
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 5)

print(a.matmul(b).shape)
print('============= POW ================')

## Pow
a = torch.tensor([1, 2])
print(torch.pow(a, 3))
print(a.pow(3))
print(a ** 3)
print(a.pow_(3))
print('a=', a)

print('============EXP & LOG================')
## exp
a = torch.tensor([1, 2], dtype=torch.float)
print(torch.exp(a))
print(torch.exp_(a))
print(a.exp())
print(a.exp_())

print(torch.log(a))
print(torch.log_(a))
print(a.log())
print(a.log_())

print('================SQRT================')
print(torch.sqrt(a))
print(torch.sqrt_(a))
print(a.sqrt())
print(a.sqrt_())
# agent = Matrix()
# agent.mat_op()