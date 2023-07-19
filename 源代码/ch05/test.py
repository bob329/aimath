import numpy as np
from sympy import *
import pandas as pd
'''
A = [1,2,3,4,5,6]
B = np.array(A)
C1 = B.reshape(2,3)
C2 = B.reshape(3,2)
print("矩阵B:\n",B)
print("转换为2行3列矩阵C1:\n",C1)
print("转换为3行2列矩阵C2:\n",C2)
print("输出C1的第0行元素C1[0]:\n",C1[0])
print("输出C1的前2行元素C1[0:2]:\n",C1[0:2])
print("输出C2的第0行元素和第2行元素C2[[0:2]]:\n",C2[[0,2]])
print("输出C2的第1列元素C2[:,1]:\n",C2[:,1])
print("输出C2的前2列元素C2[:,0:2]:\n",C2[:,0:2])
print("输出C1的第0列和第2列元素C1[:,[0,2]]:\n",C1[:,[0,2]])
print("输出C2的第2行第1列元素C2[2,1]:\n",C2[2,1])

A = [[1,2,3,4,5]]
B = [[1],[2],[3],[4],[5]]
C = np.array(A)
D = np.array(B)
print("行向量C:\n",C)
print("列向量D:\n",D)
print("A的类型：%s,C的类型：%s"%(type(A),type(C)))
print("B的类型：%s,D的类型：%s"%(type(B),type(D)))
print("C的大小：%s,D的大小：%s"%(C.shape,D.shape))

arr1 = np.random.random((1,2,3))  #默认范围为0~1
arr2 = np.random.random((3,2,1))  #默认范围为0~1
arr3 = np.random.randint(3,10,size=(1,2,3))
arr4 = np.random.randint(3,10,size=(3,2,1))
print("创建的三维行向量(由随机浮点数组成)：\n",arr1)
print("创建的三维列向量(由随机浮点数组成)：\n",arr2)
print("创建的三维行向量(由3~30(不包括30)的随机整数组成)：\n",arr3)
print("创建的三维列向量(由3~30(不包括30)的随机整数组成)：\n",arr4)

A = [1,2,3,4,5]
B = np.array(A)
C = B.reshape(1,5)
D = B.reshape(5,1)
print("一维数组B：\n",B)
print("行向量C：\n",C)
print("列向量D:\n",D)
print("B的维数：",B.shape)
print("C的维数：",C.shape)
print("D的维数：",D.shape)

arr1 = np.zeros(10)
arr2 = np.zeros((3,4))
arr3 = np.array([np.zeros(10)])
print("通过zeros函数创建的零数组arr1：\n",arr1)
print("通过zeros函数创建的零数组arr2：\n",arr2)
print("通过zeros函数创建的零数组arr3：\n",arr3)
print("arr1的形状",np.shape((arr1)))
print("arr2的形状",np.shape((arr2)))
print("arr3的形状",np.shape((arr3)))

arr = np.random.randint(1,16,size=[3,3])
print("元素值为1~16随机整数的三阶方阵：\n",arr)

E1 = np.eye(3)
E2 = np.identity(3)
print("通过eye()创建的三阶单位矩阵E1：\n",E1)
print("通过identity()创建的三阶单位矩阵E2：\n",E2)

a = [1,2,3] #对角线元素
arr1 = np.diag(a) #创建对角矩阵
print("创建主对角线为1,2,3的对角矩阵arr1：\n",arr1)
arr2 = np.diag(arr1)
print("获取矩阵arr1的对角线元素：\n",arr2)
print("arr2的类型",arr2.shape)

A = [[1,2,3],[4,5,6],[7,8,9]]
C = np.array(A)
arr = np.diag(np.diag(C))
print("原始矩阵C：\n",C)
print("根据C的对角线元素生成的对角矩阵：\n",arr)

C = np.array([[1,2,3],[4,5,6]])
arr = np.diag(C)
print("获取矩阵C的对角线元素：\n",arr)

A = np.array([[1,2,3,2],[4,5,6,3],[7,8,9,4],[3,5,6,8]])
upper_A = np.triu(A,0)
low_A = np.tril(A,0)
print("A矩阵：\n",A)
print("A的上三角矩阵：\n",upper_A)
print("A的下三角矩阵：\n",low_A)

A = [3,4]
B = [4,3]
arr1 = np.random.randint(3,9,size=A)
arr2 = np.random.randint(10,30,size=A)
arr3 = np.random.randint(50,100,size=B)
print("arr1=\n",arr1)
print("arr2=\n",arr2)
print("arr3=\n",arr3)
print("arr1与arr2是否同型：",np.shape(arr1)==np.shape(arr2))
print("arr1与arr3是否同型：",np.shape(arr1)==np.shape(arr3))

A = np.array([[1,1,1],[1,2,4],[1,3,9]])
B = np.array([2,3,5])
C = np.array([[1,1,1],[1,2,4],[1,3,9]])
#利用allclose()检验矩阵是否相等，True代表相等，False代表不等
print("A和B是否相等：",np.allclose(A,B))
print("A和C是否相等：",np.allclose(A,C))

A = [[1,2,3],[3,2,1]]
B = [[6,8,12],[10,5,12]]
C = np.array(A)
D = np.array(B)
print("C+D=\n",C+D)
print("C-D=\n",C-D)

A = [[1,2,3],[4,5,6]]
C = np.array(A)
print("矩阵的数承2C=\n",2*C)

#矩阵乘法：
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])
C =np.dot(A,B)
#或者C = A.dot(B)
print("矩阵的乘法：\n",C)
#一维数组乘法
one_vec1 = np.array([1,2,3])
one_vec2 = np.array([4,5,6])
one_multi_result = np.dot(one_vec1,one_vec2)
print("一维数组的乘法：\n",one_multi_result)

#A.B为同型矩阵
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[7,8,9],[4,7,1]])
#方法1，使用*实现对应元素的乘积
C1 = A * B
print("方法1，使用*实现对应元素的乘积：\n",C1)
#方法2，使用np.multiply()函数实现对应元素的乘积
C2 = np.multiply(A,B)
print("方法2，使用np.multiply()函数实现对应元素的乘积：\n",C2)

A = np.mat([[1,2,3],[4,5,6]])
B = np.mat([[1,2],[3,4],[5,6]])
C = np.mat([[3,2,3],[5,4,6]])
#方法1，使用np.dot()函数实现矩阵相乘
D1 = np.dot(A,B)
#方法2，使用*实现矩阵相乘
D2 = A*B
E = np.multiply(A,C)
print("方法1，用np.dot()函数实现矩阵乘法：\n",D1)
print("方法2，用*实现矩阵乘法：\n",D2)
print("用np.multiply()函数实现矩阵A和C对应元素的乘积：\n",E)

#矩阵与向量的乘法
two_matrix = np.array([[1,2,3],[4,5,6]])
vector = np.array([[7],[8],[9]])
result = np.dot(two_matrix,vector)
print(two_matrix.shape)
print(vector.shape)
print("矩阵与列向量的乘法：\n",result)

#矩阵的乘方：
A = [[1,2,3],[4,5,6],[7,8,9]]
A_array = np.array(A)
A_matrix = np.mat(A)
B = A_array.dot(A_array).dot(A_array)
C = A_matrix**3
D = A_array**3
print("array的三次方：\n",B)
print("matrix的三次方：\n",C)
print("array元素的三次方：\n",D)

A = [[2,6,10],[1,-2,9]]
B = np.array(A)
print("采用np.transpose()函数求B的转置矩阵：\n",np.transpose(B))
print("采用T属性求B的转置矩阵：\n",B.T)

A = [[1,2,3],[4,5,6]]
B = [[7,8],[9,10],[11,12]]
C = np.array(A)
D = np.array(B)
print("矩阵相乘后的转置结果：\n",(C.dot(D)).T)
print("矩阵转置后相乘的结果：\n",D.T.dot(C.T))

#创建一个方阵
arr1 = np.random.randint(1,16,size=[3,3])
#保留其上三角部分
arr2 = np.triu(arr1)
#生成对称矩阵
arr3 = arr2 + arr2.T - np.diag(np.diag(arr2))#将上三角“拷贝”到下三角部分
print("创建的方阵arr1：\n",arr1)
print("创建的方阵arr2：\n",arr2)
print("创建的方阵arr2.T：\n",arr2.T)
print("创建的方阵np.diag(arr2)：\n",np.diag(arr2))
print("创建的方阵np.diag(np.diag(arr2))：\n",np.diag(np.diag(arr2)))
print("生成的对称矩阵arr3：\n",arr3)

arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[1,2,3],[2,4,5],[3,5,6]])
print("arr1是否为对称矩阵：\n",np.allclose(arr1,arr1.T))
print("arr2是否为对称矩阵：\n",np.allclose(arr2,arr2.T))

A = [[1,2,3,4,5]]
B = np.array(A)
C = B.T
print("行向量B=\n",B)
print("列向量C=\n",C)

A = [[1,2],[2,5]]
C1 = np.array(A)
C2 = np.mat(A)
C1_inverse = np.linalg.inv(C1)
C2_inverse = C2.I
print("通过inv()求出C1的逆矩阵：\n",C1_inverse)
print("通过I属性求出C2的逆矩阵：\n",C2_inverse)
print("C1与C1的逆相乘的结果：\n",np.dot(C1,C1_inverse))

A = [[1,-4,0,2],[-1,2,-1,-1],[1,-2,3,5],[2,-6,1,3]]
B = np.array(A)
B_inverse = np.linalg.inv(B)    #求B的逆矩阵
print("B的逆矩阵：\n",B_inverse)

B = [[1,-2,3],[-1,2,1],[-3,-4,-2]]
A = np.array(B)
C = np.linalg.det(A)
print("A的行列式的值：\n",C)
print("C的-1次方：\n",C**(-1))
print("1/C:\n",1/C)

A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("A的第1行行向量：\n",A[1,:].reshape(1,-1))
print("A的第2列列向量：\n",A[:,2].reshape(-1,1))

A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("A的各行向量：")
for i in range(np.shape(A)[0]):
    print("第",i,"行：",A[i,:].reshape(1,-1))
print("A的各列向量：")
for i in range(np.shape(A)[1]):
    print("第",i,"列\n",A[:,i].reshape(-1,1))

E = np.eye(4)
print("单位矩阵E的秩：",np.linalg.matrix_rank(E))
A = [[1,-4,0,2],[-1,2,-1,-1],[1,-2,3,5],[2,-6,1,3]]
B = np.array(A)
print("B的秩：",np.linalg.matrix_rank(B))

A = [[1,2,3]]
B = [[4,5,6]]
C1 = np.array(A).reshape(3,1)   #C1为三维列向量
C2 = np.array(B).reshape(3,1)   #C2为三维列向量
D1 = np.dot(C1.T,C2)
D2 = np.dot(C2.T,C1)
print("向量C1和C2的内积：\n",D1)
print("向量C2和C1的内积：\n",D2)

A = np.array([1,2,3])
B = np.array([4,5,6])
print("使用dot实现A和B的内积：\n",np.dot(A,B))
print("使用innre实现A和B的内积：\n",np.inner(A,B))

A = np.array([[0,3,4]])
B = np.linalg.norm(A)
print("向量A的长度：",B)
C = A/B         #单位向量=向量 / 长度
print(A,"对应的单位向量=",C)
D = np.linalg.norm(C)
print("单位向量的长度=",D)

A = np.array([0,3,4])
B = np.sum(A**2)**0.5
print("向量A的长度：",B)

A = np.array([[1,1]])
B = np.array([[2,0]])
C = A.dot(B.T)
print("向量A和B夹角的余弦值：",C/(np.linalg.norm(A)*np.linalg.norm(B)))

A = np.array([[0,1,0],[1/2**0.5,0,1/2**0.5],[-1/2**0.5,0,1/2**0.5]])
print("A×A.T=\n",np.round(A.dot(A.T),0))

A = np.array([[1,2,1],[2,-1,3],[3,1,2]])#系数矩阵
B = np.array([7,7,18]).reshape(3,1)     #系数矩阵
print("系数矩阵A的大小：",A.shape)
print("系数矩阵A的秩：",np.linalg.matrix_rank(A))
AB = np.hstack((A,B))                   #增广矩阵
print("增广矩阵的秩：",np.linalg.matrix_rank(AB))
print("增广矩阵：\n",AB)

A = np.array([[1,2,1],[2,-1,3],[3,1,2]])    #系数矩阵A
B = np.array([7,7,18]).reshape(3,1)         #常数项矩阵B
A_inv = np.linalg.inv(A)                    #A的逆矩阵
X = A_inv.dot(B)
print("A的逆矩阵：\n",A_inv)
print("利用逆矩阵求出X的值：\n",X)
C = np.dot(A,X)
print("A和X的乘积C：\n",C)
#利用allclose()函数检验矩阵是否相等，True代表等，False代表不等
print("B和C是否相等",np.allclose(C,B))

A = np.array([[1,2,1],[2,-1,3],[3,1,2]])    #系数矩阵A
B = np.array([7,7,18]).reshape(3,1)         #常数项矩阵B
X = np.linalg.solve(A,B)
print("利用逆矩阵求出X的值：\n",X)
C = np.dot(A,X)
print("A和X的乘积C：\n",C)
#利用allclose()函数检验矩阵是否相等，True代表等，False代表不等
print("B和C是否相等",np.allclose(C,B))

x,y,z = symbols("x y z")    #3个变量
eq = [x+2*y+z-7,2*x-y+3*z-7,3*x+y+2*z-18]
result = solve(eq,[x,y,z])
print("结果：",result)

A = np.array([[1,2,1,-2],[2,3,0,-1],[1,-1,-5,7]])   #系数矩阵
print("系数矩阵A的大小：",A.shape)
print("系数矩阵A的秩：",np.linalg.matrix_rank(A))

x, y, z, w = symbols("x y z w")
eq = [x+2*y+z-2*w,2*x+3*y-w,x-y-5*z+7*w]
result = solve(eq,[x,y,z,w])
print("结果是：",result)
A = {z:1,w:2}
x =float(result[x].evalf(subs=A))
y =float(result[y].evalf(subs=A))
print("x=",x," y=",y," z=",1," w=",2)

A = np.array([[1,2],[3,4]])
B = np.array([[1,0],[2,3]])
C1 = (A+B).dot(A+B)
C2 = A.dot(A)+2*A.dot(B)+B.dot(B)
C3 = A.dot(A)+A.dot(B)+B.dot(A)+B.dot(B)
D1 = (A+B).dot(A-B)
D2 = A.dot(A)-B.dot(B)
D3 = A.dot(A)-A.dot(B)+B.dot(A)-B.dot(B)
E = A.dot(B)
E1 = A.dot(B).dot(A.dot(B))
E2 = A.dot(A).dot(B).dot(B)
print("(A+B)的平方=\n",C1)
print("A平方+2AB+B方法=\n",C2)
print("A平方+AB+BA+B平方=\n",C3)
print("(A+B)(A-B)=\n",D1)
print("A平方-B平方=\n",D2)
print("A平方-AB+BA-B平方=\n",D3)
print("AB=\n",E)
print("AB平方=\n",E1)
print("A平方与B平方的乘积=\n",E2)

A = np.array([[1,2,3,1],[4,5,6,0],[7,8,9,1]])
print(pd.DataFrame(A))

A = np.arange(0,10,1)
print("A=",A)
print("A.shape=",A.shape)

A = np.linspace(1,10,10)
B = np.linspace(1,10,10, endpoint = False)
C = np.linspace(1,10,10, endpoint = False, retstep = True)
print("A=",A)
print("B=",B)
print("C=",C)

A = np.logspace(0,2,5)  #从10到0次到10的二次方，有5个元素的等比数列
print("A=",A)
B = np.logspace(0,6,3,base=2)
print("B=",B)           #从2的零次方到2的六次方，有3个元素的等比数列

A = np.array([[1,2,3],[4,5,6]])
print("矩阵A：\n",A)            #获得整个矩阵的最大值，结果：6
print("整个矩阵的最大值：",A.max()) #获得整个矩阵的最大值，结果：6
print("整个矩阵的最小值：",A.min()) #结果：1
print("每列的最大值：",A.max(axis=0))   #结果：[4 5 6]
print("每行的最大值：",A.max(axis=1))   #结果：[3 6]
#要想获得最大最小值元素所在的位置，可以通过argmax()函数获得
print("每列的最大值的位置：",A.argmax(axis=1))  #结果：[2 2]
print("矩阵求和：",A.sum())         #对整个矩阵求和，结果：21
print("按列求和：",A.sum(axis=0))   #对列方向求和，结果：[5 7 9]
print("按行求和：",A.sum(axis=1))   #对行方向求和，结果：[6 15]
print("整个矩阵的平均值：",A.mean()) #结果： 3.5
print("每列的平均值：",A.mean(axis=0))  #结果： [2.5 3.5 4.5]
print("所有数取中值：",np.median(A))    #对所有数取中值，结果：3.5
print("按列取中值：",np.median(A,axis=0))   #结果：[2.5 3.5 4.5]
print("按行取中值：",np.median(A,axis=1))   #结果：[2. 5.]

A = np.array([[1, 1, 1], [1,2,4], [1, 3, 9]])   #系数矩阵
B = np.array([2, 3, 5]).reshape(3,1)            #系数矩阵
A_inv=np.linalg.inv(A)                  #A的逆矩阵
X=A_inv.dot(B)                        #未知数矩阵X
print("A的逆矩阵为：\n",A_inv)
print("利用逆矩阵求出X的值为：\n",X)
C=np.dot(A, X)                  #系数矩阵A与X乘积
print("A和X的乘积C为：\n",C)
#利用allclose检验矩阵是否相等，True代表相等，False代表不等
print("B和C是否相等：",np.allclose(C, B))

A = np.array([[1, 1, 1], [1,2,4], [1, 3, 9]])   #系数矩阵
B = np.array([2, 3, 5]).reshape(3,1)            #系数矩阵
X = np.linalg.solve(A,B)
print("利用solve()求出X的值为：\n",X)
C = np.dot(A,X)
print("A和X的乘积C为：\n",C)
#利用allclose检验矩阵是否相等，True代表相等，False代表不等
print("B和C是否相等：",np.allclose(C, B))

x, y, z = symbols("x y z")#三个变量
eq = [x+y+z-2,x+2*y+4*z-3,x+3*y+9*z-5]#将三个公式改写为等式为0
result=solve(eq,[x,y,z])
print("结果是：",result)

arr1 = np.random.randint(1,21,(3,4))
arr2 = np.random.randint(1,21,(4,5))
print("arr1=\n",arr1)
print("arr2=\n",arr2)
mult = np.dot(arr1,arr2)
print("arr1的秩为：",np.linalg.matrix_rank(arr1))
print("arr2的秩为：",np.linalg.matrix_rank(arr2))
print("arr1和arr2的乘积：\n",mult)
print("arr1和arr2的和：\n",arr1+arr2)

A =[[1,2,3],[2,2,1],[3,4,3]]
B=np.array(A)
C=np.linalg.det(B)
print("B的行列式的值：\n",C)
B2=np.mat(A)
C1_inverse = np.linalg.inv(B)     #求C1的逆矩阵，不能使用I方法
C2_inverse = B2.I                #求C2的逆矩阵
print("通过inv()求出B的逆矩阵：\n",C1_inverse)
print("通过I属性求出B2的逆矩阵：\n",C2_inverse)

A = np.zeros((4,4))
print("通过zeros函数创建的四阶零矩阵A：\n",A)
E1= np.eye(4)
E2= np.identity(4)
print("通过eye()创建的四阶单位矩阵E1为：\n",E1)
print("通过identity ()创建的四阶单位矩阵E2为：\n",E2)
b=[1,2,3,4]  #对角线元素
arr1=np.diag(b)  #使用diag()创建对角矩阵
print("创建主对角线为1,2,3,4的对角矩阵arr1为：\n",arr1)
arr2=np.diag(arr1)
print("获取矩阵arr1的对角线元素：\n",arr2)
print("arr2的类型",arr2.shape)

arr=np.random.uniform(1,21,(4,4))
upper_A=np.triu(arr,0)#上三角矩阵
low_A=np.tril(arr,0)#下三角矩阵
print("arr矩阵：\n",arr)
print("A的上三角矩阵：\n",upper_A)
print("A的下三角矩阵：\n",low_A)
#生成上三角矩阵的对称矩阵
upper_A += upper_A.T - np.diag(np.diag(upper_A))#将上三角”拷贝”到下三角部分
print("生成的对称矩阵arr2：\n",upper_A)
#生成下三角矩阵的对称矩阵
low_A += low_A.T - np.diag(np.diag(low_A))#将上三角”拷贝”到下三角部分
print("生成的对称矩阵arr2：\n",low_A)
'''