# 이미 있는 데이터셋을 이용해서 예측값 찾기?
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
# plt.show()


x = diabetes.data[:, 2] # 입력 데이터 
y = diabetes.target # 타겟 (예측값과 비교할 실제 결과값)

print('x', x)
print('y', y)

w = 1.0 # 가중치
b = 1.0 # 절편

y_hat = x[0] * w + b # 예측값 구하기
print("y_hat ===", y_hat, "y[0] ===", y[0])

w_inc = w - 0.1
y_hat_inc = x[0] * w_inc + b
print("y_hat_inc ===", y_hat_inc)
print("가중치 증가량에 따른 예측값의 변화량 =", (y_hat_inc - y_hat) / (w_inc - w))


'''
변화율로 가중치를 업데이트해 예측값을 근사치로 만드는 함수 (절편을 배제)
'''
def weightUpdate(x, y, w):
  predicted_value = x * w
  while(y - predicted_value > x):
    predicted_value = x * w
    w += x
  return predicted_value

print("인풋: ", x[0], " 실제 결과", y[0])
print("예측값 = ", weightUpdate(x[0], y[0], 1))

