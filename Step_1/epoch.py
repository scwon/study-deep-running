# 에포크를 통해, 반복적으로 가중치와 절편을 업데이트하기
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

x = diabetes.data[:, 2] # 입력 데이터 
y = diabetes.target # 타겟 (예측값과 비교할 실제 결과값)

w = 1.0 # 가중치
b = 1.0 # 절편

# 오차역전파 함수
# w: 가중치, b: 절편, x: 인풋값, y: 실제 결과값 
def backpropagation(w, b, x, y):
  y_hat = x * w + b
  err = y - y_hat
  w_rate = x
  b_rate = 1
  w_new = w + w_rate * err
  b_new = b + b_rate * err

  return w_new, b_new

for i in range(0, 100):
  for j in range(0, len(x)):
    w, b = backpropagation(w, b, x[j], y[j])

print('w ==', w, 'b ==', b)

print('x[0] 의 예측값: ', 0.18 * w + b)