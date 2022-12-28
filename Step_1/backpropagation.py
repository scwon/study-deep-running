# 오차 역전파를 사용하여 가중치와 절편 업데이트하기
# 오차 역전파 => 예측값과 실제 값의 차이를 이용하여 가중치와 절편 값을 업데이트 하는 과정
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

x = diabetes.data[:, 2] # 입력 데이터 
y = diabetes.target # 타겟 (예측값과 비교할 실제 결과값)

w = 1.0 # 가중치
b = 1.0 # 절편

# count = 테스트할 데이터의 수
def getBackpropagation(w, b, count, index = 0):
  y_hat = x[index] * w + b # 초기 예측값
  w_rate = x[index] # 가중치 변화율
  b_rate = 1 # 절편 변화율

  err = y[index] - y_hat # 예측값과 실제 값의 차이
  w_new = w + w_rate * err # 새로운 가중치
  b_new = b + b_rate * err # 새로운 절편

  print('인풋 값:', x[index])
  print('실제 값: ', y[index])
  print('before 가중: ', w, '절편: ', b, '예측값: ', y_hat)
  print('새로운 가중치: ', w_new, '새로운 절편: ', b_new)
  print('새로운 예측값: ', x[index] * w_new + b_new)
  if(index < count):
    return getBackpropagation(w_new, b_new, count, index + 1)
  return w_new, b_new


w_new, b_new = getBackpropagation(w, b, len(x) - 1)

print('구한 가중치와 절편으로 x[0] 예측값 구하기', x[0] * w_new + b_new)