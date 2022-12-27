from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

x = diabetes.data[:, 2] # 입력 데이터 
y = diabetes.target # 타겟 (예측값과 비교할 실제 결과값)

w = 1.0 # 가중치
b = 1.0 # 절편

y_hat = x[0] * w + b # 예측값 구하기
print("예측값 = ", y_hat)

b_inc = b + 0.3
y_hat_inc = x[0] * w + b_inc
print("절편을 0.1 올렸을 때, 예측값 = ", y_hat_inc)

b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print("졀편에 변화에 따른 예측값의 변화량 =", b_rate)

# 가중치를 1로 고정하고 절편만 업데이트해서 예측값을 근사치로 만드는 함수
def interceptUpdate(x, y, b, b_rate):
  y_hat = x * 1 + b
  sign = 1
  if(y < y_hat): sign = -1
  while(abs(y - y_hat) > b_rate):
    y_hat = x * 1 + b
    b += b_rate * sign

  if(abs(y - y_hat) > abs(y - (y_hat + b_rate * sign))): y_hat += b_rate
  return y_hat

for i in range(0,10):
  print(i + 1, "번째 ========")
  print("입력값: ", x[i], "실제 값: ", y[i])
  print("예측값: ", interceptUpdate(x[i], y[i], 1, b_rate))