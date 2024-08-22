import mod1

# import 모듈이름 :
    # 모듈이름.함수명()
    # mod1.add(3, 4)

# [2] from 모듈이름 import 함수명
from mod1 import add
add(3, 4)   # 함수명

# [3] from 모듈이름 import *
from mod1 import *
sub(3, 4)



# [4]
import mod2
print(mod2.PI)
a = mod2.Math()
print(a)
print(a.solv(2))

print(mod2.add(3, 4))

from mod2 import Math, PI
print(PI)
b = Math()
print(b)

# [5] 다른 패키지의 모듈 호출
from src.day06.Task6 import nameAge

