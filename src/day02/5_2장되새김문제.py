# 5_2장되새김문제.py

# Q1.
print ("1.")
kor = 80     # 국어 점수
eng = 75     # 영어 점수
math = 55    # 수학 점수
sum = kor + eng + math  # 합계
avg = sum / 3           # 평균
print(avg)              # 70.0


# Q2.
print ("2.")
if (13 % 2 == 0) :
    print("짝수")
else :
    print("홀수")        # 홀수


# Q3.
print ("3.")
pin = "881120-1068234"
yyyymmdd = pin[0:6]
num = pin[7:14]
print(yyyymmdd)     # 연월일 부분  881120
print(num)          # 숫자 부분    1068234


# Q4.
print ("4.")
pin = "881120-1068234"
print(pin[7])       # 1


# Q5.
print ("5.")
a = "a:b:c:d"
b = a.replace(":", "#")
print(b)            # a#b#c#d


# Q6.
print ("6.")
a = [ 1, 3, 5, 4, 2 ]
a.sort()            # [1, 2, 3, 4, 5]로 변경
a.reverse()         # [5, 4, 3, 2, 1]로 변경
print(a)            # [5, 4, 3, 2, 1]


# Q7.
print ("7.")
a = ['Life', 'is', 'too', 'short']
result = " ".join(a)
print(result)       # Life is too short


# Q8.
print ("8.")
a = (1, 2, 3)
a = a + (4,)
print(a)            # (1, 2, 3, 4)



# Q9.
# print ("9.")
# a = dict()
# a
# a['name'] = 'python'
# a[('a', )] = 'python'
# a[[1]] = 'python'
# a[250] = 'python'


# Q10.
print ("10.")
#a = {'A':90, 'B':80, 'C':70}


# Q11.
# print ("11.")
# a = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5,]
# aSet =
# b =
# print(b)



# Q12.
print ("12.")
a = b = [1, 2, 3]
a[1] = 4
print(b)

