# 파이썬 배포본에 함께 들어 있는 함수들 = 라이브러리
# import를 하지 않아도 된다

# 1. abs(숫자) : 절대값 함수
print(f'{ abs(3) }, { abs(-3) }, { abs(-1.2)}')

# 2. all(리스트/튜플/문자열/딕셔너리/집합) : 모두 참이면 참 반환하는 함수
    # 데이터들의 참과 거짓, day02 4.불.py 참고
print(f'{ all([1,2,3]) }, { all([1,2,3,0]) }, { all([]) }')

# 3. any(리스트/튜플/문자열/딕셔너리/집합) : 하나라도 참이면 참 반환하는 함수
print(f'{ any([1,2,3]) }, { any([1,2,3,0]) }, { any([]) }')

# 4. chr(유니코드) : 유니코드 숫자를 문자로 반환하는 함수
print(f'{ chr(97) }, { chr(44032) }')   # a, 가

# 5. dir(객체) : 해당 객체가 가지는 변수나 함수를 보여주는 함수
print(f'{ dir([]) }, { dir( {} ) }')

# 6. divmod(a, b) : a를 b로 나눈 몫과 나머지를 튜플로 반환
print(divmod(7, 3))    # 몫 : 2, 나머지 : 1,   (2, 1)

# 7. enumerate(리스트/튜플/문자열) : 인덱스 값을 포함한 객체를 반환한다.
for i, name in enumerate(['body', 'foo', 'bar']) :
    print(i, name)

# 8. eval(문자열로 구성된 코드) :
print(eval('1+2'))              # 3
print(eval("'hi'+'a'"))         # hia
print(eval('divmod(4, 3)'))     # (1, 1)

# 9. filter(함수, 데이터) : 함수 내 조건이 충족하면 데이터를 반환 함수, list 타입으로 변환 가능
def positive(x) :
    return x > 0
data = [1, -3, 2, 0, -5, 6]
result = filter(positive, data)
print(list(result))     # list() : 리스트 타입으로 반환 해주는 함수
# 람다식 함수, 함수명 = lambda 매개변수1, 매개변수2 : 실행문
    # 주로 간단한 함수를 간결하게 사용한다.
add = lambda a, b : a + b   # return 명령어가 없어도 결과값이 리턴된다.
print(add(3, 4))    # 7
# filter와 람다식 활용
result = filter(lambda x : x > 0, data)      # js : data.filter(x => x > 0)
print(list(result))     # [1, 2, 6]

# 10. map(함수, 데이터) : 함수 내 실행문 데이터를 반환 함수, list 타입으로 변환 가능
result = map(lambda x : x * 2, data)
print(list(result))     # [2, -6, 4, 0, -10, 12]

# 11. hex : 정수를 입력받아 16진수 문자열로 변환하여 리턴하는 함수
print(hex(234))     # 0xea  -> type : <class 'str'>
print(hex(3))       # 0x3   -> type : <class 'str'>

# 12. id : id(object) - 객체를 입력받아 객체의 고유 주솟값을 리턴하는 함수
a = 3
print(id(3))    # 140734403512824
print(id(a))    # 140734403512824
b = a
print(id(b))    # 140734403512824
print(id(4))    # 140734403512856

# 13. input : input([prompt)는 사용자 입력을 받는 함수
a = input()     # 사용자가 입력한 정보를 변수 a에 저장
print(a)        # 사용자 입력으로 받은 값 출력
b = input("Enter : ")   # 입력 인수로 "Enter : " 문자열 전달
# Enter : 프롬프트를 띄우고 사용자 입력을 받음
print(b)        # 사용자 입력으로 받은 값 출력

# 14. int : 문자열 형태의 숫자나 소수점이 있는 숫자를 정수로 리턴하는 함수
print(int('3'))     # 3
print(int(3.4))     # 3

    # - int(x, radix) : radix 진수로 표현된 문자열 x를 10진수로 변환하여 리턴하는 함수
print(int('11', 2))     # 3
print(int('1A', 16))    # 26

# 15. isinstance : isinstance(object, class) - 객체가 그 클래스의 인스턴스라면 참 반환
class Person :      # 아무런 기능이 없는 Person 클래스 생성
    pass
a = Person()        # Person 클래스의 인스턴스 a 생성
print(isinstance(a, Person))    # True  # a가 Person 클래스의 인스턴스인지 확인

b = 3
print(isinstance(b, Person))    # False  # a가 Person 클래스의 인스턴스인지 확인

# 16. len : 입력값의 길이(요소의 전체 개수)를 리턴하는 함수
print(len("python"))    # 6
print(len([1, 2, 3]))   # 3
print(len((1, a)))      # 2

# 17. list : 반복 가능한 데이터를 입력받아 리스트로 만들어 리턴하는 함수
print(list("python"))       # ['p', 'y', 't', 'h', 'o', 'n']
print(list((1, 2, 3)))      # [1, 2, 3]

    # - list 함수에 리스트를 입력하면 똑같은 리스트를 복사하여 리턴한다
a = [1, 2, 3]
b = list(a)
print(b)        # [1, 2, 3]

# 18. max : 최댓값을 리턴하는 함수
print(max([1, 2, 3]))   # 3
print(max("python"))    # y     # 문자열의 경우, 유니코드 값이 가장 큰 문자를 리턴
#print(max(5))           # 에러 발생.

# 19. min : 최솟값을 리턴하는 함수
print(min([1, 2, 3]))   # 1
print(min("python"))    # h
#print(min(5))           # 에러 발생. TypeError: 'int' object is not iterable

# 20. oct : 정수를 8진수 문자열로 바꾸어 리턴하는 함수
print(oct(34))      # 0o42
print(oct(12345))   # 0o30071

# 21. open : '파일 이름'과 '읽기 방법(w, r, a, b)'을 입력받아 파일 객체를 리턴하는 함수. 읽기 방법 생략 시 기본값 r로
#f = open("binary_file", "rb")

# 22. ord : 문자의 유니코드 숫자 값을 리턴하는 함수  (chr 함수와 반대로 동작)
print(ord('a'))     # 97
print(ord('가'))    # 44032

# 23. pow : pow(x, y) - x를 y제곱한 결괏값을 리턴하는 함수
print(pow(2, 4))    # 16
print(pow(3, 3))    # 27

# 24. range : range([start,] stop [,step]) - 입력받은 숫자에 해당하는 범위 값을 반복 가능한 객체로 만들어 리턴하는 함수
    # - 인수가 하나일 경우 : 시작 숫자를 지정해 주지 않으면 range 함수는 0부터 시작한다.
print(list(range(5)))       # [0, 1, 2, 3, 4]

    # - 인수가 2개일 경우 : 시작 숫자와 끝 숫자를 나타낸다. 단, 끝 숫자는 해당 범위에 포함되지 않는다
print(list(range(5, 10)))   # [5, 6, 7, 8, 9]   # 끝 숫자 10은 포함되지 않음.

    # - 인수가 3개일 경우 : 세 번째 인수는 숫자 사이의 거리를 뜻한다
print(list(range(1, 10, 2)))        # [1, 3, 5, 7, 9]   -> 1부터 9까지. 숫자 사이의 거리는 2
print(list(range(0, -10, -1)))      # [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]   -> 0부터 -9까지. 숫자 사이의 거리는 -1
print(list(range(0, -10)))          # []   -> 위의 코드에서 -1을 빼봤는데 [] 출력됨

# 25. round : round(number, [,ndigits]) - 숫자를 입력받아 반올림해 리턴하는 함수 (ndigits : 반올림하여 표시하고 싶은 소수점의 자릿수를 의미)
print(round(4.6))         # 5
print(round(4.2))         # 4

print(round(5.678, 2))    # 5.68   -> 소수점 2자리까지만 반올림하여 표시

# 26. sorted : 정렬 후 결과를 리스트로 리턴하는 함수
print(sorted([3, 1, 2]))            # [1, 2, 3]
print(sorted(['a', 'c', 'b']))      # ['a', 'b', 'c']
print(sorted("zero"))               # ['e', 'o', 'r', 'z']
print(sorted((3, 2, 1)))            # [1, 2, 3]

# 27. str
print(str(3))       # 3
print(str('hi'))    # hi

# 28. sum
print(sum([1, 2, 3]))       # 6
print(sum((4, 5, 6)))       # 15

# 29. tuple
print(tuple("abc"))         # ('a', 'b', 'c')
print(tuple([1, 2, 3]))     # (1, 2, 3)

# 30. type
print(type("abc"))                  # <class 'str'>
print(type([]))                     # <class 'list'>
print(type(open("test", 'w')))      # <class '_io.TextIOWrapper'>

# 31. zip
print(list(zip([1, 2, 3], [4, 5, 6])))                  # [(1, 4), (2, 5), (3, 6)]
print(list(zip([1, 2, 3], [4, 5, 6], [7, 8, 9])))       # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
print(list(zip("abc", "def")))                          # [('a', 'd'), ('b', 'e'), ('c', 'f')]

