# 클래스란? 객체를 실체화 하기 위한 설계도
# 인스턴스란? 클래스 기반으로 객체가 실체화된 메모리
# 객체? 추상적 개념, 물리적 개념으로 고유성질(변수)과 행위(함수)를 정의한 것

# [1] 클래스 구조 만들기

# (1) 간단한 클래스, class 클래스명 :
class FourCal :
    pass    # 아무것도 수행하지 않는 문법으로, 임시로 코드를 작성할 때 주로 사용됨

# (2) 객체 생성, 클래스명()
a = FourCal()
print(type(a))  # <class '__main__.FourCal'>    # type() : 타입 반환해주는 함수

# [2] 클래스 내 메소드(함수) 만들기
# 함수 코드는 객체들이 공유해서 사용((하기 때문에 self로 구분해준다(?)))
class FourCal :     # (1) 클래스 내 메소드 선언
    # 생성자, 다중생성자 없다
    def __init__ (self, first, second) :
        self.first = first
        self.second = second

    def setdata(self, first, second) :      # 함수/메소드 정의     # 매개변수란? 함수 호출 시 전달되는 인자 값을 저장하는 변수
        self.first = first                  # self(자신 객체)   # self.first : 객체변수가 생성된다
        self.second = second                # 함수를 호출한 객체(self) 내 second 변수를 선언하고 매개변수(2)를 저장한다.
    def add(self) :
        result = self.first + self.second   # self
        return result
    def mul(self) :
        result = self.first * self.second  # self
        return result
    def sub(self) :
        result = self.first - self.second  # self
        return result
    def div(self) :
        result = self.first / self.second  # self
        return result


# (2) 객체 생성
# a = FourCal()
#
# # (3) 객체 내 메소드 실행
# a.setdata(4, 2)
#
# # (4) 객체 내 필드 값 호출
# print(a.first)
# print(a.second)
#
# print(a.add()); print(a.mul()); print(a.sub()); print(a.div())
#
# b = FourCal()   # 객체 생성, a에 저장된 객체와 b에 저장된 객체는 다르다. 타입 같지만.
# b.setdata(3, 7)
# print(b.first)
# print(b.second)
#
# print(b.add()); print(b.mul()); print(b.sub()); print(b.div())

c = FourCal(4, 2)   # (4) 객체 생성, 생성자 매개변수
print(c.first)
print(c.second)


# [4] 상속
# (1) 하위 클래스 정의, class 클래스명(상위클래스명) :
class MoreFourCal(FourCal) :
    # 메서드 정의
    def pow(self):
        return self.first ** self.second
    # 오버라이딩 : 상위 클래스의 메서드 재정의
    def div(self) :     # 메서드 선언부를 동일하게 작성
        if self.second == 0 :
            return 0

# (2) 하위 클래스로 객체 생성
a = MoreFourCal(4, 2)
print(type(a))      # <class '__main__.MoreFourCal'>
print(a.add())      # 6 상위 클래스의 메소드 호출

print(a.pow())      # 7 본인 클래스의 메소드 호출



# [5] 클래스 변수
    # 객체 변수 : 객체 마다의 사용되는 변수    # 객체변수명.변수명 # 필드    # 멤머변수라고도 한다
    # 클래스 변수 : 모든 객체가 공유해서ㅓ 사용하는 변수 # 클래스명.변수명
        # 객체변수와 클래스변수의 이름이 같아도 식별이 가능하다
class Family :
    lastname = "김"      # 클래스 변수
print(Family.lastname)   # 김

a = Family()
print(a.lastname)        # 김

b = Family()
print(b.lastname)       # 박
Family.lastname = "박"
print(f'{a.lastname},{b.lastname}')     # 박, 박

