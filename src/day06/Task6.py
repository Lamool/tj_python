# day06 > Task3.py
# 객체/리스트 활용 , p.190 ~ p.207
# [조건1] : 각 함수 들을 구현 해서 프로그램 완성하기
# [조건2] :  1. 한 명의 name, age를 입력받아 저장합니다.
#           2. 저장된 객체들을 name, age를 모두 출력 합니다.
#           3. 수정할 이름을 입력받아 존재하면 새로운 name, age를 입력받고 수정 합니다.
#           4. 삭제할 이름을 입력받아 존재하면 삭제 합니다.
# [조건3] : names 변수 외 추가적인 전역 변수 생성 불가능합니다.
# 제출 : git에 commit 후 카톡방에 해당 과제가 있는 링크 제출

names = [ ] # 샘플 데이터

class nameAge :
    def __init__ (self, name, age) :
        self.name1 = name
        self.age1 = age

# 1. 한 명의 name, age를 입력받아 저장하는 함수
def nameCreate( ) :
    global names
    inputName = input('이름을 입력해주세요 : ')
    inputAge = int(input('나이를 입력해주세요 : '))
    var1 = nameAge(inputName, inputAge)
    varF = f'{var1.name1},{var1.age1}'
    names.append(varF)
    print('저장되었습니다')
    return

# 2. 저장된 객체들 name, age를 모두 출력하는 함수
def nameRead( ) :
    global names
    print(names)
    return

# 3. 수정할 이름을 입력 받아 존재하면 새로운 name, age를 입력받고 수정하는 함수
def nameUpdate(  ) :
    global names
    inputName = input('수정하고자 하는 이름을 입력해주세요 : ')
    tf = 0;     # 요소를 찾았는지 판단할 변수
    save = ""      # 찾은 요소의 값을 저장받을 변수

    for a in names :    # 수정하고자 하는 이름이 존재하는지 판단하기 위한 반복문
        # print(a)
        if a.count(inputName) == 1 :    # 만약 수정하고자 하는 이름이 존재한다면
            save = a                       # 요소 전체를 저장
            tf = 1
            break

    if tf == 1 :        # 수정하고자 하는 이름이 존재한다면
        # print(names.index(f'{inputName},'))
        newName = input('새로운 이름을 입력해주세요 : ')
        newAge = int(input('새로운 나이를 입력해주세요 : '))
        var1 = nameAge(newName, newAge)         # 객체 생성
        varF = f'{var1.name1},{var1.age1}'
        index = names.index(save)               # 수정하고자 하는 요소의 인덱스 찾기
        # print(save)
        names.insert(index, varF)               # 수정하고자 하는 요소의 인덱스에 값 추가
        names.remove(save)                      # 수정하고자 하는 요소 삭제
        print('수정이 완료되었습니다')
    else :
        print('수정하고자 하는 이름이 존재하지 않습니다')

    return

# 4. 삭제할 이름을 입력 받아 존재하면 삭제하는 함수
def nameDelete( ) :
    global names

    inputName = input('삭제하고자 하는 이름을 입력해주세요 : ')
    tf = 0;  # 요소를 찾았는지 판단할 변수
    save = ""  # 찾은 요소의 값을 저장받을 변수

    for a in names:  # 삭제하고자 하는 이름이 존재하는지 판단하기 위한 반복문
        # print(a)
        if a.count(inputName) == 1:  # 만약 삭제하고자 하는 이름이 존재한다면
            save = a  # 요소 전체를 저장
            tf = 1
            break

    if tf == 1:  # 삭제하고자 하는 이름이 존재한다면
        # print(names.index(f'{inputName},'))
        index = names.index(save)  # 삭제하고자 하는 요소의 인덱스 찾기
        # print(save)
        names.remove(save)  # 삭제하고자 하는 요소 삭제
        print('삭제가 완료되었습니다')
    else:
        print('삭제하고자 하는 이름이 존재하지 않습니다')

    return

while True :
    ch = int( input('1.create 2.read 3.update 4.delete : ') )
    if ch == 1 : nameCreate( )
    elif ch == 2 : nameRead( )
    elif ch == 3 : nameUpdate( )
    elif ch == 4 : nameDelete( )


