# 문자열 활용, p.58 ~ p.76
# [조건1] : 각 함수들을 구현해서 프로그램 완성
# [조건2] : 1. 하나의 이름을 입력 받아 여러 명의 이름을 저장
#           2. 저장된 여러 명의 이름을 모두 출력
#           3. 수정할 이름 존재하면 새로운 이름을 입력 받아 수정
#           4. 삭제할 이름을 입력 받아 존재하면 삭제
# [조건3] names 변수 외 추가적인 전역 변수 생성 불가능

# 하나의 변수에 여러 가지 정보 # 1. JSON(몇 가지 필드를 KEY로 구분) 2. CSV(몇 가지 필드를 쉼표(,)로 구분), 주로 문자열 타입 사용

names = ""  # 여러 개 name들을 저장하는 문자열 변수

# 1. 이름을 입력 받아 여러 명의 이름을 저장하는 함수
def nameCreate() :
    name = input("이름을 입력해주세요 : ")
    global names
    if names == "" :                            # 공백이면 바로 이름 저장
        names = names + "'" + name + "'"
        print("이름 저장을 완료했습니다.")
        return
    names = names + ", " + "'" + name + "'"     # 공백이 아니면 저장된 이름 사이에 쉼표를 추가하여 이름 저장
    print("이름 저장을 완료했습니다.")
    return

# 2. 저장된 전체 이름을 모두 출력하는 함수
def nameRead() :
    print("저장된 전체 이름")
    print(names)

# 3. 수정할 이름 존재하면 새로운 이름을 입력 받아 수정하는 함수
def nameUpdate() :
    global names
    name = input("수정하고자 하는 이름을 입력해주세요 : ")
    if names.find("'" + name + "'") == -1 :
        print("수정하고자 하는 이름이 존재하지 않습니다.")
        return
    else :
        newName = input("새로운 이름을 입력해주세요 : ")
        names = names.replace("'" + name + "'", "'" + newName + "'")
    return

# 4. 삭제할 이름을 입력 받아 존재하면 삭제하는 함수
def nameDelete() :
    global names
    name = input("삭제하고자 하는 이름을 입력해주세요 : ")
    if names.find("'" + name + "'") == -1:
        print("삭제하고자 하는 이름이 존재하지 않습니다.")
        return
    else:
        index = names.index("'" + name + "'")       # 삭제할 이름의 인덱스 찾기
        # print(index)
        # print(names[index])
        length = len(name)

        # 문자열 잘라서 저장하기
        if index == 0 :
            names = names[index + length + 4:]
        else :
            a = names[0:index - 2]
            # print(a)
            # print(names)
            a = a + names[index + length + 2:]
            # print(a)
            names = a
    return


while True :    # 무한루프
    ch = int(input("1.create 2.read 3.update 4.delete : "))     # input으로 입력받은 값 문자열이기에 int형으로 변환
    if ch == 1 :
        nameCreate()
    elif ch == 2 :
        nameRead()
    elif ch == 3 :
        nameUpdate()
    elif ch == 4 :
        nameDelete()


'''
    for (nume in namessplit(',') :  # 문자열내, 쉼표 기준으로 분해
        print(name
    return
'''