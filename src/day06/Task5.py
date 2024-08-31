# day06 > Task3.py
# 딕셔너리/리스트 활용 , p.93 ~ p.101
# [조건1] : 각 함수 들을 구현 해서 프로그램 완성하기
# [조건2] :  1. 한 명의 name, age를 입력받아 저장합니다.
#           2. 저장된 여러명의 name, age를 모두 출력 합니다.
#           3. 수정할 이름을 입력받아 존재하면 새로운 name, age를 입력받고 수정 합니다.
#           4. 삭제할 이름을 입력받아 존재하면 삭제 합니다.
# [조건3] : names 변수 외 추가적인 전역 변수 생성 불가능합니다.
# [조건4] : 프로그램이 종료되고 다시 실행되더라도 기존의 names 데이터가 유지되도록 파일처리 하시오.
#             - dataLoad
# 제출 : git에 commit 후 카톡방에 해당 과제가 있는 링크 제출
# 파일 내 데이터 설계 :다수의 이름과 나이의 데이터들을 저장하는 방법을 설계
    # CSV : 필드 구분, 객체 구분 \n

names = [ ]       # 샘플 데이터

def dataLoad() :        # 파일 내 데이터를 불러오기    # while True 위에서 실행, 프로그램 시작시 실행
    global names
    f = open("names.txt", 'r')      # 파일 읽기모드로 객체 반환    # 같은 패키지에 names.txt 파일을 직접 만들고 실행
    var = f.read()                  # 파일 내 데이터 전체 읽어오기  # 파일 객체.read()
    print(var)                      # 확인
    # 딕셔너리 만들고 리스트에 담기
    lines = var.split('\n')  # 줄마다(요소)
    print(len(var.split('\n')))
    for line in lines[:len(lines) - 1] :           # 읽어온 파일 내용을 \n 분해해서 한 줄씩 반복처리 # \n으로 객체 구분중    # 마지막 줄 제외
        print(line)
        dic = {'name' : line.split(',')[0], 'age' : line.split(',')[1]}     # 해당 줄에, (쉼표) 분해, [0] 이름 [1] 나이
        names.append(dic)
    f.close()
    return

def dataSave():         # 데이터를 파일 내 저장하기    # 사용처 nameCreate, NameUpdate, nameDelete
    global names
    f = open("names.txt", 'w', encoding="utf-8")      # 파일 쓰기모드로 객체 반환
    # 파이썬의 딕셔너리 -> 문자열 만들고 파일 쓰기
    outstr = ""     # 파일에 작성할 문자열 변수
    for dic in names :
        outstr += f'{dic['name']},{dic['age']}\n'      # 딕셔너리를 CSV 형식의 문자열로 변환   # ,(쉼표) 필드 구분   # \n (줄바꿈) 객체/딕셔너리 구분
    f.write(outstr)                 # 파일 객체 이용한 데이터 쓰기  # 파일 객체.write(데이터)
    f.close()                       # 파일 객체 닫기  # 파일 객체.close()
    return

def nameCreate( ) :

    global names
    newName = input('newName : ')
    newAge = {int(input('newAge : '))}
    dic = {'name': newName, 'age': newAge}  # 딕셔너리 구성
    names.append(dic)  # 딕셔너리를 리스트에 삽입
    dataSave()
    return

def nameRead( ) :
    global names
    for dic in names : # 리스트내 딕셔너리 하나씩 호출
        print( f'name : { dic['name'] }, age : {dic['age']}' )  # 딕셔너리변수명[key] 또는 딕셔너리변수명.get( key )
    return


def nameUpdate(  ) :
    global names
    oldName = input('oldName : ')
    for dic in names:
        if dic['name'] == oldName :
            newName = input('newName : ')
            dic['name'] = newName  # 해당 딕셔너리의 속성 값 수정 하기.
            newAge = input('newAge : ')
            dic['age'] = newAge  # 해당 딕셔너리의 속성 값 수정 하기.
            dataSave()
            return
    return

def nameDelete( ) :
    global names
    deleteName = input('deleteName : ')
    for dic in names:
        if dic['name'] == deleteName:  # 만약에 삭제할 이름과 같으면
            names.remove(dic)  # 리스트변수명.remove( 삭제할딕셔너리 )
            dataSave()
            return  # 1개만 삭제하기 위해서는 삭제후 return
    return

dataLoad()      # 프로그램이 실행될 때 파일 내용 읽어오기
while True :
    ch = int( input('1.create 2.read 3.update 4.delete : ') )
    if ch == 1 : nameCreate( )
    elif ch == 2 : nameRead( )
    elif ch == 3 : nameUpdate( )
    elif ch == 4 : nameDelete( )



