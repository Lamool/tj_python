from user import User

# 1. load
def dataLoad() :
    try : # 예외 처리 # 예외가 발생할 것 같은 코드
        # #1.읽기 모드 파일 객체
        f = open("user.txt", 'r', encoding='utf-8')

        # 2. 읽어오기
        names = []      # 읽어온 내용들의 객체들을 저장하는 리스트
        data = f.read()
        rows = data.split("\n")        # 행 구분
        for row in rows :
            if row : # 만약에 데이터가 존재하면
                cols = row.split(',')
                user = User(cols[0], cols[1])
                names.append(user)
        f.close()
        return names
    except FileNotFoundError:  # 예외 처리 # 예외가 발생 했을때 실행되는 구역
        return []  # 빈 리스트  반환

# 2. save
def dataSave(names):
    # 1. 쓰기 모드 파일 객체
    f = open("user.txt", 'w', encoding="utf-8")\

    # 2. 내용 구성
        # 2. 객체를 문자열 변환
    outstr = ""
    for user in names :
        outstr += 'f{user.name},{user.age}\n'   # CSV 형식

    f.write(outstr)  # 3. 출력
    f.close()                       # 파일 닫기  # 파일 객체.close()
    return
