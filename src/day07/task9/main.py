'''
    csv 파일 다루기
    파일 : 인천광역시_부평구_인구현황.csv
    [조건1] 부평구의 동마다 Region 객체 생성해서 리스트 담기
    [조건2] Region 객체 변수 :
        1.동이름 2.총인구수 3.남인구수 4.여인구수 5.세대수
        Region 함수 :
            남자 비율 계산 함수
            여자 비율 계산 함수
    [조건3] 모든 객체의 정보를 f포매팅해서 console 창에 출력하시오.
    [조건4] 출력시 동마다 남 여 비율 계산해서 백분율로 출력하시오.
    출력예시
        부평1동,  35141,  16835,  18306,  16861, 59%  41%

'''

from region import Region, malePercent, femalePercent



if __name__ == "__main__" :
    print('--start--')

    data = []

    # 1.읽기 모드 파일 객체
    f = open("population.txt", 'r', encoding='utf-8')

    # 2. 읽어오기
    population = []       # 읽어온 내용들을 저장할 리스트
    population = f.read()
    print(population)

    rows = population.split("\n")     # 행 구분
    print(rows)

    for row in rows :
        if row :    # 만약에 데이터가 존재한다면
            cols = row.split(',')
            print(cols)
            print(cols[0])
            region = Region(cols[0], cols[1], cols[2], cols[3], cols[4])
            data.append(region)
    f.close()
    count = 0
    for a in data :
        # print('for')
        try :
            # print('if')
        # print(a.totalNum)
        # print(isinstance(a.totalNum, int))
        # if isinstance(a.totalNum, int) :
            mPercent = malePercent(int(a.totalNum), int(a.maleNum))
            fPercent = femalePercent(int(a.totalNum), int(a.femaleNum))
            print(f'{a.dongName}, {a.totalNum}, {a.maleNum}, {a.femaleNum}, {a.householdsNum}, {mPercent}, {fPercent}')
        count += 1


'''
        # print('for')
        if count > 0 :
            # print('if')
        # print(a.totalNum)
        # print(isinstance(a.totalNum, int))
        # if isinstance(a.totalNum, int) :
            mPercent = malePercent(int(a.totalNum), int(a.maleNum))
            fPercent = femalePercent(int(a.totalNum), int(a.femaleNum))
            print(f'{a.dongName}, {a.totalNum}, {a.maleNum}, {a.femaleNum}, {a.householdsNum}, {mPercent}, {fPercent}')
        count += 1
'''




