from region import Region
def personData() :
    regionList = []

    # 1. 파일 읽기 모드
    f = open("인천광역시_부평구_인구현황.csv", 'r')

    # 2. 파일 전체 읽어오기
    data = f.read()

    # 3. 데이터 가공 (csv 형식), 행마다 분리
    rows = data.split('\n')

    # 4. 행마다 반복문, 첫 줄, 뒤에 2줄 제외
    rowCount = len(rows)  # 데이터의 전체 행 개수

    for row in rows[1:rowCount - 2]:
        # 5. 열마다 분리
        cols = row.split(',');

        # 6. 해당 열들을 객체화, 타입 변환
        region = Region(cols[0], int(cols[1]), int(cols[2]), int(cols[3]), int(cols[4]));

        # 7. 리스트 담기
        regionList.append(region);

    f.close()
    return regionList       # 리스트 반환