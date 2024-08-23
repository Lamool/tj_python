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
# from py파일명 import 해당파일에호출할식별자((클래스, 함수, 변수 등?))
from region import Region

if __name__ == "__main__" :
    try :   # 예외처리
        regionList = []

        # 1. 파일 읽기 모드
        f = open("인천광역시_부평구_인구현황.csv", 'r')

        # 2. 파일 전체 읽어오기
        data = f.read();    print(data)

        # 3. 데이터 가공 (csv 형식), 행마다 분리
        rows = data.split('\n');    print(rows)

        # 4. 행마다 반복문, 첫 줄, 뒤에 2줄 제외
        rowCount = len(rows)    # 데이터의 전체 행 개수

        for row in rows[1:rowCount-2] :
            print(row)
            # 5. 열마다 분리
            cols = row.split(',');      print(cols)

            # 6. 해당 열들을 객체화, 타입 변환
            region = Region(cols[0], int(cols[1]), int(cols[2]), int(cols[3]), int(cols[4]));       print(region)

            # 7. 리스트 담기
            regionList.append(region);      print(regionList)

        # 8. 리스트 내 객체 정보 호출
        for region in regionList :
            print(region.toString())

        f.close()
    # 예외가 발생했을 때
    except FileNotFoundError as e : print(e)
    except Exception as e : print(e)




