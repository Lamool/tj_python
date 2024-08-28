# [1] 플라스크 객체 가져오기
from appstart import app
# [2] (우리가 만든) 서비스 모듈 가져오기
import service

# [3] HTTP 매핑 주소 정의
@app.route('/job', methods=['get'])     # http://localhost:5000/job
def getjob() :
    # (1) 만약에 크롤링 된 CSV 파일이 없거나 최신화 하고 싶을 때
    result = []
    service.jobkoreaInfo(result);         print(result); # 크롤링해서 CSV 파일 생성
    service.list2d_to_csv(result, '잡코리아일자리정보', ['회사명', '공고명', '경력', '학력', '계약유형', '지역', '채용기간'])  # 잡코리아 정보를 csv로 저장 서비스 호출

    # (2) CSV에 저장된 JSON으로 가져오기
    result2 = service.read_csv_to_json('잡코리아일자리정보')  # csv 파일을 json으로 가져오는 서비스 호출

    # (3)
    service.announcement_number(result)

    # (4) 서비스로부터 받은 데이터로 HTTP 응답하기
    return result2

