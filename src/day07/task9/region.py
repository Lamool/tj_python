class Region :
    # 생성자
    def __init__ (self, dongName, totalNum, maleNum, femaleNum, householdsNum) :
        self.dongName = dongName
        self.totalNum = totalNum
        self.maleNum = maleNum
        self.femaleNum = femaleNum
        self.householdsNum = householdsNum

    # 남자 비율 계산 함수
def malePercent(totalNum, maleNum) :
    percent = maleNum / totalNum * 100
    # print('남자 비율')
    # print(percent)
    return percent

    # 여자 비율 계산 함수
def femalePercent(totalNum, femaleNum) :
    percent = femaleNum / totalNum * 100
    # print('여자 비율')
    # print(percent)
    return percent