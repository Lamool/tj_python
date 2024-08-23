class Region :
    # 생성자
    def __init__ (self, name, total, male, female, house) :
        # self : 자기 자신(객체)  # self.필드명 (객체변수)
        self.name = name
        self.total = total
        self.male = male
        self.female = female
        self.house = house
        self.malePercent = round((male/total) * 100, 1)       # 남성비율
        self.femalePercent = round((female/total) * 100, 1)   # 여성비율