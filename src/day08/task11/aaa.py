list = [{'aa' : '인천광역시 미추홀구 학익동', 'bb' : '1'}, {'aa' : '인천광역시 연수구 연수동', 'bb' : '2'}, {'aa' : '인천광역시 미추홀구 숭의동', 'bb' : '3'}]

q = []
for a in list :
    q.append(a['aa'].split(' ')[1])
print(q)        # [['인천광역시', '미추홀구', '학익동'], ['인천광역시', '연수구', '연수동'], ['인천광역시', '미추홀구', '숭의동']]


# 중복값 제거 해 리스트에 담기
result = []
for value in q:
    if value not in result:
        result.append(value)
print(result)


# for a in result :
#     print('aaaa')
#     print(a)
#
#     print(q[1].count(a))
#
