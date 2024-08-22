'''
    user.py : user
'''

from user import User
from file import dataLoad, dataSave

names = [ ]

def nameCreate( ) :
    name = input('이름을 입력해주세요 : ')
    age = input('나이를 입력해주세요 : ')
    user = User(name, age)
    names.append(user)
    dataSave(names)      # 파일처리
    return
def nameRead( ) :
    for user in names :          # 리스트내 딕셔너리 하나씩 호출
        print( f'name : { user.name }, age : { user.age }' )
    return

if __name__ == "__main__" :
    while True :
        ch = int( input('1.create 2.read : ') )
        if ch == 1 : nameCreate( )
        elif ch == 2 : nameRead( )