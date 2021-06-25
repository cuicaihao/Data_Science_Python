"""猜数字游戏"""


def guess(target: int):
    user_guess = int(input("请输入你猜的数 >>> "))
    breakpoint()  # add this line
    if user_guess == target:
        return "你猜对了!"
    else:
        return "猜错了"


if __name__ == '__main__':
    a = 100
    print(guess(a))
