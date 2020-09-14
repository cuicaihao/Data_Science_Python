import threading
import time

event = threading.Event()


def lighter():
    count = 0
    event.clear()  # 初始者为绿灯
    while True:
        if 5 < count <= 10:
            event.clear()  # 红灯，清除标志位
            print("red light is on... ")
        elif count > 10:
            event.set()  # 绿灯，设置标志位
            count = 0
        else:
            print("green light is on... ")

        time.sleep(1)
        count += 1


def car(name):
    while True:
        if event.is_set():  # 判断是否设置了标志位
            print('[%s] running.....' % name)
            time.sleep(1)
        else:
            print('[%s] sees red light,waiting...' % name)
            event.wait()
            print('[%s] green light is on,start going...' % name)


startTime = time.time()
light = threading.Thread(target=lighter,)
light.start()

car = threading.Thread(target=car, args=('Car',))
car.start()
endTime = time.time()
print('Time Cost：', endTime-startTime)

# sudo py-spy record -o python.thread.svg -- python python.thread.py
