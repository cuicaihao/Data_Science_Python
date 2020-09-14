import time
import threading


def main():
    for i in range(10):
        th = threading.Thread(target=lambda: time.sleep(1))
        th.name = "CustomThreadName-%s" % i
        th.start()
        print(th.name)
    time.sleep(5)


if __name__ == "__main__":
    main()

# sudo py-spy record -o thread_names.svg -- python thread_names.py
