def recurse(x):
    if x == 0:
        return
    recurse(x-1)


while True:
    recurse(20)

# sudo py-spy record -o recursive.svg -- python recursive.py
