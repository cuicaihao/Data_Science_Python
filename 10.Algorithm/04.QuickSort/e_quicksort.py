def quicksort(array):
    if type(array) is not list:
        print("input data format is wrong")
        return None
    elif len(array) < 2:
        # base case, arrays with 0 or 1 element are already "sorted"
        return array
    else:
        # recursive case
        pivot = array[0]
        # sub-array of all the elements less than the pivot
        less = [i for i in array[1:] if i <= pivot]
        # sub-array of all the elements greater than the pivot
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)


print(quicksort([10, 5, 2, 3]))


def test_quicksort():
    input = [10, 2, 3, 4, 6]
    target = [2, 3, 4, 6, 10]
    output = quicksort(input)
    assert output == target


def test_quicksort_empty():
    input = []
    target = []
    output = quicksort(input)
    assert output == target


def test_quicksort_none():
    input = None
    target = None
    output = quicksort(input)
    assert output == target
