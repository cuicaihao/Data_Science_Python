#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Created on   :2020/11/27 17:25:25
@author      :Caihao (Chris) Cui
@file        :Untitled-1
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
"""

# here put the import lib


def binarySearch(arr, l, r, x):
    """Binary Search
    Parameters
    arr: input array / list sorted in ascend order
    l: left pointer
    r: right pointer
    x: value
    Example:
    >>> binarySearch([ 2, 3, 4, 10, 40 ], 0, 4, 10)
    3
    >>> binarySearch([ 1, 2, 3, 100, 4 ], 0, 4, 100)
    4
    Test: python -m doctest -v BinarySearch.py
    """
    arr.sort()
    while l <= r:
        mid = l + (r - l) // 2
        if x == arr[mid]:
            return mid
        elif x < arr[mid]:
            r = mid - 1
        else:
            l = mid + 1
    return None


# result = binarySearch([2, 3, 4, 10, 40], 0, 4, 10)
# print(result)

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
