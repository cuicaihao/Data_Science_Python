#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2020/11/28 02:18:34
@author      :Caihao (Chris) Cui
@file        :longest-palindromic-substring.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib

# 给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

# 示例 1：

# 	输入: "babad"
# 	输出: "bab"
# 	注意: "aba" 也是一个有效答案。

# 示例 2：

# 	输入: "cbbd"
# 	输出: "bb"


def find_string(s):
    """
    给定一个字符串 s，找到 s 中最长的回文子串

    >>> find_string("babad")
    'bab'
    >>> find_string("abcdcba12")
    'abcdcba'
    """
    n = len(s)
    matrix = [[0 for i in range(n)] for i in range(n)]
    res = ""
    current_longest = 0

    for j in range(n):
        for i in range(j+1):
            if i >= j-1:
                if s[i] == s[j]:  # "字符相等"
                    matrix[i][j] = 1
                    # 将s[i:j]的长度与当前的回文子串的最长长度相比
                    temp_longest = j - i + 1
                    if current_longest < temp_longest:
                        res = s[i:j+1]  # 取当前的最长回文子串
                        current_longest = temp_longest   # 当前最长回文子串的长度
            else:
                if s[i] == s[j] and matrix[i+1][j-1]:
                    matrix[i][j] = 1
                    if current_longest < j - i + 1:
                        res = s[i:j+1]
                        current_longest = j - i + 1
    return res


# s = "babad"
# res = find_string(s)
# print("{} 最长的回文子串:{}".format(s, res))
# # print("{}  ".format(matrx1))


# s = "abcdcba12"
# res = find_string(s)
# print("{} 最长的回文子串:{}".format(s, res))
# # print("{}  ".format(matrx2))

# TODO python -m doctest longest-palindromic-substring.py -n
if __name__ == "__main__":
    print("Start printing ...")
    import doctest
    doctest.testmod()
