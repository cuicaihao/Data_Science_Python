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


def lengthOfLongestSubstring(s):
    n = len(s)
    ans = 0
    # mp stores the current index of a character
    mp = {}

    i = 0
    # try to extend the range [i, j]
    for j in range(n):
        if s[j] in mp:
            i = max(mp[s[j]], i)

        ans = max(ans, j - i + 1)
        print(mp)
        mp[s[j]] = j + 1

    print(s[i:j])
    return ans


s = "abcabcbb"
ans = lengthOfLongestSubstring(s)
# The answer is "abc", with the length of 3.
