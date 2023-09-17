import pprint

word_a = "Fish"
word_b = "Hist"
word_c = "Vista"

# matrix = [[0 for i in range(n)] for i in range(n)]


def compute(word_a, word_b):
    Matrix = [[0 for j in range(len(word_b))] for i in range(len(word_a))]
    for i in range(len(word_a)):
        for j in range(len(word_b)):
            if word_a[i] == word_b[j]:
                # The letters match.
                Matrix[i][j] = Matrix[i - 1][j - 1] + 1
            else:
                # The letters don't match.
                Matrix[i][j] = max(Matrix[i - 1][j], Matrix[i][j - 1])
    return Matrix


M1 = compute(word_a, word_b)
M2 = compute(word_b, word_c)
# pprint.pprint(cell)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(M1)
pp.pprint(M2)

# Test
word1 = "blue"
word2 = "clues"
pp.pprint(compute(word1, word2))
