import numpy as np
C = "NVRTM YJOBI ZVXIK KEEGE ZVDLM ZYAKM MVNTV KTIEL KIWWI XVBAS IBLTRMKHRE TSEDF ZRICI JWRDQ ZYEXR JVXDJ IFICG OUECG K"
C = C.replace(" ", "")
alphabet = "A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z"
letters = alphabet.split(", ")
# Stuff = [[]]
# for x in range(7):
#     for i in range(17):
#         Stuff[x][i] = C[(i*7+x)]
#         print(str(Stuff) + " " + str(x) + " " + str(i))

sum = 0
for letter in letters:
    count = C.count(letter)
    print(letter + "  " + str(count))
    sum += count*(count-1)
print(sum)
print(sum/(len(C)*(len(C)-1)))
print(1/5*.066+4/5*.038)
print(len(C))
# print(C.find("EZ", 19))
# str = ""
# for i in range(len(C)):
#     ind = letters.index(C[i])
#     if i % 4 == 1:
#         str += letters[(ind+6) % 26]
#     elif i % 4 == 3:
#         str += letters[(ind-8) % 26]
#     else:
#         str += C[i]
# print(C)
# # str = "bmzsymzltniljynytzbmymzytzbmljoltni".upper()
# for i in range(26):
#     vig = ""
#     for j in range(len(str)):
#         ind = letters.index(str[j])
#         vig += letters[(ind + i) % 26]
#     print(vig)
#     print(letters[i] + letters[(i+20) % 26] + letters[(i+8) % 26])
