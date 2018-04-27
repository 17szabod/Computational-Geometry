C = "011110110010100010000000111110101001010100100100101000000000110000111010101010010111101110000000110100101010101101000000010110101001101100111100011100000000101010111001100010111010000000011110110010100010000000111000100000101011101111000000100110101010110110"
P = "The quick brown fox jumps over the lazy dog"
tbl = {}
for i in range(len(P)):
    tbl[P[i]] = C[i*6:i*6+6]
tbl['Cap'] = '000001'
print(tbl)


def answer(plaintext):
    for i in range(len(plaintext)):
        let = plaintext[i]
        if let.isupper():
            print(tbl['Cap'], end="")
        print(tbl[let], end="")


answer(P)
print("")
print('000001' + C)
