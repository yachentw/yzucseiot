from random import randint

symbols = ['*','#','@','$']
fd = open("symbols.txt", "w")
for i in range(20):
    for j in range(randint(1,30)):
        fd.write(symbols[randint(0,len(symbols)-1)])
    fd.write("\n")
