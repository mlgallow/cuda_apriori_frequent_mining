import random
f = open('topic-3.txt', 'w')
#a = [x for x in range(1, 256)]
a = []
for x in range(1,64):
    #a.append(random.randint(1,32))
    a.append(x)
i = 0
s = ''
for item in a:
    s += str(item);
    s += ' '

s+= '\n' 
while i < 1000000:
    f.write(s)
    i +=1

f.close()
