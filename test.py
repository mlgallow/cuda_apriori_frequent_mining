
f = open('topic-3.txt', 'w')
a = [1,2,3,4,5,6,7]
i = 0
s = ''
for item in a:
    s += str(item);
    s += ' '

s+= '\n' 
while i < 10000:
    f.write(s)
    i +=1

f.close()
