def getDict():
    a_label = []
    b_label = []
    c_label = []

    with open('train_a.txt') as f:
        isFirst = True

        while True:
            line = f.readline()

            if isFirst:
                isFirst = False
                continue            

            if not line:
                break

            l = line.split('\t')

            a_label.append(l[5])

            b_label.append(l[6])

            c_label.append(l[7].replace('\n',''))


    a_label = list(set(a_label))
    b_label = list(set(b_label))
    c_label = list(set(c_label))

    a_label.sort()
    b_label.sort()
    c_label.sort()
    
    a_dict = {}
    b_dict = {}
    c_dict = {}
    count = 1

    for i in a_label:
        a_dict[i] = str(count)
        count+=1

    for i in b_label:
        b_dict[i] = str(count)
        count+=1

    for i in c_label:
        c_dict[i] = str(count)
        count+=1


    print(a_dict)
    print(b_dict)
    print(c_dict)
    return (a_dict,b_dict,c_dict)

(a_dict,b_dict,c_dict) = getDict()
	
def findKeyByValue(dic,value):
    for k,v in dic.items():
        if value == v:
            return k
    print(len(dic))
    print(value)
    return None

idFile = open('id.txt')
firstFile = open('First.txt')
secondFile = open('Second.txt')
thirdFile = open('Third.txt')

target = open('result.txt','w')
target.write('item_id	cate1_id	cate2_id	cate3_id\n')
while True:
    idLine = idFile.readline()
    if not idLine:
        break
    firstLine = firstFile.readline()
    secondLine = secondFile.readline()
    thirdLine = thirdFile.readline()
    
    buffer = idLine.split('\t')[0].replace('\n','') + '\t' + findKeyByValue(a_dict, firstLine[:-1]) + '\t' + findKeyByValue(b_dict, str(int(secondLine[:-1])+10)) + '\t' + findKeyByValue(c_dict, str(int(thirdLine[:-1])+64+10)) + '\n'
    target.write(buffer)
    
idFile.close()
firstFile.close()
secondFile.close()
thirdFile.close()
target.close()