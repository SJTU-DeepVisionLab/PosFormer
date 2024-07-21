def make_label(file_dir,out_dir):
    f = open(file_dir, "r")
    n = open(out_dir, 'w')
    for line in f.readlines():           #依次读取每行  
        line = line.strip()              #去掉每行头尾空白  
        split_line=line.split()
        line_num=len(split_line)
        flag=0
        i=0
        while (i<line_num):
            if i==0:
                n.write(split_line[i])
                i=i+1
            else:
                if split_line[i]=='^':
                    flag=1
                    flagnum=1
                    n.write(' ' + str(flag)+' ' + str(flag))
                    i=i+1
                    while (flagnum>0):
                        i=i+1
                        n.write(' ' + str(flag))
                        if split_line[i]=='{':
                            flagnum=flagnum+1
                        elif split_line[i]=='}':
                            flagnum=flagnum-1
                    flag=0
                    i=i+1
                elif split_line[i]=='_':
                    flag=2
                    flagnum=1
                    n.write(' ' + str(flag)+' ' + str(flag))
                    i=i+1
                    while (flagnum>0):
                        i=i+1
                        n.write(' ' + str(flag))
                        if split_line[i]=='{':
                            flagnum=flagnum+1
                        elif split_line[i]=='}':
                            flagnum=flagnum-1
                    flag=0
                    i=i+1
                else:
                    n.write(" "+ str(flag)) 
                    i=i+1
                if i==line_num:
                    n.write('\n')
    # 关闭文件
    f.close()
    n.close()
# f1="data/2014/caption.txt"
# n1='data/2014/label.txt'
# f2="data/2016/caption.txt"
# n2='data/2016/label.txt'
# f3="data/2019/caption.txt"
# n3='data/2019/label.txt'
# make_label(f1,n1)
# make_label(f2,n2)
# make_label(f3,n3)

def tgt2updown(tgt):
    updowns=[]
    for indices in tgt:
        updowns.append(indices2updown(indices))
    return updowns

def indices2updown(indices):
    line_num=len(indices)
    updown=[0]*line_num
    flag=[0,1,2]
    i=0
    is_reverse=False
    if indices[0]==2:
        indices.reverse()
        is_reverse=True
    while (i<line_num):
        if indices[i]==82:                # ^
            end=find_end_bigbracket(indices,i+1,line_num)
            if end==0:
                print("没找到}",indices)            
            for i in range(i+2,end):
                if i >line_num:
                    print("越界",indices)
                updown[i]=flag[1]
            i=end+1
        elif indices[i]==83:              # _
            end=find_end_bigbracket(indices,i+1,line_num)
            if end==0:
                print("没找到}",indices)
            for i in range(i+2,end):
                if i >line_num:
                    print("越界",indices)
                updown[i]=flag[2]
            i=end+1
        else:
            i+=1
    if is_reverse:
        updown.reverse()

    return updown

def out2updown(tgt):
    updowns=[]
    for indices in tgt:
        updown_out=indices2updown(indices)
        updown_out.append(updown_out[0])
        updown_out.remove(updown_out[0])
        updowns.append(updown_out)
    return updowns

def find_end_midbracket(indices,start_i,end):   #寻找 ]
    count=1
    i=start_i+1
    while(count>0 and i<end):
        if indices[i]==42:
            count+=1
        elif indices[i]==81:
            count-=1
        i+=1
    if count==0:
        return i-1
    else:
        return 0
def find_end_bigbracket(indices,start_i,end):   #寻找 }  
    count=1
    i=start_i+1
    while(count>0 and i<end):
        if indices[i]==110:
            count+=1
        elif indices[i]==112:
            count-=1
        i+=1
    if count==0:
        return i-1
    else:
        return 0
def helper(indices,start,end,result):   #start:{   end:}
    flag=[0,1,2,3,4,5]
    special=True   #是否对无实体符号编码
    i=start+1
    while(i < end):
        if indices[i]==82:            # ^
            end1=find_end_bigbracket(indices,i+1,end)
            if special:
                result[i].append(flag[3])
                result[i+1].append(flag[3])
                result[end1].append(flag[3])
            for j in range(i+2, end1):
                result[j].append(flag[4]) #if indices[j] not in {42,81,82, 83, 110, 112} else None
            result=helper(indices,i+1,end1,result)
            i=end1+1
        elif indices[i]==83:          # _
            end1=find_end_bigbracket(indices,i+1,end)
            if special:
                result[i].append(flag[3])
                result[i+1].append(flag[3])
                result[end1].append(flag[3])
            for j in range(i+2, end1):
                result[j].append(flag[5]) #if indices[j] not in {42,81,82, 83, 110, 112} else None
            result=helper(indices,i+1,end1,result)
            i=end1+1
        elif indices[i]==53:          # \frac
            result[i].append(flag[3])
            end1=find_end_bigbracket(indices,i+1,end)
            for j in range(i+2, end1):
                result[j].append(flag[4]) #if indices[j] not in {42,81,82, 83, 110, 112} else None            
            end2=find_end_bigbracket(indices,end1+1,end)
            for j in range(end1+2, end2):
                result[j].append(flag[5]) #if indices[j] not in {42,81,82, 83, 110, 112} else None                      
            if special:
                result[i+1].append(flag[3])
                result[end1].append(flag[3])
                result[end1+1].append(flag[3])
                result[end2].append(flag[3])
            result=helper(indices,i+1,end1,result)
            result=helper(indices,end1+1,end2,result)
            i=end2+1
        elif indices[i]==74:          # \sqrt                  
            result[i].append(flag[3])
            if indices[i+1]==42:
                end1=find_end_midbracket(indices,i+1,end)
                for j in range(i+2, end1):
                    result[j].append(flag[4]) #if indices[j] not in {42,81,82, 83, 110, 112} else None
                end2=find_end_bigbracket(indices,end1+1,end)
                for j in range(end1+2, end2):
                    result[j].append(flag[5]) #if indices[j] not in {42,81,82, 83, 110, 112} else None                                      
                if special:
                    result[i+1].append(flag[3])
                    result[end1].append(flag[3])
                    result[end1+1].append(flag[3])
                    result[end2].append(flag[3])                
                result=helper(indices,i+1,end1,result)
                result=helper(indices,end1+1,end2,result)
                i=end2+1    
            else:
                end1=find_end_bigbracket(indices,i+1,end)
                for j in range(i+2, end1):
                    result[j].append(flag[5]) #if indices[j] not in {42,81,82, 83, 110, 112} else None                
                if special:
                    result[i+1].append(flag[3])
                    result[end1].append(flag[3])                
                result=helper(indices,i+1,end1,result)
                i=end1+1         
        elif indices[i]==0:
            result[i].append(flag[0])
            i+=1
        elif indices[i]==1:
            result[i].append(flag[1])
            i+=1
        elif indices[i]==2:
            result[i].append(flag[2])
            i+=1
        else:
            result[i].append(flag[3])
            i+=1            
    return result        
def indices2muti_label(indices):
    result = [[] for _ in range(len(indices))]
    is_reverse=False
    if indices[0]==2:
        indices.reverse()
        is_reverse=True    
    result=helper(indices,-1,len(indices),result)
    if is_reverse:
        result.reverse()
    return result
def tgt2muti_label(tgt):
    muti_label_batch=[]
    max_length=0
    for indices in tgt:
        label=indices2muti_label(indices)
        for i in range(len(label)):
            if len(label[i])<=5:
                label[i].extend([0] * (5 - len(label[i])))
            else:
                label[i]=label[i][:5]
        muti_label_batch.append(label)
    return muti_label_batch
def tgt2layernum_and_pos(tgt):
    layer_num=[]
    final_pos=[]
    for indices in tgt:
        layer_num_sub=[]
        final_pos_sub=[]
        label=indices2muti_label(indices)    # l,?
        for i in range(len(label)):
            if len(label[i])==1:
                layer_num_sub.append(0)
                final_pos_sub.append(label[i][0])
            elif len(label[i])<=5:
                layer_num_sub.append(len(label[i])-1)
                final_pos_sub.append(label[i][-2])
            else:
                layer_num_sub.append(4)
                final_pos_sub.append(label[i][3])
        layer_num.append(layer_num_sub)
        final_pos.append(final_pos_sub)
    return layer_num , final_pos
def out2muti_label(tgt):
    muti_label_batch=[]
    max_length=0
    for indices in tgt:
        label=indices2muti_label(indices)
        label.append(label[0])
        label.remove(label[0])
        for i in range(len(label)):
            if len(label[i])<=5:
                label[i].extend([0] * (5 - len(label[i])))
            else:
                label[i]=label[i][:5]
        muti_label_batch.append(label)
    return muti_label_batch
def out2layernum_and_pos(tgt):
    layer_num=[]
    final_pos=[]
    for indices in tgt:
        layer_num_sub=[]
        final_pos_sub=[]
        label=indices2muti_label(indices)    # l,?
        label.append(label[0])
        label.remove(label[0])
        # if label[0]==0: 
        #     label.append(1)
        #     label.remove(label[0])
        # elif label[0]==1:
        #     label.append(0)
        #     label.remove(label[0])
        for i in range(len(label)):
            if len(label[i])==1:
                layer_num_sub.append(0)
                final_pos_sub.append(label[i][0])
            elif len(label[i])<=5:
                layer_num_sub.append(len(label[i])-1)
                final_pos_sub.append(label[i][-2])
            else:
                layer_num_sub.append(4)
                final_pos_sub.append(label[i][3])
        layer_num.append(layer_num_sub)
        final_pos.append(final_pos_sub)
    return layer_num , final_pos

def pad_sublists(lst, n):
    for i in range(len(lst)):
        lst[i].extend([0] * (n - len(lst[i])))
    return lst
def get_longest_sublist_length(lst):
    longest_length = 0
    for sublist in lst:
        sublist_length = len(sublist)
        if sublist_length > longest_length:
            longest_length = sublist_length
    return longest_length





# testlist=[1,53,110,8,9,82,110,74,110,10,112,112,11,112,110,21,22,112,2,0,0]
# print(indices2updown(testlist))
testtgt=[[1,53,110,8,9,82,110,74,110,10,112,112,11,112,110,21,22,112,2,0,0]]
print(tgt2layernum_and_pos(testtgt))
testlist=[1,53,110,8,9,82,110,74,110,10,112,112,11,112,110,21,22,112,2,0,0]
print(indices2muti_label(testlist))

# testlist=[1,53,110,14,83,110,15,112,112,110,83,82,112]
# print(indices2muti_label(testlist))



# testlist=[[1,56,67,53,110,83,112,110,83,82,112],[1,56,67,74,42,83,83,81,110,82,112,85]]
# print(out2updown(testlist))





# testtgt=[[  1,  53, 110, 107,  83, 110,  12, 112, 112, 110,  53, 110, 107,  83,
#          110,  13, 112, 112, 110,  53, 110, 107,  83, 110,  14, 112, 112, 110,
#          107,  83, 110,  15, 112, 112, 112, 112,   0,   0,   0,   0,   0,   0,
#            0,   0],
#         [  1, 104,  83, 110,  97,   6,  12, 112,  22,  74, 110, 104,  83, 110,
#           97, 112, 105,  83, 110,  97, 112, 112,   0,   0,   0,   0,   0,   0,
#            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#            0,   0],
#         [  1,  99,  22,  74, 110,  84,  82, 110,  13, 112,   6,  85,  82, 110,
#           13, 112,   8,  13,  84,  85,  49,  24, 112,   0,   0,   0,   0,   0,
#            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#            0,   0],
#         [  1,  53, 110,  12, 112, 110, 101,  82, 110,  13, 112, 112,  22,  53,
#          110,  12, 112, 110,   4,  36,   8,  96,   5,  82, 110,  13, 112, 112,
#            6,  53, 110,  12, 112, 110,   4,  36,   6,  96,   5,  82, 110,  13,
#          112, 112],
#         [  2, 112, 112, 112, 112,  15, 110,  83, 107, 110, 112, 112,  14, 110,
#           83, 107, 110,  53, 110, 112, 112,  13, 110,  83, 107, 110,  53, 110,
#          112, 112,  12, 110,  83, 107, 110,  53,   0,   0,   0,   0,   0,   0,
#            0,   0],
#         [  2, 112, 112,  97, 110,  83, 105, 112,  97, 110,  83, 104, 110,  74,
#           22, 112,  12,   6,  97, 110,  83, 104,   0,   0,   0,   0,   0,   0,
#            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#            0,   0],
#         [  2, 112,  24,  49,  85,  84,  13,   8, 112,  13, 110,  82,  85,   6,
#          112,  13, 110,  82,  84, 110,  74,  22,  99,   0,   0,   0,   0,   0,
#            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#            0,   0],
#         [  2, 112, 112,  13, 110,  82,   5,  96,   6,  36,   4, 110, 112,  12,
#          110,  53,   6, 112, 112,  13, 110,  82,   5,  96,   8,  36,   4, 110,
#          112,  12, 110,  53,  22, 112, 112,  13, 110,  82, 101, 110, 112,  12,
#          110,  53]]
# print(tgt2updown(testtgt))
