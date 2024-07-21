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