import re
import json
from pprint import pprint
import random
import operator
import collections

data_path = '../Data/'
out_path = '../kgat_data_ping/'

# dataset = 'amazon-book'
# dataset = 'yelp2018'
dataset = 'last-fm'

no_valid = 30000
if dataset == 'last-fm':
    no_valid = 100000


def load_data(path, dataset, fname):
    fp = open(path+dataset+'/'+fname, 'r')
    data = []
    for line in fp:
        data.append(line)
    fp.close()

    return data

def load_dict(path, datasest, fname):
    fp = open(path+dataset+'/'+fname, 'r')
    node2id = {}
    for line in fp:
        break
    for line in fp:
        line = re.split(' ', line[:-1])
        tmp_id = line[-1]
        tmp_key = ' '.join(line[:-1])
        node2id[tmp_key] = tmp_id
    fp.close()
    
    return node2id

def load_item_dict(path, datasest, fname):
    fp = open(path+dataset+'/'+fname, 'r')
    node2id = {}
    for line in fp:
        break
        
    cnt = 0
    for line in fp:
        line = re.split(' ', line[:-1])
        if datasest != 'yelp2018':
            tmp_id = line[1]
            tmp_key = line[-1]
        else:
            tmp_id = cnt
            tmp_key = line[0]
            cnt += 1
        node2id[tmp_key] = tmp_id        
    fp.close()
    
    return node2id

# def load_item_dict(path, datasest, fname):
#     fp = open(path+dataset+'/'+fname, 'r')
#     node2id = {}
#     for line in fp:
#         break
#     for line in fp:
#         line = re.split(' ', line[:-1])
#         tmp_id = line[1]
#         tmp_key = line[-1]
#         node2id[tmp_key] = tmp_id
#     fp.close()
    
#     return node2id


print('Processing dataset: {} ...'.format(dataset))
entity2id = load_dict(data_path, dataset, 'entity_list.txt')
id2entity = {entity2id[itm]: itm for itm in entity2id}
user2id = load_dict(data_path, dataset, 'user_list.txt')
id2user = {user2id[itm]: itm for itm in user2id}
item2id = load_item_dict(data_path, dataset, 'item_list.txt')
id2item = {item2id[itm]: itm for itm in item2id}

print('No. of entities: {}, user: {}: item {}'.format(len(entity2id), len(user2id), len(item2id)))
ent2id = {}
for itm in entity2id:
    ent2id[itm] = 0
for itm in user2id:
    ent2id[itm] = 0
cnt = 0
for itm in ent2id:
    ent2id[itm] = cnt
    cnt += 1
print('No. of entities in the whole graph: {}'.format(len(ent2id)))

relations = load_data(data_path, dataset, 'relation_list.txt')
no_rel = len(relations)
rel2id = {'user.item': 0}
for ii in range(1, no_rel):
    tmp = re.split(' ', relations[ii][:-1])
    rel2id[tmp[0]] = int(tmp[1])+1
print('No. of relations: {}'.format(len(rel2id)))

for fname in ['ent2id', 'rel2id']:
    fout = open('{}{}/{}.txt'.format(out_path, dataset, fname), 'w')
    if fname == 'ent2id':
        out = json.dump(ent2id, fout)
    else:
        out = json.dump(rel2id, fout)
    fout.close()

# process kg_final
kg_final = load_data(data_path, dataset, 'kg_final.txt')
print(len(kg_final))
fout = open(out_path+dataset+'/kg_final.txt', 'w')
for ee in kg_final:
    ee = re.split(' ', ee[:-1])
    src = ent2id[id2entity[ee[0]]]
    trg = ent2id[id2entity[ee[2]]]
    ee_new = [str(int(ee[1])+1), str(src), str(trg)]
    out = ' '.join(ee_new)
    fout.write(out)
    fout.write('\n')
fout.close()


# process user-item edges
data_edges = {}
for task in ['train', 'test']:
    data_edges[task] = []
    data = load_data(data_path, dataset, task+'.txt')
    print(task, len(data))
#     fout = open(out_path+dataset+'/'+task+'.txt', 'w')
    for line in data:
        line = re.split(' ', line[:-1])
        src = ent2id[id2user[line[0]]]
        rel = rel2id['user.item']
        for ii in range(1, len(line)):
            if line[ii] == '':
                pass
            else:
                trg = ent2id[id2entity[line[ii]]]
                ee = [str(rel), str(src), str(trg)]
                data_edges[task].append(ee)
#                 out = ' '.join(ee)
#                 fout.write(out)
#                 fout.write('\n')
#     fout.close()
    print(len(data_edges[task]))
    print('Done for {}\n'.format(task))
    
train_edges = data_edges['train']
test_edges = data_edges['test']
random.shuffle(train_edges)
train_edges = train_edges[:-no_valid]
valid_edges = train_edges[-no_valid:]
print('No. of training: {}, valid: {}, test: {}'\
      .format(len(train_edges), len(valid_edges), len(test_edges)))
no_neg = len(valid_edges) + len(test_edges)



# negative sampling
user2item = {}
for task in ['train', 'test']:
    data = load_data(data_path, dataset, task+'.txt')
    print(task, len(data))
    for line in data:
        line = re.split(' ', line[:-1])
        user = line[0]
        if user not in user2item:
            user2item[user] = []
            
        for ii in range(1, len(line)):
            if line[ii] == '':
                pass
            else:
                trg = line[ii]
                user2item[user].append(trg)
# print(len(user2item))

item_id_set = [iid for iid in id2item]
print(len(item_id_set))
user2item_negative = {}
cnt = 0
for user in user2item:    
    negative_set = [itm for itm in item_id_set if itm not in user2item[user]]
    user2item_negative[user] = negative_set
    cnt += 1
    if cnt%5000 == 0:
        print(cnt)

negative_edges = []
cnt = 0
rel = rel2id['user.item']
while 1:
    for user in user2item_negative:
        src = ent2id[id2user[user]]
        if len(user2item_negative[user]) > 0:
            item = random.sample(user2item_negative[user], 1)[0]
            user2item_negative[user].remove(item)
#             print('check', item)
#             print(id2entity)
            trg = ent2id[id2entity[str(item)]]
            ee = [rel, src, trg, 0]
#             print(ee)
            negative_edges.append(ee)
            cnt += 1
            if cnt == no_neg:
                break
            if cnt%10000 == 0:
                print(cnt)
    if cnt == no_neg:
        break
print(len(negative_edges))
random.shuffle(negative_edges)
valid_edges += negative_edges[:no_valid]
test_edges += negative_edges[no_valid:]
print('No. of edges in training: {}, valid: {}, test: {}'\
      .format(len(train_edges), len(valid_edges), len(test_edges)))

# dump edges
fedge = fout = open('{}{}/{}.{}.txt'.format(out_path, dataset, dataset, 'edgelist'), 'w')
tasks = {'train':train_edges, 'valid':valid_edges, 'test':test_edges}
cnt = 0
for task in tasks:
    print(task)
#     print(tasks[task])
    fout = open('{}{}/{}.txt'.format(out_path, dataset, task), 'w')
    for ee in tasks[task]:
        ee = [str(itm) for itm in ee]
        if task != 'train' and len(ee) == 3:
            ee.append('1')
        out = ' '.join(ee)
        fout.write(out)
        fout.write('\n')
        
        if task != 'train' and ee[-1] == '0':
            continue
        else:
            edge = [ee[1], ee[2]]
            eout = ' '.join(edge)
            fedge.write(eout)
            fedge.write('\n')
            cnt += 1
    fout.close()
fedge.close()
print('No. of edges in the edgelist: {}'.format(cnt))

print('Normal terminate.')