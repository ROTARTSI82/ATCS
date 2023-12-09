import csv
import json
import io

with open("messages/index.json", 'r') as fp:
    index = json.load(fp)

master_list = []
for k in index:
    v = index[k]
    name = v if v else k
    try:
        with open(f"messages/c{k}/messages.csv", 'r') as fp:
            dat = fp.read()
            dat = dat.replace("\x00", "NULL")
            data = list(csv.reader(io.StringIO(dat)))
        for msg in data:
            if msg[1] == "Timestamp":
                continue
            master_list.append([msg[1], name, msg[2], msg[3]])
    except:
        print(f"err on {k} {v} (messages/c{k}/messages.csv)")
        raise Exception()

master_list = sorted(master_list, key=lambda x: x[0])
for i in master_list:
    msg = i[2].replace('\n', '\n ')
    print(f" {msg.strip()}")
    # print(f"[newdoc] {i[0][:7]} {i[1]}\n {msg} {i[3]}".rstrip())

