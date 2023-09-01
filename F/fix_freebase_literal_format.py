import gzip

in_file = "freebase-rdf-latest.gz"
out_file = "freebase-rdf-latest-literal_fixed.gz"

# datatype strings
datatype_string = {}
datatype_string["type.int"] = "<http://www.w3.org/2001/XMLSchema#integer>"
datatype_string["type.float"] = "<http://www.w3.org/2001/XMLSchema#float>"
datatype_string["type.boolean"] = "<http://www.w3.org/2001/XMLSchema#boolean>"

# get the properties with literal object value
type_map = {}
with open("numeric_properties.txt", "r") as f_in:
    for line in f_in:
        line = line.strip()
        pred, type = line.split("\t")
        type_map[pred] = datatype_string[type]

# update literal type line by line
f_in = gzip.open(in_file, "r")
f_out = gzip.open(out_file, "w")
line_num = 0
for line in f_in:
    line_num += 1
    if not line:
        continue
    subj, pred, obj, rest = line.split("\t")
    pred_t = pred[pred.rfind("/")+1:len(pred)-1]
    try:
        datatype_string = type_map[pred_t]
        if "^^" in obj:
          pass
        else:
            if "\"" in obj:
                obj = obj + "^^" + datatype_string
            else:
                obj = "\"" + obj + "\"^^" + datatype_string
            line = "\t".join([subj, pred, obj, rest])
    except:
        pass
    f_out.write(line)
    if line_num % 1000000 == 0:
        print(line_num)

f_in.close()
f_out.close()
