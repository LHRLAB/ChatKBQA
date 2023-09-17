import json
import os
import argparse

def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)

def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)

def webqsp_evaluate_valid_results(args):
    if args.split == 'dev':
        res = main(args.pred_file, f'data/WebQSP/origin/WebQSP.pdev.json')
    else:
        res = main(args.pred_file, f'data/WebQSP/origin/WebQSP.{args.split}.json')
    dirname = os.path.dirname(args.pred_file)
    filename = os.path.basename(args.pred_file)
    with open (os.path.join(dirname,f'{filename}_final_eval_results_official.txt'),'w') as f:
        f.write(res)
        f.flush()

def FindInList(entry,elist):
    for item in elist:
        if entry == item:
            return True
    return False
            
def CalculatePRF1(goldAnswerList, predAnswerList):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [1.0, 1.0, 1.0, 1]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [0.0, 1.0, 0.0, 1]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif len(predAnswerList)==0:
        return [1.0, 0.0, 0.0, 0]    # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        glist =[x["AnswerArgument"] for x in goldAnswerList]
        plist =predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if FindInList(gentry,plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not FindInList(pentry,glist):
                fp += 1


        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        
        f1 = (2*precision*recall)/(precision+recall)
        
        if tp > 1e-40:
            hit = 1
        else:
            hit = 0
        return [precision, recall, f1, hit]


def main(pred_data, dataset_data):

    goldData = load_json(dataset_data)
    predAnswers = load_json(pred_data)

    PredAnswersById = {}

    for item in predAnswers:
        PredAnswersById[item["QuestionId"]] = item["Answers"]

    total = 0.0
    f1sum = 0.0
    recSum = 0.0
    precSum = 0.0
    hitSum = 0
    numCorrect = 0
    prediction_res = []
    if "Questions" in goldData:
        goldData = goldData["Questions"]
    for entry in goldData:

        skip = True
        for pidx in range(0,len(entry["Parses"])):
            np = entry["Parses"][pidx]
            if np["AnnotatorComment"]["QuestionQuality"] == "Good" and np["AnnotatorComment"]["ParseQuality"] == "Complete":
                skip = False

        if(len(entry["Parses"])==0 or skip):
            continue

        total += 1
    
        id = entry["QuestionId"]
    
        if id not in PredAnswersById:
            print("The problem " + id + " is not in the prediction set")
            print("Continue to evaluate the other entries")
            continue

        if len(entry["Parses"]) == 0:
            print("Empty parses in the gold set. Breaking!!")
            break

        predAnswers = PredAnswersById[id]

        bestf1 = -9999
        bestf1Rec = -9999
        bestf1Prec = -9999
        besthit = 0
        for pidx in range(0,len(entry["Parses"])):
            pidxAnswers = entry["Parses"][pidx]["Answers"]
            prec,rec,f1,hit = CalculatePRF1(pidxAnswers,predAnswers)
            if f1 > bestf1:
                bestf1 = f1
                bestf1Rec = rec
                bestf1Prec = prec
            if hit > besthit:
                besthit = hit

        f1sum += bestf1
        recSum += bestf1Rec
        precSum += bestf1Prec
        hitSum += besthit

        pred = {}
        pred['qid'] = id
        pred['precision'] = bestf1Prec
        pred['recall'] = bestf1Rec
        pred['f1'] = bestf1
        pred['hit'] = besthit
        prediction_res.append(pred)

        if bestf1 == 1.0:
            numCorrect += 1

    print("Number of questions:", int(total))
    print("Average precision over questions: %.3f" % (precSum / total))
    print("Average recall over questions: %.3f" % (recSum / total))
    print("Average f1 over questions (accuracy): %.3f" % (f1sum / total))
    print("F1 of average recall and average precision: %.3f" % (2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total)))
    print("True accuracy (ratio of questions answered exactly correctly): %.3f" % (numCorrect / total))
    print("Hits@1 over questions: %.3f" % (hitSum / total))
    res = f'Number of questions:{int(total)}\n, Average precision over questions: {(precSum / total)}\n, Average recall over questions: {(recSum / total)}\n, Average f1 over questions (accuracy): {(f1sum / total)}\n, F1 of average recall and average precision: {(2 * (recSum / total) * (precSum / total) / (recSum / total + precSum / total))}\n, True accuracy (ratio of questions answered exactly correctly): {(numCorrect / total)}\n, Hits@1 over questions: {(hitSum / total)}'
    dirname = os.path.dirname(pred_data)
    filename = os.path.basename(pred_data)
    dump_json(prediction_res, os.path.join(dirname, f'{filename}_new.json'))
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="split to operate on, can be `test`, `dev` and `train`",
    )
    parser.add_argument(
        "--pred_file", type=str, default=None, help="prediction results file"
    )
    args = parser.parse_args()
    
    webqsp_evaluate_valid_results(args)