import json
import math
from pathlib import Path
from typing import Union

import torch

from task.task_medical.net import NerBaseBert


class Predictor:
    def __init__(self, model_dir, bert_path):
        with open(model_dir.joinpath('class2idx.json'), 'r', encoding='utf-8') as file:
            class2idx = json.load(file)
        self.idx2class = {v: k for k, v in class2idx.items()}
        self.net = NerBaseBert(num_classes=len(class2idx), model_path=bert_path)
        # 模型参数恢复
        self.net.load_state_dict(torch.load(model_dir.joinpath('best.pkl'),
                                            map_location=torch.device('cpu'))['model_state'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 512

    def entity_decode(self, result: list):
        entity_results = []
        for res in result:
            entity_result = {}
            text = ''.join(res[0])
            entity_result['originalText'] = text
            entity_class = [self.idx2class[e] for e in res[1]]
            index = 0
            entities = []
            while index < len(entity_class):
                if entity_class[index][0] == 'B':
                    start_pos = index
                    # 连续实体处理
                    if entity_class[index + 1][0] == 'I':
                        while entity_class[index + 1][0] == 'I' and entity_class[start_pos][2:] == entity_class[
                                                                                                       index + 1][2:]:
                            index += 1
                        end_pos = index + 1
                        entity_span = text[start_pos: end_pos]
                        entities.append({
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'label_type': entity_class[start_pos][2:],
                            'entity_span': entity_span
                        })

                    elif entity_class[index + 1][0] in ['B', 'O']:
                        end_pos = index + 1
                        entity_span = text[start_pos: end_pos]
                        entities.append({
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'label_type': entity_class[start_pos][2:],
                            'entity_span': entity_span
                        })
                    index += 1
                else:
                    index += 1
            entity_result['entities'] = entities
            entity_results.append(entity_result)
        return entity_results

    @torch.no_grad()
    def str_predict(self, text: str):
        self.net.eval().to(device=self.device)
        if len(text) <= self.max_len - 2:
            result = list(self.net.predict(text, device=self.device)[0])
        else:
            bucket_len = len(text) // (math.ceil(len(text) / (self.max_len - 2)))
            start_pos = 0
            tokens, entity_class_idx = [], []
            while start_pos < len(text):
                sub_text = text[start_pos: start_pos + bucket_len]
                if start_pos + bucket_len < len(text):
                    for i in range(len(sub_text) - 1, -1, -1):
                        if sub_text[i] in [',', '.', '，', '。']:
                            sub_text = sub_text[:i]
                            break
                tokens_, entity_class_idx_ = self.net.predict(sub_text, device=self.device)[0]
                tokens.extend(tokens_)
                entity_class_idx.extend(entity_class_idx_)
                start_pos += len(sub_text)
            result = [tokens, entity_class_idx]

        return result

    def predict(self, text: Union[str, list]):
        result = []
        if isinstance(text, str):
            result.append(self.str_predict(text))
        elif isinstance(text, list):
            for t in text:
                result.append(self.str_predict(t))
        return self.entity_decode(result)


if __name__ == '__main__':
    model_dir_ = Path(r'./output/medical')
    bert_path_ = Path(r'C:\Users\du\.cache\huggingface\hub\hub\bert-base-chinese')
    predictor = Predictor(model_dir_, bert_path=bert_path_)
    text1 = "，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。"
    text2 = [
        '，患者2008年9月3日因“腹胀，发现腹部包块”在我院腹科行手术探查，术中见盆腹腔肿物，与肠管及子宫关系密切，遂行“全子宫左附件切除+盆腔肿物切除+右半结肠切除+DIXON术”，术后病理示颗粒细胞瘤，诊断为颗粒细胞瘤IIIC期，术后自2008年11月起行BEP方案化疗共4程，末次化疗时间为2009年3月26日。之后患者定期复查，2015-6-1，复查CT示：髂嵴水平上腹部L5腰椎前见软组织肿块，大小约30MM×45MM，密度欠均匀，边界尚清楚，轻度强化。查肿瘤标志物均正常。于2015-7-6行剖腹探查+膀胱旁肿物切除+骶前肿物切除+肠表面肿物切除术，术程顺利，，术后病理示：膀胱旁肿物及骶前肿物符合颗粒细胞瘤。于2015-7-13、8-14给予泰素240MG+伯尔定600MG化疗2程，过程顺利。出院至今，无发热，无腹痛、腹胀，有脱发，现返院复诊，拟行再次化疗收入院。起病以来，精神、胃纳、睡眠可，大小便正常，体重无明显改变。',
        '，患者于2011年9月29日在我院因“子宫内膜癌II期”在全麻上行“广泛全子宫切除+两侧附件切除+盆腔淋巴结清扫+腹主动脉旁淋巴结活检术”，术中探查见盆腹腔未见腹水，子宫增大，约10*8*7CM，饱满，两侧附件未见异常，盆腔及腹主动脉旁淋巴结未及肿大。术程顺利，，术后病理回报：腹水未见癌；（全子宫+两附件）送检子宫大小为10*6*4CM，宫腔内见菜花样肿物大小为5*4*3CM，灰黄质硬，浸润浅肌层；镜上中至低分化子宫内膜样腺癌，部分呈鳞状分化，浸润子宫浅肌层，未累及宫颈管；右输卵管系膜内见子宫内膜异位；两附件、阴道残端、淋巴结未见癌；，免疫组化：ER（+），PR（-）。，术后诊断：子宫内膜样腺癌IA1期。因肿瘤为中至低分化且大小为5*4*3CM，术后有化疗指征。于2011年10月11日、11月16日行TP（泰素+伯尔定）方案化疗2程，化疗后出现轻度恶心、呕吐，伴脱发，无骨痛及四肢麻木等不适，白细胞最低降至2.7×109/L，未处理可自行升至正常。自发病以来，精神、食欲、睡眠良好，无腹痛及腹胀，无腰酸，大小便正常。体重较下次化疗增加3KG。，既往化疗及肿瘤标志物情况：，化疗药物毒副反应：。',
        '，患者于2010年10月因\\\"上腹痛伴大便习惯改变\\\"外院行肠镜，活检病理示中分化腺癌；，外院B超示：盆腔内2个巨大团块，考虑卵巢来源可能性大；盆腔大量积液。来我院就诊，考虑乙状结肠癌伴不完全梗阻、盆腔肿物，2010-10-25我院行\\\"子宫切除+两附件切除+DIXON术+大网膜切除术开腹恶性肿瘤特殊治疗术\\\"，术中5-FU1000MG肠腔化疗，门静脉5-FU250MG化疗，中人氟安800MG腹腔化疗，，病理回报：乙状结肠中分化腺癌，两侧卵巢转移瘤，中央组淋巴结转移4/15，PT4BN2M1。2010-11-10我院复查CT提示肝S5、6段肝转移。遂于2010-11-16至2011-3-11行FORFIRI方案化疗7程，4程、6程复查CT疗效评价为SD，后于2011-4-6转外科行\\\"肝转移瘤切除术\\\"，，术后病理：符合肠癌肝转移。后于2011-05-20至2011-08-05继续FORFIRI方案化疗4程。后定期复查至2012-02-21我院CT提示右下肺及右上肺结节、左前胸壁结节，考虑转移瘤，两肺多发结节结节状小空洞影未排除转移。考虑肿瘤复发，于2012-03-01始行FOLFOX方案6程，3，程后复查CT疗效评价：SD。，6程后2012-7-10我院CT：肝S6、8、2病灶，可疑转移瘤，建议MR检查。脾内低密度影，可疑转移瘤。，疗效评价：SD。现为进一步诊治入院，患者自觉无不适，胃纳、睡眠可，二便如常。'
    ]

    result1 = predictor.predict(text1)
    print(result1)
    print(len(result1))

    result2 = predictor.predict(text2)
    print(result2)
    print(len(result2))
