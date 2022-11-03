import onnxruntime as ort
import numpy as np
import torch
import onnx
from transformers import AutoTokenizer
from scipy.special import softmax

class InferenceONNX:
    def __init__(self, model_path= 'checkpoints', file_onnx = 'worldcup-model.onnx'):
        onnx_path = f'{model_path}/worldcup-model.onnx'
        self.session = ort.InferenceSession(onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        intent_label = open('data/intent_label.txt', 'r').readlines()
        intent_label = [i.replace('\n', '') for i in intent_label]

        slot_label = open('data/slot_label.txt', 'r').readlines()
        slot_label = [i.replace('\n', '') for i in slot_label]

        self.intent2id = {intent:idx for idx, intent in enumerate(intent_label)}
        self.slot2id = {slot:idx for idx, slot in enumerate(slot_label)}
        self.id2intent = {idx:intent for idx, intent in enumerate(intent_label)}
        self.id2slot = {idx:slot for idx, slot in enumerate(slot_label)}
    
    def inference(self, batch_text): # param `batch_text` can be a string or a list of strings
        if isinstance(batch_text, str):
            batch_text = [batch_text]
        encoded = self.tokenizer(batch_text, return_tensors = 'np', return_offsets_mapping= True, padding = 'longest')
        intent_logits, slot_logits = self.session.run(
            ['intent_logits', 'slot_logits'], 
            {
                'input_ids': encoded["input_ids"],
                'attention_mask': encoded["attention_mask"]
            })

        intent_softmax_batch = softmax(intent_logits, axis=-1)
        intent_id_batch = np.argmax(intent_logits, axis = -1)
        slot_softmax_batch = softmax(slot_logits, axis = -1)
        slot_id_batch= np.argmax(slot_logits, axis = -1)

        numbatch = slot_id_batch.shape[0]
        output_batch = []
        for i in range(numbatch):
            slot_softmax = slot_softmax_batch[i]
            intent_softmax = intent_softmax_batch[i]
            intent_id = intent_id_batch[i]
            slot_id = slot_id_batch[i]
            offset_mapping = encoded['offset_mapping'][i]
            raw_text = batch_text[i]
            output_batch.append(self.get_explicit_output(raw_text, intent_softmax, slot_softmax, intent_id, slot_id, offset_mapping))
        return output_batch

    def get_explicit_output(self, raw_text, intent_softmax, slot_softmax, intent_id, slot_id, offset_mapping):
        intent_label = self.id2intent[intent_id]
        slot_label = [self.id2slot[id] for id in slot_id]
        slot_list = []
        real_tag = None
        for i, tag in enumerate(slot_label):
            if tag.startswith("B-") and real_tag is None:
                start = offset_mapping[i][0]
                start = start + 1 if start != 0 else start # + 1 to offset of token with a space before
                real_tag = tag[2:]
                score = slot_softmax[i][slot_id[i]]
            elif tag.startswith("I-") and tag[2:] == real_tag:
                continue
            elif real_tag is not None:
                end = offset_mapping[i-1][1]
                slot_list.append({
                    'slot_type': real_tag,
                    'score': score,
                    'slot_text':  raw_text[start:end],
                    'start': start,
                    'end': end
                })
                real_tag = None
                if tag.startswith("B-"):
                    start = offset_mapping[i][0]
                    start = start + 1 if start != 0 else start # + 1 to offset of token with a space before
                    real_tag = tag[2:]
                    score = slot_softmax[i][slot_id[i]]
        intent_score = intent_softmax[intent_id]
        return {
            'intent':{
                'intent_type': intent_label,
                'score': intent_score,
            },
            'slot': slot_list,
            'text': raw_text,
        }

if __name__ == '__main__':
    batch_text = "ai sẽ vô địch worldcup năm ni"
    # batch_text = ["đội nga đấu với pháp thì ai là người đi tiếp",
    #     'bò đào nha và pháp đá với nhau thế ai đi tiếp',
    #     'cho tớ biết tỉ số anh và thụy điển vào hôm qua',]
    
    onnx_obj = InferenceONNX(model_path = 'checkpoints', file_onnx = 'model.onnx')
    output = onnx_obj.inference(batch_text)
    print(output)