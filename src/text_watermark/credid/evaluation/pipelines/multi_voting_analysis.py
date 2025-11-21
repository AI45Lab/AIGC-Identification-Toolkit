import json 
import os

path ="./my_watermark_result/lm_new_7_10/tiiuae-falcon-7b_1.5_1.5_300_50_250_42_42_10_4_4_1.0_10_-1_200_max_confidence_updated_0.5_huggyllama/llama-7b_1000.json"

with open(path, 'r') as f:
        data = json.load(f)
        
confidence1 = [item[0][2] for item in data['confidence']]
con = [item['confidence'] for item in data['analysis_att_ori']['message_confidence']]
confidence2 =  [item['confidence'] for item in data['analysis_att_gemma']['message_confidence']]
confidence3 =  [item['confidence'] for item in data['analysis_att_llama']['message_confidence']]
# message =   [item[0] for item in data['decoded_message']]
message = [item['decoded_message'] for item in data['analysis_att_ori']['message_confidence']]
count = 0
for i in range(len(message)):
    if (message[i] != [42]) :#& (con[i] > confidence2[i]) & (confidence1[i]>confidence3[i]):
        print(f"  Decoded message: {message[i]}")
        print(f"  ori confidence: {con[i]}")
        print(f"  gemma confidence: {confidence2[i]}")
        print(f"  llama confidence: {confidence3[i]}")
        count += 1
print(count)